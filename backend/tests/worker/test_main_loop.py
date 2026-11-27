import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.db.engine import get_engine, init_db
from app.db.tables import Clip, Job, Project
from app.worker import main, queue
from app.worker.render_job import run_render_job
from tests.worker.test_render_job import FakeRender, _setup


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


def _enqueue(engine, type_: str) -> str:
    with Session(engine) as s:
        p = Project()
        s.add(p)
        s.commit()
        return queue.enqueue(s, type_, p.id).id


def _job(engine, jid) -> Job:
    with Session(engine) as s:
        return s.get(Job, jid)


def test_main_loop_requeues_yielded_process_without_counting_attempt(engine, settings, monkeypatch):
    jid = _enqueue(engine, "process")
    calls = []

    def fake_process(eng, job_id, deps):
        calls.append((job_id, deps.settings))
        assert _job(engine, job_id).attempts == 1
        return "yield"

    monkeypatch.setattr(main, "run_process_job", fake_process)
    main.worker_main(settings, once=True, sleep=lambda s: pytest.fail("must not sleep after a job"))

    assert calls == [(jid, settings)]
    job = _job(engine, jid)
    assert job.status == "queued" and job.attempts == 0


def test_main_loop_dispatches_render_job(engine, settings, monkeypatch):
    jid = _enqueue(engine, "render")
    calls = []

    def fake_render(eng, job_id, *, settings):
        calls.append((job_id, settings))
        with Session(eng) as s:
            queue.finish_job(s, job_id)

    monkeypatch.setattr(main, "run_render_job", fake_render)
    main.worker_main(settings, once=True)

    assert calls == [(jid, settings)]
    assert _job(engine, jid).status == "succeeded"


def test_main_loop_fails_unknown_job_type(engine, settings):
    jid = _enqueue(engine, "bogus")
    main.worker_main(settings, once=True)

    job = _job(engine, jid)
    assert job.status == "failed" and job.error_code == "UNKNOWN"


def test_main_loop_survives_handler_exception(engine, settings, monkeypatch):
    first, second = _enqueue(engine, "process"), _enqueue(engine, "process")

    def boom(eng, job_id, deps):
        raise RuntimeError("meledak")

    monkeypatch.setattr(main, "run_process_job", boom)
    main.worker_main(settings, once=True)
    job = _job(engine, first)
    assert job.status == "failed" and job.error_code == "UNKNOWN" and "meledak" in job.error_message

    main.worker_main(settings, once=True)
    assert _job(engine, second).status == "failed"


def test_main_loop_sleeps_when_idle_and_recovers_stale(engine, settings):
    jid = _enqueue(engine, "bogus")
    with Session(engine) as s:
        queue.claim_next(s)  # left `running` by a dead worker
    slept = []

    main.worker_main(settings, once=True, sleep=slept.append)  # recovers, then fails the bogus job
    assert _job(engine, jid).status == "failed"

    main.worker_main(settings, once=True, sleep=slept.append)
    assert slept == [main.POLL_INTERVAL] and main.POLL_INTERVAL == 1.5


def test_startup_recovery_resets_stuck_rendering_clip(engine, settings, monkeypatch):
    _, cid, jid = _setup(engine, settings)  # job claimed -> `running`
    with Session(engine) as s:  # a crash / Ctrl+C mid-render left the clip `rendering`
        clip = s.get(Clip, cid)
        clip.status = "rendering"
        s.add(clip)
        s.commit()
    assert _job(engine, jid).status == "running"
    fake = FakeRender()
    monkeypatch.setattr(main, "run_render_job",
                        lambda eng, job_id, *, settings: run_render_job(eng, job_id, render=fake, settings=settings))

    main.worker_main(settings, once=True)

    assert len(fake.calls) == 1
    with Session(engine) as s:
        assert s.get(Clip, cid).status == "rendered"
    job = _job(engine, jid)
    assert job.status == "succeeded" and job.attempts == 1


def test_worker_main_without_recovery_leaves_running_jobs(engine, settings):
    jid = _enqueue(engine, "bogus")
    with Session(engine) as s:
        queue.claim_next(s)  # `running` in another worker
    slept = []
    main.worker_main(settings, once=True, sleep=slept.append, recover=False)
    job = _job(engine, jid)
    assert job.status == "running" and job.attempts == 1 and slept == [main.POLL_INTERVAL]


def test_worker_main_claims_only_the_given_project(engine, settings, monkeypatch):
    mine, other = _enqueue(engine, "process"), _enqueue(engine, "process")
    calls = []

    def fake_process(eng, job_id, deps):
        calls.append(job_id)
        with Session(eng) as s:
            queue.finish_job(s, job_id)
        return "done"

    monkeypatch.setattr(main, "run_process_job", fake_process)
    main.worker_main(settings, once=True, recover=False, project_id=_job(engine, mine).project_id,
                     sleep=lambda s: None)
    main.worker_main(settings, once=True, recover=False, project_id=_job(engine, mine).project_id,
                     sleep=lambda s: None)
    assert calls == [mine] and _job(engine, other).status == "queued"


def test_recover_startup_counts_jobs_and_clips(engine, settings):
    _, cid, jid = _setup(engine, settings)  # render job claimed -> `running`
    with Session(engine) as s:
        clip = s.get(Clip, cid)
        clip.status = "rendering"
        s.add(clip)
        s.commit()
        assert main.recover_startup(s) == (1, 1)
    assert _job(engine, jid).status == "queued"
    with Session(engine) as s:
        assert s.get(Clip, cid).status == "approved"
