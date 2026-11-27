from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.db.engine import get_engine, init_db
from app.db.tables import Job, Project
from app.worker import queue

T0 = datetime(2026, 11, 26, 10, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch):
    c = Clock(T0)
    monkeypatch.setattr(queue, "now", c)
    return c


@pytest.fixture
def engine(tmp_path):
    eng = get_engine(Settings(data_dir=tmp_path))
    init_db(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture
def project_ids(session):
    projects = [Project(), Project()]
    session.add_all(projects)
    session.commit()
    return [p.id for p in projects]


def _fresh(engine, job_id: str) -> Job:
    """Read the row through a separate session, as another process would."""
    with Session(engine) as s:
        return s.get(Job, job_id)


def test_now_is_timezone_aware_utc():
    assert queue.now().utcoffset() == timedelta(0)


def test_render_jobs_claimed_before_older_process_jobs(session, project_ids, clock):
    p1, p2 = project_ids
    old_process = queue.enqueue(session, "process", p1)
    clock.advance(10)
    render = queue.enqueue(session, "render", p2)
    clock.advance(10)
    newer_process = queue.enqueue(session, "process", p2)

    first = queue.claim_next(session)
    assert first.id == render.id
    assert first.status == "running"
    assert first.attempts == 1
    assert first.started_at == clock()
    assert queue.claim_next(session).id == old_process.id
    assert queue.claim_next(session).id == newer_process.id
    assert queue.claim_next(session) is None


def test_has_queued_render(session, project_ids, clock):
    p1, _ = project_ids
    assert queue.has_queued_render(session) is False
    queue.enqueue(session, "process", p1)
    assert queue.has_queued_render(session) is False
    queue.enqueue(session, "render", p1)
    assert queue.has_queued_render(session) is True
    queue.claim_next(session)  # claims the render first
    assert queue.has_queued_render(session) is False


def test_enqueue_dedupes_active_job(session, project_ids, clock):
    p1, p2 = project_ids
    a = queue.enqueue(session, "process", p1)
    assert queue.enqueue(session, "process", p1).id == a.id
    assert queue.enqueue(session, "process", p2).id != a.id
    assert queue.enqueue(session, "render", p1).id != a.id

    while queue.claim_next(session) is not None:  # everything running; dedupe covers running too
        pass
    assert queue.enqueue(session, "process", p1).id == a.id

    queue.finish_job(session, a.id)
    assert queue.enqueue(session, "process", p1).id != a.id


def test_recover_stale_running_jobs_resume_from_stage(engine, session, project_ids, clock):
    p1, _ = project_ids
    job = queue.enqueue(session, "process", p1)
    queue.claim_next(session)
    queue.set_progress(session, job.id, "transcribe", 0.4, {"chunk": 3})
    assert _fresh(engine, job.id).status == "running"

    assert queue.recover_stale(session) == 1
    row = _fresh(engine, job.id)
    assert row.status == "queued"
    assert row.stage == "transcribe"
    assert row.resume_cursor == {"chunk": 3}

    claimed = queue.claim_next(session)
    assert claimed.id == job.id
    assert claimed.stage == "transcribe"
    assert claimed.resume_cursor == {"chunk": 3}


def test_recover_stale_does_not_consume_attempt(engine, session, project_ids, clock):
    p1, _ = project_ids
    job = queue.enqueue(session, "process", p1)
    queue.claim_next(session)
    assert _fresh(engine, job.id).attempts == 1
    queue.recover_stale(session)  # worker crashed mid-run
    assert _fresh(engine, job.id).attempts == 0
    assert queue.claim_next(session).attempts == 1


def test_has_queued_render_ignores_backed_off_render(session, project_ids, clock):
    p1, _ = project_ids
    render = queue.enqueue(session, "render", p1)
    queue.claim_next(session)
    queue.fail_job(session, render.id, AppError("RENDER_STALLED", "Render macet", retryable=True))
    assert queue.has_queued_render(session) is False  # not_before is +1 s
    clock.advance(1)
    assert queue.has_queued_render(session) is True


def test_set_progress_without_cursor_keeps_existing_cursor(engine, session, project_ids, clock):
    p1, _ = project_ids
    job = queue.enqueue(session, "process", p1)
    queue.claim_next(session)
    queue.set_progress(session, job.id, "transcribe", 0.2, {"chunk": 1})
    queue.set_progress(session, job.id, "transcribe", 0.3)
    row = _fresh(engine, job.id)
    assert row.stage == "transcribe"
    assert row.progress == pytest.approx(0.3)
    assert row.resume_cursor == {"chunk": 1}


def test_finish_job(engine, session, project_ids, clock):
    p1, _ = project_ids
    job = queue.enqueue(session, "process", p1)
    queue.claim_next(session)
    clock.advance(5)
    queue.finish_job(session, job.id)
    row = _fresh(engine, job.id)
    assert row.status == "succeeded"
    assert row.progress == 1.0
    assert row.finished_at == clock()


def test_retryable_error_requeues_with_backoff_then_fails_after_3_retries(engine, session, project_ids, clock):
    p1, _ = project_ids
    job = queue.enqueue(session, "process", p1)
    err = AppError("OLLAMA_UNAVAILABLE", "Ollama tidak merespons", retryable=True)

    # attempt 1 -> queued, not_before = +1 s, existing cursor keys kept
    queue.claim_next(session)
    queue.set_progress(session, job.id, "analyze", 0.5, {"window": 2})
    queue.fail_job(session, job.id, err)
    row = _fresh(engine, job.id)
    assert row.status == "queued"
    assert datetime.fromisoformat(row.resume_cursor["not_before"]) - clock() == timedelta(seconds=1)
    assert row.resume_cursor["window"] == 2
    assert queue.claim_next(session) is None  # backoff honoured
    clock.advance(1)

    # attempt 2 -> queued, not_before = +2 s
    assert queue.claim_next(session).id == job.id
    queue.fail_job(session, job.id, err)
    row = _fresh(engine, job.id)
    assert row.status == "queued"
    assert datetime.fromisoformat(row.resume_cursor["not_before"]) - clock() == timedelta(seconds=2)
    clock.advance(1.5)
    assert queue.claim_next(session) is None
    clock.advance(0.5)

    # attempt 3 -> queued, not_before = +4 s
    assert queue.claim_next(session).id == job.id
    queue.fail_job(session, job.id, err)
    row = _fresh(engine, job.id)
    assert row.status == "queued"
    assert datetime.fromisoformat(row.resume_cursor["not_before"]) - clock() == timedelta(seconds=4)
    clock.advance(3.5)
    assert queue.claim_next(session) is None
    clock.advance(0.5)

    # attempt 4 -> failed
    assert queue.claim_next(session).id == job.id
    queue.fail_job(session, job.id, err)
    row = _fresh(engine, job.id)
    assert row.status == "failed"
    assert row.attempts == 4
    assert row.error_code == "OLLAMA_UNAVAILABLE"
    assert row.error_message == "Ollama tidak merespons"
    assert row.finished_at == clock()
    clock.advance(60)
    assert queue.claim_next(session) is None


def test_backoff_skips_only_delayed_job(session, project_ids, clock):
    p1, p2 = project_ids
    delayed = queue.enqueue(session, "process", p1)
    clock.advance(1)
    other = queue.enqueue(session, "process", p2)
    queue.claim_next(session)
    queue.fail_job(session, delayed.id, AppError("UNKNOWN", "gagal", retryable=True))
    assert queue.claim_next(session).id == other.id


def test_non_retryable_error_fails_immediately(engine, session, project_ids, clock):
    p1, _ = project_ids
    job = queue.enqueue(session, "process", p1)
    queue.claim_next(session)
    queue.fail_job(session, job.id, AppError("VIDEO_UNAVAILABLE", "Video tidak tersedia"))
    row = _fresh(engine, job.id)
    assert row.status == "failed"
    assert row.attempts == 1
    assert row.error_code == "VIDEO_UNAVAILABLE"
    assert row.error_message == "Video tidak tersedia"
    assert queue.claim_next(session) is None


def test_requeue_does_not_consume_attempt(engine, session, project_ids, clock):
    p1, _ = project_ids
    job = queue.enqueue(session, "process", p1)
    queue.claim_next(session)
    queue.set_progress(session, job.id, "analyze", 0.6, {"window": 4})
    queue.requeue(session, job.id)
    row = _fresh(engine, job.id)
    assert row.status == "queued"
    assert row.attempts == 0
    assert row.stage == "analyze"
    assert row.resume_cursor == {"window": 4}
    assert queue.claim_next(session).attempts == 1


def test_claim_is_atomic_across_sessions(engine, session, project_ids, clock):
    p1, _ = project_ids
    job = queue.enqueue(session, "process", p1)
    with Session(engine, expire_on_commit=False) as other:
        assert queue.claim_next(other).id == job.id
    assert queue.claim_next(session) is None
    assert _fresh(engine, job.id).attempts == 1


def test_claim_next_filters_by_project(session, project_ids, clock):
    p1, p2 = project_ids
    other_render = queue.enqueue(session, "render", p2)
    clock.advance(1)
    process = queue.enqueue(session, "process", p1)
    clock.advance(1)
    render = queue.enqueue(session, "render", p1)

    assert queue.claim_next(session, p1).id == render.id  # render first, within the project
    assert queue.claim_next(session, p1).id == process.id
    assert queue.claim_next(session, p1) is None
    assert _fresh(session.get_bind(), other_render.id).status == "queued"
    assert queue.claim_next(session).id == other_render.id
