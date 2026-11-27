"""The single worker process: `python -m app.worker.main`.

Claims one job at a time (render before process) and never dies on a job error.
"""
import logging
import time
from collections.abc import Callable

from sqlalchemy import Engine, update
from sqlmodel import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db.engine import get_engine, init_db
from app.db.tables import Clip, utcnow
from app.worker import queue
from app.worker.process import ProcessDeps, run_process_job, to_app_error
from app.worker.render_job import run_render_job

log = logging.getLogger("kliplah.worker")

POLL_INTERVAL = 1.5  # seconds between polls of an empty queue


def _dispatch(engine: Engine, settings: Settings, job_id: str, job_type: str,
              process_deps: ProcessDeps | None = None) -> None:
    if job_type == "process":
        outcome = run_process_job(engine, job_id, process_deps or ProcessDeps(settings=settings))
        if outcome == "yield":
            with Session(engine) as s:
                queue.requeue(s, job_id)
        log.info("job %s (process) selesai: %s", job_id, outcome)
    elif job_type == "render":
        run_render_job(engine, job_id, settings=settings)
        log.info("job %s (render) selesai", job_id)
    else:
        raise AppError("UNKNOWN", f"Jenis job '{job_type}' tidak dikenal.")


def run_next(engine: Engine, settings: Settings, process_deps: ProcessDeps | None = None,
             project_id: str | None = None) -> bool:
    """Claim and run one job (of `project_id` only, when given); False when nothing was runnable."""
    with Session(engine) as s:
        job = queue.claim_next(s, project_id)
        if job is None:
            return False
        job_id, job_type, attempt = job.id, job.type, job.attempts
    log.info("job %s (%s) mulai, percobaan ke-%d", job_id, job_type, attempt)
    try:
        _dispatch(engine, settings, job_id, job_type, process_deps)
    except Exception as e:
        log.exception("job %s (%s) gagal", job_id, job_type)
        try:
            with Session(engine) as s:
                queue.fail_job(s, job_id, to_app_error(e))
        except Exception:
            log.exception("job %s tidak bisa ditandai gagal", job_id)
    return True


def recover_startup(session: Session) -> tuple[int, int]:
    """Return jobs left `running` to `queued` and clips left `rendering` to `approved`: (jobs, clips).

    Only safe when no worker is running: it cannot tell a crashed job from one another process is working on.
    """
    recovered = queue.recover_stale(session)
    stuck = session.exec(update(Clip).where(Clip.status == "rendering")
                         .values(status="approved", updated_at=utcnow())
                         .execution_options(synchronize_session=False)).rowcount
    session.commit()
    return recovered, stuck


def worker_main(settings: Settings, *, once: bool = False, sleep: Callable[[float], None] = time.sleep,
                process_deps: ProcessDeps | None = None, recover: bool = True,
                project_id: str | None = None) -> None:
    """Run jobs forever (or one poll with `once`).

    `process_deps` replaces the pipeline of process jobs; `recover=False` skips start-up recovery and
    `project_id` limits claims to one project (both used by the synchronous CLI).
    """
    engine = get_engine(settings)
    init_db(engine)
    if recover:
        # only one worker exists, so a job `running` / clip `rendering` at start-up was left by a crash or Ctrl+C
        with Session(engine) as s:
            recovered, stuck = recover_startup(s)
        if recovered or stuck:
            log.info("%d job dan %d klip tertinggal dikembalikan ke antrean", recovered, stuck)
    while True:
        if not run_next(engine, settings, process_deps, project_id):
            sleep(POLL_INTERVAL)
        if once:
            return

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("worker mulai")
    try:
        worker_main(get_settings())
    except KeyboardInterrupt:
        log.info("worker berhenti")
