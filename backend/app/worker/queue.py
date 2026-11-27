"""SQLite job queue shared by the API process (enqueue) and the single worker process.

Every function commits its own change before returning so another process sees it.
`resume_cursor` is a JSON column without mutation tracking: always assign a new dict.
"""
from datetime import datetime, timedelta, timezone

from sqlalchemy import case, update
from sqlmodel import Session, select

from app.core.errors import AppError
from app.db.tables import Job

MAX_ATTEMPTS = 4  # 1 run + 3 retries (spec 8: delays 1, 2, 4 s)
ACTIVE_STATUSES = ("queued", "running")


def now() -> datetime:
    """Timezone-aware UTC clock; tests monkeypatch `app.worker.queue.now`."""
    return datetime.now(timezone.utc)


def _get(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id, populate_existing=True)
    if job is None:
        raise AppError("NOT_FOUND", f"Job {job_id} tidak ditemukan", status=404)
    return job


def enqueue(session: Session, type: str, project_id: str, clip_id: str | None = None) -> Job:
    """Add a job; an active (queued/running) job for the same (type, project_id, clip_id) is returned as-is."""
    clip_match = Job.clip_id.is_(None) if clip_id is None else Job.clip_id == clip_id
    existing = session.exec(
        select(Job)
        .where(Job.type == type, Job.project_id == project_id, clip_match, Job.status.in_(ACTIVE_STATUSES))
        .order_by(Job.created_at)
        .execution_options(populate_existing=True)
    ).first()
    if existing is not None:
        return existing
    job = Job(type=type, project_id=project_id, clip_id=clip_id, status="queued", created_at=now())
    session.add(job)
    session.commit()
    return job


def _is_due(job: Job, current: datetime) -> bool:
    """False while the job's retry backoff (`resume_cursor["not_before"]`) is still in the future."""
    raw = (job.resume_cursor or {}).get("not_before")
    return not raw or datetime.fromisoformat(raw) <= current


def claim_next(session: Session, project_id: str | None = None) -> Job | None:
    """Atomically claim the next runnable job: render first, then oldest created_at.

    `project_id` limits the claim to that project's jobs (the synchronous CLI), keeping the same order.
    """
    query = select(Job).where(Job.status == "queued")
    if project_id is not None:
        query = query.where(Job.project_id == project_id)
    candidates = session.exec(
        query
        .order_by(case((Job.type == "render", 0), else_=1), Job.created_at)
        .execution_options(populate_existing=True)
    ).all()
    current = now()
    for cand in candidates:
        if not _is_due(cand, current):
            continue
        result = session.exec(
            update(Job)
            .where(Job.id == cand.id, Job.status == "queued")
            .values(status="running", started_at=current, attempts=Job.attempts + 1)
            .execution_options(synchronize_session=False)
        )
        session.commit()
        if result.rowcount == 1:
            return _get(session, cand.id)
    return None


def has_queued_render(session: Session) -> bool:
    """True if a queued render job is claimable now (backed-off renders are ignored)."""
    renders = session.exec(
        select(Job).where(Job.status == "queued", Job.type == "render")
        .execution_options(populate_existing=True)
    ).all()
    current = now()
    return any(_is_due(job, current) for job in renders)


def recover_stale(session: Session) -> int:
    """Return `running` jobs left by a dead worker to `queued`, keeping stage and resume_cursor.

    A crash is not an error, so the attempt consumed by the interrupted claim is given back (floor 0).
    """
    result = session.exec(
        update(Job).where(Job.status == "running")
        .values(status="queued", attempts=case((Job.attempts > 0, Job.attempts - 1), else_=0))
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return result.rowcount


def requeue(session: Session, job_id: str) -> None:
    """Put a yielding `running` job back to `queued` without consuming an attempt."""
    session.exec(
        update(Job).where(Job.id == job_id, Job.status == "running")
        .values(status="queued", attempts=Job.attempts - 1)
        .execution_options(synchronize_session=False)
    )
    session.commit()


def set_progress(session: Session, job_id: str, stage: str, progress: float, cursor: dict | None = None) -> None:
    """Update stage/progress; `cursor=None` keeps the existing resume_cursor."""
    job = _get(session, job_id)
    job.stage = stage
    job.progress = progress
    if cursor is not None:
        job.resume_cursor = dict(cursor)
    session.add(job)
    session.commit()


def finish_job(session: Session, job_id: str) -> None:
    job = _get(session, job_id)
    job.status = "succeeded"
    job.progress = 1.0
    job.finished_at = now()
    session.add(job)
    session.commit()


def fail_job(session: Session, job_id: str, err: AppError) -> None:
    """Retryable errors are retried up to 3 times (1, 2, 4 s); the 4th failed attempt is `failed`."""
    job = _get(session, job_id)
    current = now()
    if err.retryable and job.attempts < MAX_ATTEMPTS:
        delay = 2 ** max(job.attempts - 1, 0)
        job.status = "queued"
        job.resume_cursor = {**(job.resume_cursor or {}),
                             "not_before": (current + timedelta(seconds=delay)).isoformat()}
    else:
        job.status = "failed"
        job.error_code = err.code
        job.error_message = err.message
        job.finished_at = current
    session.add(job)
    session.commit()
