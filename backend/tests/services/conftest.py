import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.db.engine import get_engine, init_db
from app.db.tables import Clip, Project
from app.domain.clip_doc import subtitles_from_m1
from app.domain.style import default_style
from app.pipeline.subtitles import group_lines
from tests.helpers import make_transcript


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


@pytest.fixture
def ready_project(engine, settings):
    """A `ready` project (whisper model `small`) with a 60 s transcript on disk: 10 sentences of 6 s."""
    t = make_transcript(10, each=6.0)
    with Session(engine, expire_on_commit=False) as s:
        p = Project(status="ready", whisper_model="small", style=default_style(), duration_sec=t.duration,
                    last_completed_stage="prepare")
        s.add(p)
        s.commit()
    tdir = settings.data_dir / "projects" / p.id / "transcripts" / "small"
    tdir.mkdir(parents=True)
    (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
    return p, t


@pytest.fixture
def make_clip(engine, ready_project):
    """Insert a single-piece clip of the ready project with subtitles from its transcript."""
    project, t = ready_project

    def make(start: float = 12.0, end: float = 24.0, status: str = "proposed", rank: int = 1, **fields) -> Clip:
        subs = subtitles_from_m1(group_lines(t.words, start, end, max_words=4))
        fields.setdefault("layout_segments", [{"start": 0.0, "end": end - start, "layout": "fit_blur",
                                               "params": {}, "piece_index": 0, "user_override": False}])
        clip = Clip(project_id=project.id, origin="ai", rank=rank, score=70, status=status,
                    pieces=[{"project_id": project.id, "start": start, "end": end}], subtitles=subs, **fields)
        with Session(engine, expire_on_commit=False) as s:
            s.add(clip)
            s.commit()
        return clip

    return make
