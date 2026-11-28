import pytest
from sqlmodel import Session, select

from app.core.errors import AppError
from app.db.tables import Clip, Job, Project
from app.domain.style import deep_merge, default_style
from app.services import projects, styles

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def _jobs(engine, project_id=None) -> list[Job]:
    with Session(engine) as s:
        q = select(Job)
        if project_id is not None:
            q = q.where(Job.project_id == project_id)
        return list(s.exec(q).all())


def test_create_project_copies_default_template_style_and_enqueues(engine, settings):
    p = projects.create_project(engine, url=URL, settings=settings)
    default = next(t for t in styles.list_templates(engine) if t.is_default)
    assert p.style == default.style == default_style()
    assert p.style_template_id == default.id
    assert p.source_type == "youtube" and p.youtube_url == URL and p.video_id == "dQw4w9WgXcQ"
    assert p.status == "queued" and p.moment_source == "claude_manual" and p.max_clips == 8
    assert p.whisper_model == settings.whisper_model
    assert len(p.id) == 32
    jobs = _jobs(engine, p.id)
    assert [(j.type, j.status, j.clip_id) for j in jobs] == [("process", "queued", None)]


def test_create_project_copies_chosen_template(engine, settings):
    t = styles.create_template(engine, "Kuning", deep_merge(default_style(), {"subtitle": {"color": "#FFE14D"}}))
    p = projects.create_project(engine, url=URL, style_template_id=t.id, settings=settings)
    assert p.style == t.style and p.style_template_id == t.id
    with pytest.raises(AppError) as e:
        projects.create_project(engine, url=URL, style_template_id="tidak-ada", settings=settings)
    assert e.value.code == "NOT_FOUND" and e.value.status == 404


def test_create_project_invalid_url(engine, settings):
    with pytest.raises(AppError) as e:
        projects.create_project(engine, url="https://example.com/video", settings=settings)
    assert e.value.code == "INVALID_URL" and e.value.status == 400
    assert _jobs(engine) == []


def test_create_project_needs_exactly_one_source(engine, settings, tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    for kwargs in ({}, {"url": URL, "upload_path": f}):
        with pytest.raises(AppError) as e:
            projects.create_project(engine, settings=settings, **kwargs)
        assert e.value.code == "VALIDATION" and e.value.status == 422


def test_profile_defaults_fill_durations(engine, settings):
    p = projects.create_project(engine, url=URL, content_type="stream", settings=settings)
    assert (p.min_clip_sec, p.max_clip_sec) == (15, 35)
    q = projects.create_project(engine, url=URL, content_type="stream", min_clip_sec=20, max_clip_sec=40,
                                settings=settings)
    assert (q.min_clip_sec, q.max_clip_sec) == (20, 40)
    with pytest.raises(AppError) as e:
        projects.create_project(engine, url=URL, content_type="opera", settings=settings)
    assert e.value.code == "VALIDATION" and e.value.status == 422


def test_create_project_uses_app_setting_default_content_type(engine, settings):
    from app.db.engine import session_scope
    from app.services.settings_store import put_app_settings

    with session_scope(engine) as s:
        put_app_settings(s, {"content_type": "stream"}, settings=settings)
    p = projects.create_project(engine, url=URL, settings=settings)
    assert (p.content_type, p.min_clip_sec, p.max_clip_sec) == ("stream", 15, 35)
    explicit = projects.create_project(engine, url=URL, content_type="podcast", settings=settings)
    assert (explicit.content_type, explicit.min_clip_sec, explicit.max_clip_sec) == ("podcast", 30, 90)


def test_create_project_upload_moves_file(engine, settings, tmp_path):
    src = tmp_path / "incoming" / "tmp123.part"
    src.parent.mkdir()
    src.write_bytes(b"video")
    p = projects.create_project(engine, upload_path=src, original_filename="rekaman saya.mp4", settings=settings)
    dest = settings.data_dir / "uploads" / p.id / "rekaman saya.mp4"
    assert dest.read_bytes() == b"video" and not src.exists()
    assert p.source_type == "upload" and p.original_filename == "rekaman saya.mp4" and p.ranges is None
    assert [j.type for j in _jobs(engine, p.id)] == ["process"]


def test_list_and_get_project(engine, settings):
    a = projects.create_project(engine, url=URL, settings=settings)
    b = projects.create_project(engine, url=URL, settings=settings)
    assert [p.id for p in projects.list_projects(engine)] == [b.id, a.id]
    project, job = projects.get_project(engine, a.id)
    assert project.id == a.id and job is not None and job.type == "process"
    with pytest.raises(AppError) as e:
        projects.get_project(engine, "tidak-ada")
    assert e.value.status == 404


def test_delete_project_removes_rows_and_folders(engine, settings, ready_project, make_clip):
    project, _ = ready_project
    make_clip()
    with Session(engine) as s:
        s.add(Job(type="process", project_id=project.id, status="succeeded"))
        s.commit()
    up = settings.data_dir / "uploads" / project.id
    up.mkdir(parents=True)
    projects.delete_project(engine, project.id, settings=settings)
    with Session(engine) as s:
        assert s.get(Project, project.id) is None
        assert s.exec(select(Clip)).all() == [] and s.exec(select(Job)).all() == []
    assert not (settings.data_dir / "projects" / project.id).exists() and not up.exists()


def test_delete_project_refused_while_job_running(engine, settings, ready_project):
    project, _ = ready_project
    with Session(engine) as s:
        s.add(Job(type="process", project_id=project.id, status="running"))
        s.commit()
    with pytest.raises(AppError) as e:
        projects.delete_project(engine, project.id, settings=settings)
    assert e.value.code == "INVALID_STATE" and e.value.status == 409


def test_retry_project_only_failed(engine, settings, ready_project):
    project, _ = ready_project
    with pytest.raises(AppError) as e:
        projects.retry_project(engine, project.id)
    assert e.value.code == "INVALID_STATE"
    with Session(engine) as s:
        row = s.get(Project, project.id)
        row.status = "failed"
        s.add(row)
        s.commit()
    out = projects.retry_project(engine, project.id)
    assert out.status == "queued"
    assert [j.type for j in _jobs(engine, project.id)] == ["process"]


def test_update_project_fields_and_style(engine, ready_project, make_clip):
    project, _ = ready_project
    clip = make_clip(status="rendered")
    new_style = deep_merge(default_style(), {"hook": {"size": 80}})
    out = projects.update_project(engine, project.id, {"style": new_style, "keep_source": True,
                                                       "content_type": "podcast",
                                                       "facecam_box": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}})
    assert out.style == new_style and out.keep_source and out.content_type == "podcast"
    assert out.facecam_box == {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    with Session(engine) as s:
        assert s.get(Clip, clip.id).needs_rerender is True
    for bad in ({"title": "x"}, {"keep_source": "ya"}, {"facecam_box": {"x": 2, "y": 0, "w": 0.1, "h": 0.1}},
                {"content_type": "opera"}):
        with pytest.raises(AppError) as e:
            projects.update_project(engine, project.id, bad)
        assert e.value.code == "VALIDATION" and e.value.status == 422


def test_partial_project_style_patch_keeps_other_fields(engine, ready_project):
    project, _ = ready_project
    custom = deep_merge(default_style(), {"subtitle": {"color": "#FFE14D"}, "hook": {"size": 80}})
    projects.update_project(engine, project.id, {"style": custom})
    out = projects.update_project(engine, project.id, {"style": {"subtitle": {"size": 120}}})
    assert out.style == deep_merge(custom, {"subtitle": {"size": 120}})
    assert styles.set_project_style(engine, project.id, {"hook": {"size": 70}}).style["subtitle"]["size"] == 120


def test_update_project_is_one_transaction(engine, ready_project):
    project, _ = ready_project
    with pytest.raises(AppError):
        projects.update_project(engine, project.id, {"keep_source": True, "style": {"subtitle": {"size": -1}}})
    with Session(engine) as s:
        assert s.get(Project, project.id).keep_source is False


def test_facecam_box_change_flags_rendered_facecam_clips(engine, ready_project, make_clip):
    project, _ = ready_project
    facecam = make_clip(status="rendered", rank=1, layout_segments=[
        {"start": 0.0, "end": 12.0, "layout": "facecam_full", "params": {}, "piece_index": 0,
         "user_override": False}])
    plain = make_clip(status="rendered", rank=2)
    projects.update_project(engine, project.id, {"facecam_box": {"x": 0.0, "y": 0.0, "w": 0.3, "h": 0.3}})
    with Session(engine) as s:
        assert s.get(Clip, facecam.id).needs_rerender is True
        assert s.get(Clip, plain.id).needs_rerender is False


def test_probe_estimates_and_warns_for_long_video(engine, settings, monkeypatch):
    from app.core.models import VideoMeta

    meta = VideoMeta(
        source_type="youtube", project_id="vid1", title="Stream 8 Jam", channel="Tara",
        duration=31380.0, video_id="vid1", heatmap=[], chapters=[{"start": 0.0, "end": 600.0, "title": "Intro"}],
        subtitles={"manual": [], "auto": ["id"]}, has_live_chat=True,
    )
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta)

    full = projects.probe(engine, URL, settings=settings)
    assert full["title"] == "Stream 8 Jam" and full["channel"] == "Tara" and full["duration"] == 31380.0
    assert full["has_live_chat"] is True and full["has_heatmap"] is False
    assert full["transcript_source_auto"] == "whisper"
    assert full["warnings"] and any("2 jam" in w for w in full["warnings"])
    assert full["estimate"]["disk_gb"] > 40.0

    ranged = projects.probe(engine, URL, ranges=[[1800, 3600]], settings=settings)
    assert ranged["warnings"] == []
    assert ranged["estimate"]["process_minutes"] == 37
    assert ranged["estimate"]["disk_gb"] == 1.1


def test_probe_estimate_includes_encode_time(engine, settings, monkeypatch):
    from app.core.models import VideoMeta

    assert projects.ENCODE_FACTOR == pytest.approx(0.35)
    meta = VideoMeta(
        source_type="youtube", project_id="vid2", title="Stream", channel="Ch",
        duration=10800.0, video_id="vid2", subtitles={"manual": ["id"], "auto": []}, has_live_chat=False,
    )
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta)

    # 30 minutes (1800 s) with youtube_manual (no whisper): 60 s download + 630 s encode = 690 s -> 12 min, 1.1 GB
    ranged = projects.probe(engine, URL, ranges=[[0, 1800]], settings=settings)
    assert ranged["estimate"]["process_minutes"] == 12
    assert ranged["estimate"]["disk_gb"] == 1.1

    # Full video keeps 1.5 MB/s and adds no encode factor (10800 s -> 360 s download = 6 min)
    full = projects.probe(engine, URL, settings=settings)
    assert full["estimate"]["process_minutes"] == 6
    assert full["estimate"]["disk_gb"] == 16.1
    assert set(full["estimate"]["breakdown"]) == {
        "download_min", "encode_min", "transcribe_min", "chat_min", "translate_min", "llm_min",
    }
    assert full["estimate"]["process_minutes"] == sum(full["estimate"]["breakdown"].values())


def test_probe_breakdown_stream_ollama_30m_range_and_profiles(engine, settings, monkeypatch):
    from app.core.models import VideoMeta
    from app.db.engine import session_scope
    from app.services.settings_store import put_app_settings

    assert projects.WHISPER_FACTOR == pytest.approx(0.85)
    assert projects.CHAT_SEC_PER_HOUR == pytest.approx(50.0)
    assert projects.LLM_SEC_PER_WINDOW == pytest.approx(480.0)
    assert projects.LLM_SEC_PER_CLIP == pytest.approx(25.0)

    meta = VideoMeta(
        source_type="youtube", project_id="vid_tara", title="Tara Arts GTA", channel="Tara",
        duration=31380.0, video_id="vid_tara", subtitles={"manual": [], "auto": ["id"]},
        has_live_chat=True,
    )
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta)

    with session_scope(engine) as s:
        put_app_settings(s, {"content_type": "stream", "moment_source": "ollama", "max_clips": 8}, settings=settings)

    # 30-min range at 05:20-05:50 (19200-21000) -> ~93 min
    ranged = projects.probe(engine, URL, ranges=[[19200, 21000]], settings=settings)
    est = ranged["estimate"]
    b = est["breakdown"]
    assert set(b) == {"download_min", "encode_min", "transcribe_min", "chat_min", "translate_min", "llm_min"}
    assert est["process_minutes"] == sum(b.values())
    assert 88 <= est["process_minutes"] <= 132
    assert b["chat_min"] > 0 and b["llm_min"] > 0

    # claude_manual sets llm_min = 0, keeps chat_min
    manual = projects.probe(engine, URL, ranges=[[19200, 21000]], moment_source="claude_manual", settings=settings)
    assert manual["estimate"]["breakdown"]["llm_min"] == 0
    assert manual["estimate"]["breakdown"]["chat_min"] == b["chat_min"]
    assert manual["estimate"]["process_minutes"] == sum(manual["estimate"]["breakdown"].values())

    # variety uses live_chat; podcast does not
    variety = projects.probe(engine, URL, ranges=[[19200, 21000]], content_type="variety", settings=settings)
    podcast = projects.probe(engine, URL, ranges=[[19200, 21000]], content_type="podcast", settings=settings)
    assert variety["estimate"]["breakdown"]["chat_min"] > 0
    assert podcast["estimate"]["breakdown"]["chat_min"] == 0

    # No live chat on video -> chat_min = 0 even for stream
    no_chat_meta = meta.model_copy(update={"has_live_chat": False})
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": no_chat_meta)
    no_chat = projects.probe(engine, URL, ranges=[[19200, 21000]], content_type="stream", settings=settings)
    assert no_chat["estimate"]["breakdown"]["chat_min"] == 0


def test_create_project_normalizes_ranges(engine, settings):
    p = projects.create_project(
        engine, url=URL, ranges=[[3600, 3900], [1800, 2000], [1900, 2100]],
        transcript_source="youtube_auto", settings=settings,
    )
    assert p.ranges == [[1800.0, 2100.0], [3600.0, 3900.0]]
    assert p.transcript_source == "youtube_auto"
    with pytest.raises(AppError) as e:
        projects.create_project(engine, url=URL, transcript_source="gpt", settings=settings)
    assert e.value.code == "VALIDATION" and e.value.status == 422


def test_create_upload_with_ranges_rejected(engine, settings, tmp_path):
    src = tmp_path / "up.mp4"
    src.write_bytes(b"video")
    with pytest.raises(AppError) as e:
        projects.create_project(engine, upload_path=src, original_filename="up.mp4",
                                ranges=[[0, 60]], settings=settings)
    assert e.value.code == "VALIDATION" and e.value.status == 422
    assert src.exists()


def test_create_project_normalizes_source_language(engine, settings):
    default_proj = projects.create_project(engine, url=URL, settings=settings)
    assert default_proj.source_language is None

    ko = projects.create_project(engine, url=URL, source_language="KO", settings=settings)
    assert ko.source_language == "ko"

    region = projects.create_project(engine, url=URL, source_language="en-US", settings=settings)
    assert region.source_language == "en"

    auto = projects.create_project(engine, url=URL, source_language="auto", settings=settings)
    assert auto.source_language is None

    with pytest.raises(AppError) as e:
        projects.create_project(engine, url=URL, source_language="xx", settings=settings)
    assert e.value.code == "VALIDATION" and e.value.status == 422


def test_probe_reports_subtitle_plan_and_translation_engine(engine, settings, monkeypatch):
    from app.core.models import VideoMeta

    meta_en = VideoMeta(
        source_type="youtube",
        project_id="vid_en",
        title="English Talk",
        channel="Ch",
        duration=1800.0,
        video_id="vid_en",
        subtitles={"manual": ["en"], "auto": []},
        language="ko",
    )
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_en)

    res_claude = projects.probe(engine, URL, target_language="id", moment_source="claude_manual", settings=settings)
    assert res_claude["source_language_hint"] == "ko"
    assert res_claude["subtitle_plan"] == {"source": "youtube_manual", "language": "en", "needs_translation": True}
    assert res_claude["transcript_source_auto"] == "youtube_manual"
    assert res_claude["translation_engine"] == "claude"

    res_ollama = projects.probe(engine, URL, target_language="id", moment_source="ollama", settings=settings)
    assert res_ollama["translation_engine"] == "ollama"

    meta_id = meta_en.model_copy(update={"subtitles": {"manual": ["id", "en"], "auto": []}, "language": "id"})
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_id)
    res_id = projects.probe(engine, URL, target_language="id", settings=settings)
    assert res_id["subtitle_plan"] == {"source": "youtube_manual", "language": "id", "needs_translation": False}

    meta_multi = meta_en.model_copy(update={"subtitles": {"manual": ["ja", "th"], "auto": []}})
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_multi)
    res_th = projects.probe(engine, URL, source_language="TH", target_language="id", settings=settings)
    assert res_th["subtitle_plan"] == {"source": "youtube_manual", "language": "th", "needs_translation": True}

    with pytest.raises(AppError) as e_src:
        projects.probe(engine, URL, source_language="xx", settings=settings)
    assert e_src.value.code == "VALIDATION" and e_src.value.status == 422

    with pytest.raises(AppError) as e_tgt:
        projects.probe(engine, URL, target_language="xx", settings=settings)
    assert e_tgt.value.code == "VALIDATION" and e_tgt.value.status == 422

    with pytest.raises(AppError) as e_create_tgt:
        projects.create_project(engine, url=URL, target_language="xx", settings=settings)
    assert e_create_tgt.value.code == "VALIDATION" and e_create_tgt.value.status == 422

    for tgt in ("auto", "AUTO"):
        with pytest.raises(AppError) as e_auto_probe:
            projects.probe(engine, URL, target_language=tgt, settings=settings)
        assert e_auto_probe.value.code == "VALIDATION" and e_auto_probe.value.status == 422

        with pytest.raises(AppError) as e_auto_create:
            projects.create_project(engine, url=URL, target_language=tgt, settings=settings)
        assert e_auto_create.value.code == "VALIDATION" and e_auto_create.value.status == 422


def test_probe_warns_weak_local_translation_for_korean_with_ollama(engine, settings, monkeypatch):
    from app.core.models import VideoMeta

    meta_ko_sub = VideoMeta(
        source_type="youtube",
        project_id="vid_ko",
        title="Korean Show",
        duration=1800.0,
        video_id="vid_ko",
        subtitles={"manual": ["ko"], "auto": []},
        language="ko",
    )
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_ko_sub)
    res_sub = projects.probe(engine, URL, moment_source="ollama", settings=settings)
    assert any("Korea" in w and "Claude" in w for w in res_sub["warnings"])
    assert not any("whisper_model" in w for w in res_sub["warnings"])

    meta_ko_whisper = meta_ko_sub.model_copy(update={"subtitles": {"manual": [], "auto": ["ko"]}})
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_ko_whisper)
    res_wh = projects.probe(engine, URL, moment_source="ollama", settings=settings)
    assert any("Korea" in w and "Claude" in w for w in res_wh["warnings"])
    assert any("whisper_model" in w and "medium" in w for w in res_wh["warnings"])

    meta_fr_whisper = meta_ko_sub.model_copy(update={"subtitles": {"manual": [], "auto": []}, "language": "fr"})
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_fr_whisper)
    res_fr = projects.probe(engine, URL, source_language="fr", moment_source="ollama", settings=settings)
    assert not any("Claude" in w for w in res_fr["warnings"])
    assert any("whisper_model" in w and "medium" in w for w in res_fr["warnings"])


def test_probe_no_warning_for_claude_manual(engine, settings, monkeypatch):
    from app.core.models import VideoMeta

    meta_ko = VideoMeta(
        source_type="youtube",
        project_id="vid_ko_manual",
        title="Korean Show",
        duration=1800.0,
        video_id="vid_ko_manual",
        subtitles={"manual": ["ko"], "auto": []},
        language="ko",
    )
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_ko)
    res = projects.probe(engine, URL, moment_source="claude_manual", settings=settings)
    assert res["warnings"] == []

    meta_wh = meta_ko.model_copy(update={"subtitles": {"manual": [], "auto": []}})
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_wh)
    res_wh = projects.probe(engine, URL, source_language="ko", moment_source="claude_manual", settings=settings)
    assert not any("Claude" in w for w in res_wh["warnings"])


def test_probe_estimate_includes_translate_minutes_only_when_needed(engine, settings, monkeypatch):
    from app.core.models import VideoMeta

    assert projects.TRANSLATE_SEC_PER_MIN == pytest.approx(190.0)

    meta_en = VideoMeta(
        source_type="youtube",
        project_id="vid_tr",
        title="Talk",
        duration=3600.0,
        video_id="vid_tr",
        subtitles={"manual": ["en"], "auto": []},
    )
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_en)

    # 30 minutes (1800 s): translate_sec = (1800/60)*190 = 5700s = 95m
    ollama_en = projects.probe(engine, URL, ranges=[[0, 1800]], moment_source="ollama", settings=settings)
    assert ollama_en["estimate"]["breakdown"]["translate_min"] == 95
    assert ollama_en["estimate"]["process_minutes"] == sum(ollama_en["estimate"]["breakdown"].values())

    claude_en = projects.probe(engine, URL, ranges=[[0, 1800]], moment_source="claude_manual", settings=settings)
    assert claude_en["estimate"]["breakdown"]["translate_min"] == 0

    meta_id = meta_en.model_copy(update={"subtitles": {"manual": ["id"], "auto": []}})
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_id)
    ollama_id = projects.probe(engine, URL, ranges=[[0, 1800]], moment_source="ollama", settings=settings)
    assert ollama_id["estimate"]["breakdown"]["translate_min"] == 0

    meta_wh = meta_en.model_copy(update={"subtitles": {"manual": [], "auto": []}})
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta_wh)
    ollama_wh_ko = projects.probe(
        engine, URL, ranges=[[0, 1800]], source_language="ko", moment_source="ollama", settings=settings
    )
    assert ollama_wh_ko["estimate"]["breakdown"]["translate_min"] == 95


def test_probe_constants_calibrated(engine, settings, monkeypatch):
    from app.core.models import VideoMeta

    assert projects.WHISPER_FACTOR == pytest.approx(0.85)
    assert projects.CHAT_SEC_PER_HOUR == pytest.approx(50.0)
    assert projects.TRANSLATE_SEC_PER_MIN == pytest.approx(190.0)

    meta = VideoMeta(
        source_type="youtube",
        project_id="vid_cal",
        title="Stream",
        duration=31380.0,
        video_id="vid_cal",
        subtitles={"manual": [], "auto": ["id"]},
        has_live_chat=True,
    )
    monkeypatch.setattr(projects, "probe_youtube", lambda url, s, cookies_browser="": meta)
    res = projects.probe(
        engine, URL, ranges=[[19200, 21000]], content_type="stream", moment_source="ollama", settings=settings
    )
    b = res["estimate"]["breakdown"]
    assert b["chat_min"] == 5
    assert 25 <= b["transcribe_min"] <= 26

