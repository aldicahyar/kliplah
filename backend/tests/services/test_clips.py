import pytest
from sqlmodel import Session, select

from app.core.errors import AppError
from app.db.tables import Clip, Job, Project
from app.services import clips


def _render_jobs(engine) -> list[Job]:
    with Session(engine) as s:
        return list(s.exec(select(Job).where(Job.type == "render")).all())


def _seg(start, end, layout="face_crop", piece_index=0):
    return {"start": start, "end": end, "layout": layout, "params": {}, "piece_index": piece_index,
            "user_override": False}


def _tiles(segs: list[dict], duration: float) -> bool:
    return (segs[0]["start"] == 0.0 and segs[-1]["end"] == duration
            and all(a["end"] == b["start"] for a, b in zip(segs, segs[1:])))


def _no_overlap(subs: list[dict]) -> bool:
    return all(a["end"] <= b["start"] + 1e-6 for a, b in zip(subs, subs[1:]))


def test_list_and_get_clips(engine, ready_project, make_clip):
    project, _ = ready_project
    a = make_clip(rank=2)
    b = make_clip(rank=1)
    r = make_clip(rank=3, status="rejected")
    assert [c.id for c in clips.list_clips(engine, project.id)] == [b.id, a.id]
    assert [c.id for c in clips.list_clips(engine, project.id, include_rejected=True)] == [b.id, a.id, r.id]
    assert clips.get_clip(engine, a.id).id == a.id
    with pytest.raises(AppError) as e:
        clips.get_clip(engine, "tidak-ada")
    assert e.value.code == "NOT_FOUND" and e.value.status == 404


def test_patch_pieces_remaps_subtitles_and_fills_new_range(engine, settings, ready_project, make_clip):
    project, _ = ready_project
    clip = make_clip(12.0, 24.0)
    subs = [dict(s) for s in clip.subtitles]
    subs[0]["text"], subs[0]["edited"] = "Teks diedit", True
    clips.patch_clip(engine, clip.id, {"subtitles": subs}, settings=settings)

    out = clips.patch_clip(engine, clip.id, {"pieces": [{"project_id": project.id, "start": 6.0, "end": 30.0}]},
                           settings=settings)
    assert out.pieces == [{"project_id": project.id, "start": 6.0, "end": 30.0}]
    edited = [s for s in out.subtitles if s["edited"]]
    assert len(edited) == 1 and edited[0]["text"] == "Teks diedit"
    assert edited[0]["start"] == pytest.approx(subs[0]["start"] + 6.0)
    assert any(s["end"] <= 6.0 for s in out.subtitles)  # new head (source 6-12) filled from the transcript
    assert any(s["start"] >= 18.0 for s in out.subtitles)  # new tail (source 24-30) filled too
    assert [s["id"] for s in out.subtitles] == list(range(len(out.subtitles)))
    assert _no_overlap(out.subtitles)
    head = next(s for s in out.subtitles if s["end"] <= 6.0)
    assert head["words"] and head["words"][0]["start"] >= 0.0
    assert out.layout_segments == [{"start": 0.0, "end": 24.0, "layout": "fit_blur", "params": {},
                                    "piece_index": 0, "user_override": False}]


def test_patch_pieces_disjoint_range_falls_back_to_fit_blur(engine, settings, ready_project, make_clip):
    project, _ = ready_project
    clip = make_clip(12.0, 24.0)
    out = clips.patch_clip(engine, clip.id, {"pieces": [{"project_id": project.id, "start": 40.0, "end": 50.0}]},
                           settings=settings)
    assert out.layout_segments == [{"start": 0.0, "end": 10.0, "layout": "fit_blur", "params": {},
                                    "piece_index": 0, "user_override": False}]
    assert out.subtitles and all(0.0 <= s["start"] < s["end"] <= 10.0 for s in out.subtitles)


def test_patch_style_override_stored_as_minimal_diff(engine, make_clip):
    clip = make_clip()
    out = clips.patch_clip(engine, clip.id, {"style_override": {"subtitle": {"color": "#FFFFFF", "size": 130}}})
    assert out.style_override == {"subtitle": {"size": 130}}
    with pytest.raises(AppError) as e:
        clips.patch_clip(engine, clip.id, {"style_override": {"subtitle": {"size": "besar"}}})
    assert e.value.code == "VALIDATION" and e.value.status == 422


def test_patch_rejects_unknown_field_and_bad_shapes(engine, make_clip):
    clip = make_clip()
    cases = [{"status": "approved"}, {"subtitles": [{"id": 0, "start": 0}]},
             {"overlays": [{"id": "o", "type": "bintang", "start": 0, "end": 1, "x": 0, "y": 0}]},
             {"layout_segments": [{"start": 0, "end": 1, "layout": "miring"}]},
             {"pieces": [{"project_id": clip.project_id, "start": 5, "end": 2}]}, {"pieces": []},
             {"hook_text": 3}, {"captions": {"tiktok": {"title": 1}}}]
    for patch in cases:
        with pytest.raises(AppError) as e:
            clips.patch_clip(engine, clip.id, patch)
        assert e.value.code == "VALIDATION" and e.value.status == 422, patch
    with pytest.raises(AppError) as e:
        clips.patch_clip(engine, clip.id, {"subtitles": [{"id": 0, "start": 0}]})
    assert "subtitles" in e.value.message


def test_patch_flags_rendered_clip_only_on_real_change(engine, make_clip):
    clip = make_clip(status="rendered", hook_text="lama")
    out = clips.patch_clip(engine, clip.id, {"captions": {"tiktok": {"title": "T", "caption": "c", "hashtags": []}}})
    assert out.needs_rerender is False and out.captions["tiktok"]["title"] == "T"
    out = clips.patch_clip(engine, clip.id, {"hook_text": "lama", "subtitles": clip.subtitles})
    assert out.needs_rerender is False
    out = clips.patch_clip(engine, clip.id, {"hook_text": "baru"})
    assert out.needs_rerender is True and out.hook_text == "baru"


def test_approve_reject_restore_transitions(engine, make_clip):
    clip = make_clip()
    assert clips.approve(engine, clip.id).status == "approved"
    assert [j.clip_id for j in _render_jobs(engine)] == [clip.id]
    assert clips.reject(engine, clip.id).status == "rejected"
    with pytest.raises(AppError) as e:
        clips.approve(engine, clip.id)
    assert e.value.code == "INVALID_STATE" and e.value.status == 409
    assert clips.restore(engine, clip.id).status == "proposed"
    with pytest.raises(AppError) as e:
        clips.restore(engine, clip.id)
    assert e.value.code == "INVALID_STATE"
    rendering = make_clip(status="rendering", rank=2)
    with pytest.raises(AppError) as e:
        clips.reject(engine, rendering.id)
    assert e.value.code == "INVALID_STATE"


def test_bulk_approve_enqueues_render_per_clip(engine, make_clip):
    ids = [make_clip(rank=i).id for i in range(1, 4)]
    rejected = make_clip(rank=4, status="rejected")
    out = clips.bulk(engine, [*ids, rejected.id, "tidak-ada"], "approve")
    assert out == {"ok": ids, "errors": {rejected.id: "INVALID_STATE", "tidak-ada": "NOT_FOUND"}}
    jobs = _render_jobs(engine)
    assert sorted(j.clip_id for j in jobs) == sorted(ids)
    with Session(engine) as s:
        assert {s.get(Clip, i).status for i in ids} == {"approved"}


def test_bulk_render_skips_proposed_and_unknown_action(engine, make_clip):
    proposed = make_clip(rank=1)
    rendered = make_clip(rank=2, status="rendered")
    out = clips.bulk(engine, [proposed.id, rendered.id], "render")
    assert out == {"ok": [rendered.id], "errors": {proposed.id: "INVALID_STATE"}}
    assert [j.clip_id for j in _render_jobs(engine)] == [rendered.id]
    out = clips.bulk(engine, [proposed.id], "reject")
    assert out == {"ok": [proposed.id], "errors": {}}
    assert clips.bulk(engine, [proposed.id], "restore") == {"ok": [proposed.id], "errors": {}}
    with pytest.raises(AppError) as e:
        clips.bulk(engine, [proposed.id], "hapus")
    assert e.value.code == "VALIDATION"


def test_create_manual_clip_has_manual_origin_and_subtitles(engine, settings, ready_project, make_clip):
    project, _ = ready_project
    make_clip(rank=5)
    clip = clips.create_manual(engine, project.id, 30.0, 45.5, settings=settings)
    assert clip.origin == "manual" and clip.status == "proposed"
    assert clip.score == 0 and clip.llm_score is None and clip.rank == 6
    assert clip.pieces == [{"project_id": project.id, "start": 30.0, "end": 45.5}]
    assert clip.layout_segments == [{"start": 0.0, "end": 15.5, "layout": "fit_blur", "params": {},
                                     "piece_index": 0, "user_override": False}]
    assert clip.subtitles and all(0.0 <= s["start"] < s["end"] <= 15.5 for s in clip.subtitles)
    assert [s["id"] for s in clip.subtitles] == list(range(len(clip.subtitles)))
    assert clip.hook_text == "" and clip.hook_options == []
    assert clip.captions == {k: {"title": "", "caption": "", "hashtags": []} for k in ("tiktok", "reels", "shorts")}


def test_create_manual_requires_ready_project(engine, settings, ready_project):
    project, _ = ready_project
    with Session(engine) as s:
        row = s.get(Project, project.id)
        row.status = "processing"
        s.add(row)
        s.commit()
    with pytest.raises(AppError) as e:
        clips.create_manual(engine, project.id, 0.0, 10.0, settings=settings)
    assert e.value.code == "INVALID_STATE" and e.value.status == 409
    with pytest.raises(AppError) as e:
        clips.create_manual(engine, "tidak-ada", 0.0, 10.0, settings=settings)
    assert e.value.code == "NOT_FOUND"


def test_patch_pieces_mid_segment_cut_keeps_layout_coverage(engine, settings, ready_project, make_clip):
    project, _ = ready_project
    clip = make_clip(12.0, 24.0, layout_segments=[_seg(0.0, 4.0), _seg(4.0, 12.0, "fit_blur")])
    pid = project.id
    # source 12-16 (face_crop) is cut in two around a fit_blur range: its longer part survives, a gap remains
    new_pieces = [{"project_id": pid, "start": 12.0, "end": 13.0}, {"project_id": pid, "start": 20.0, "end": 24.0},
                  {"project_id": pid, "start": 14.0, "end": 16.0}]
    out = clips.patch_clip(engine, clip.id, {"pieces": new_pieces}, settings=settings)
    assert out.layout_segments == [_seg(0.0, 5.0, "fit_blur", 0), _seg(5.0, 7.0, "face_crop", 2)]
    assert _tiles(out.layout_segments, 7.0)


def test_patch_client_layout_segments_are_normalized(engine, make_clip):
    clip = make_clip(12.0, 24.0)
    out = clips.patch_clip(engine, clip.id, {"layout_segments": [_seg(1.0, 3.0), _seg(5.0, 40.0, "fit_blur")]})
    assert out.layout_segments == [_seg(0.0, 5.0), _seg(5.0, 12.0, "fit_blur")]


def test_patch_client_items_rounded_and_clipped_to_output(engine, make_clip):
    clip = make_clip(12.0, 24.0)
    subs = [{"id": 0, "start": 1.00049, "end": 2.0, "text": "a", "words": [{"w": "a", "start": 1.0, "end": 2.0}]},
            {"id": 1, "start": 11.0, "end": 13.0, "text": "b", "words": [{"w": "b", "start": 11.5, "end": 12.5}]},
            {"id": 2, "start": 12.5, "end": 14.0, "text": "c"}]
    overlays = [{"id": "o1", "type": "label", "start": 10.0, "end": 30.0, "x": 0.5, "y": 0.5, "text": "hai"},
                {"id": "o2", "type": "emoji", "start": 15.0, "end": 16.0, "x": 0.5, "y": 0.5, "emoji": "x"}]
    out = clips.patch_clip(engine, clip.id, {"subtitles": subs, "overlays": overlays})
    assert [(s["start"], s["end"]) for s in out.subtitles] == [(1.0, 2.0), (11.0, 12.0)]
    assert out.subtitles[1]["words"][0]["end"] == 12.0
    assert [(o["id"], o["start"], o["end"]) for o in out.overlays] == [("o1", 10.0, 12.0)]


def test_patch_pieces_on_rendered_clip_sets_needs_rerender(engine, settings, ready_project, make_clip):
    project, _ = ready_project
    clip = make_clip(12.0, 24.0, status="rendered")
    out = clips.patch_clip(engine, clip.id, {"pieces": [{"project_id": project.id, "start": 12.0, "end": 20.0}]},
                           settings=settings)
    assert out.needs_rerender is True and out.status == "rendered"


def test_reject_cancels_queued_render_job(engine, make_clip):
    clip = make_clip()
    clips.approve(engine, clip.id)
    clips.reject(engine, clip.id)
    assert [j.status for j in _render_jobs(engine)] == ["cancelled"]


def test_patch_piece_outside_ranges_rejected(engine, settings, ready_project, make_clip):
    from app.core.models import VideoMeta

    project, _ = ready_project
    with Session(engine) as s:
        row = s.get(Project, project.id)
        row.duration_sec = 300.0
        row.ranges = [[0.0, 30.0], [100.0, 150.0]]
        s.add(row)
        s.commit()
    pdir = settings.data_dir / "projects" / project.id
    meta = VideoMeta(
        source_type="youtube", project_id=project.id, title="T", duration=300.0,
        ranges=[{"index": 0, "start": 0.0, "end": 30.0, "file": "source_r0.mp4"},
                {"index": 1, "start": 100.0, "end": 150.0, "file": "source_r1.mp4"}],
    )
    (pdir / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")

    clip = make_clip(5.0, 15.0)
    for bad in ((40.0, 50.0), (20.0, 110.0)):
        with pytest.raises(AppError) as e:
            clips.patch_clip(engine, clip.id, {"pieces": [{"project_id": project.id, "start": bad[0], "end": bad[1]}]},
                             settings=settings)
        assert e.value.code == "VALIDATION" and e.value.status == 422
        assert e.value.message == "Bagian video ini belum diunduh (di luar rentang project)."

        with pytest.raises(AppError) as e2:
            clips.create_manual(engine, project.id, bad[0], bad[1], settings=settings)
        assert e2.value.code == "VALIDATION" and e2.value.status == 422
        assert e2.value.message == "Bagian video ini belum diunduh (di luar rentang project)."

    ok = clips.create_manual(engine, project.id, 105.0, 120.0, settings=settings)
    assert ok.pieces == [{"project_id": project.id, "start": 105.0, "end": 120.0}]
