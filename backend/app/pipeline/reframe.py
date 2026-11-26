"""Reframe 16:9 to 9:16 with one fixed crop per scene, aimed at whoever is talking.

Approach (one crop per scene, mouth movement picks the speaker, 1.25x switch threshold) adapted
from fralapo/clippyme (https://github.com/fralapo/clippyme) and ClipsAI
(https://github.com/ClipsAI/clipsai), both MIT-licensed.
"""
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.core.models import ReframeSegment

TINY_FACE = 0.10        # ignore faces narrower than 10% of the largest in the frame
TRACK_JOIN = 0.08       # faces within this cx distance across frames are the same person
CENTER_SNAP = 0.10
MOVING = 0.12           # cx std above this: subject moves too much for a fixed crop
SWITCH = 1.25           # a new speaker must be this much more active to take the frame
PREV_NEAR = 0.10
MERGE_DX = 0.04
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


@dataclass
class Face:
    cx: float          # 0..1 across the frame
    width: float       # 0..1 of frame width
    mouth_open: float  # jawOpen score, 0 when unknown


def _activity(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _snap(x: float) -> float:
    return 0.5 if abs(x - 0.5) < CENTER_SNAP else x


def collapse_scene(start: float, end: float, samples: list[list[Face]], prev_x: float | None) -> ReframeSegment:
    frames = []
    for faces in samples:
        if faces:
            biggest = max(f.width for f in faces)
            frames.append([f for f in faces if f.width >= TINY_FACE * biggest])
        else:
            frames.append([])
    with_faces = sum(1 for f in frames if f)
    if not samples or with_faces < 0.5 * len(samples):
        return ReframeSegment(start=start, end=end, mode="general", x=0.5)

    tracks: list[list[Face]] = []
    for faces in frames:
        for f in faces:
            home = min(tracks, key=lambda t: abs(t[-1].cx - f.cx), default=None)
            if home is not None and abs(home[-1].cx - f.cx) <= TRACK_JOIN:
                home.append(f)
            else:
                tracks.append([f])
    tracks = [t for t in tracks if len(t) >= max(1, 0.3 * with_faces)]
    if not tracks:  # faces jump around too much to follow: show the whole frame
        return ReframeSegment(start=start, end=end, mode="wide", x=0.5)

    def info(t):
        return statistics.median(f.cx for f in t), _activity([f.mouth_open for f in t]), t

    ranked = sorted((info(t) for t in tracks), key=lambda r: r[1], reverse=True)
    chosen = None
    if len(ranked) == 1:
        chosen = ranked[0]
    else:
        best, second = ranked[0], ranked[1]
        if prev_x is not None:
            keep = [r for r in ranked if abs(r[0] - prev_x) <= PREV_NEAR and r[1] >= best[1] / SWITCH]
            if keep:
                chosen = keep[0]
        if chosen is None and best[1] > 0 and best[1] >= SWITCH * second[1]:
            chosen = best
    if chosen is None:
        return ReframeSegment(start=start, end=end, mode="wide", x=0.5)
    x, _, track = chosen
    if _activity([f.cx for f in track]) > MOVING:
        return ReframeSegment(start=start, end=end, mode="wide", x=0.5)
    return ReframeSegment(start=start, end=end, mode="track", x=round(_snap(x), 4))


def merge_segments(segs: list[ReframeSegment]) -> list[ReframeSegment]:
    out: list[ReframeSegment] = []
    for s in segs:
        if out and out[-1].mode == s.mode and abs(out[-1].x - s.x) < MERGE_DX:
            out[-1] = out[-1].model_copy(update={"end": s.end})
        else:
            out.append(s)
    return out


def crop_x_px(x: float, src_w: int, src_h: int) -> int:
    cw = int(round(src_h * 9 / 16 / 2) * 2)
    return int(min(max(round(x * src_w - cw / 2), 0), src_w - cw))


def ensure_face_model(models_dir: Path) -> Path:
    path = models_dir / "face_landmarker.task"
    if not path.exists():
        import httpx

        models_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with httpx.stream("GET", FACE_MODEL_URL, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        tmp.replace(path)
    return path


def detect_scenes(video: Path, start: float, end: float) -> list[tuple[float, float]]:
    from scenedetect import AdaptiveDetector, detect

    scenes = detect(str(video), AdaptiveDetector(), start_time=start, end_time=end)
    out = [(max(0.0, a.get_seconds() - start), min(end, b.get_seconds()) - start) for a, b in scenes]
    out = [(a, b) for a, b in out if b - a > 0.05]
    return out or [(0.0, end - start)]


class _FaceSampler:
    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
                                            running_mode=vision.RunningMode.IMAGE, num_faces=4,
                                            output_face_blendshapes=True)
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def faces(self, frame_bgr) -> list[Face]:
        import cv2

        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (640, round(640 * h / w)))
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        res = self._lm.detect(img)
        out = []
        for i, pts in enumerate(res.face_landmarks):
            xs = [p.x for p in pts]
            jaw = 0.0
            if res.face_blendshapes and i < len(res.face_blendshapes):
                jaw = next((c.score for c in res.face_blendshapes[i] if c.category_name == "jawOpen"), 0.0)
            out.append(Face(cx=(min(xs) + max(xs)) / 2, width=max(xs) - min(xs), mouth_open=jaw))
        return out

    def close(self):
        self._lm.close()


def sample_faces(video: Path, abs_start: float, abs_end: float, sampler: _FaceSampler, n: int = 9) -> list[list[Face]]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        out = []
        for i in range(n):
            t = abs_start + (i + 0.5) * (abs_end - abs_start) / n
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            out.append(sampler.faces(frame) if ok else [])
        return out
    finally:
        cap.release()


def video_size(video: Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def reframe_clip(video: Path, clip_start: float, clip_end: float,
                 models_dir: Path | None = None) -> tuple[list[ReframeSegment], Literal["face_crop", "fit_blur"]]:
    if models_dir is None:
        from app.core.config import get_settings

        models_dir = Path(get_settings().data_dir) / "models"
    try:
        sampler = _FaceSampler(ensure_face_model(models_dir))
    except Exception as e:  # offline first run, corrupt model: still produce a usable clip
        import logging

        logging.getLogger(__name__).warning("face model unavailable, using fit+blur: %s", e)
        return [ReframeSegment(start=0.0, end=round(clip_end - clip_start, 3), mode="general", x=0.5)], "fit_blur"
    try:
        segs, prev = [], None
        for a, b in detect_scenes(video, clip_start, clip_end):
            seg = collapse_scene(a, b, sample_faces(video, clip_start + a, clip_start + b, sampler), prev)
            if seg.mode == "track":
                prev = seg.x
            segs.append(seg)
    finally:
        sampler.close()
    segs = merge_segments(segs)
    total = clip_end - clip_start
    not_tracked = sum(s.end - s.start for s in segs if s.mode != "track")
    return segs, ("fit_blur" if not_tracked > 0.5 * total else "face_crop")
