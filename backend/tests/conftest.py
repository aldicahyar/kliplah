import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """20 s 1280x720 test pattern with a 440 Hz tone, silent between 8 s and 10 s."""
    out = tmp_path_factory.mktemp("media") / "synthetic.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=20",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=20",
        "-af", "volume=enable='between(t,8,10)':volume=0",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
    ], check=True)
    return out
