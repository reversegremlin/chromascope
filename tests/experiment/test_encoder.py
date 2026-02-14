"""Tests for the FFmpeg video encoder."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chromascope.experiment.encoder import encode_video


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="chromascope_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def test_audio():
    """Path to a short test audio file."""
    p = Path(__file__).parent.parent / "assets" / "madonna.mp3"
    if not p.exists():
        pytest.skip("Test audio not found")
    return p


def _solid_frames(n: int, width: int, height: int, color=(128, 64, 200)):
    """Generate N solid-color frames."""
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    for _ in range(n):
        yield frame.copy()


class TestEncoder:
    def test_produces_mp4(self, tmp_dir, test_audio):
        output = tmp_dir / "test_output.mp4"
        frames = _solid_frames(30, 320, 240)

        result = encode_video(
            frame_iterator=frames,
            audio_path=test_audio,
            output_path=output,
            width=320,
            height=240,
            fps=30,
            quality="fast",
            duration=1.0,
        )

        assert result.exists()
        assert result.stat().st_size > 0

    def test_progress_callback(self, tmp_dir, test_audio):
        output = tmp_dir / "test_progress.mp4"
        frames = _solid_frames(15, 160, 120)

        progress = []
        encode_video(
            frame_iterator=frames,
            audio_path=test_audio,
            output_path=output,
            width=160,
            height=120,
            fps=30,
            quality="fast",
            duration=0.5,
            total_frames=15,
            progress_callback=lambda c, t: progress.append((c, t)),
        )

        assert len(progress) == 15
        assert output.exists()
