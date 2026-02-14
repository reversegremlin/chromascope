"""Tests for the frame orchestrator."""

import numpy as np
import pytest

from chromascope.experiment.renderer import FractalKaleidoscopeRenderer, RenderConfig


def _mock_manifest(n_frames: int = 10, fps: int = 30) -> dict:
    """Build a minimal manifest for testing."""
    frames = []
    for i in range(n_frames):
        frames.append({
            "frame_index": i,
            "time": i / fps,
            "is_beat": i % 8 == 0,
            "is_onset": i % 4 == 0,
            "percussive_impact": 0.3 + 0.4 * (i % 8 == 0),
            "harmonic_energy": 0.5,
            "global_energy": 0.4,
            "low_energy": 0.3,
            "mid_energy": 0.4,
            "high_energy": 0.5,
            "spectral_brightness": 0.5,
            "dominant_chroma": "G",
            "pitch_hue": 0.583,
            "texture": 0.5,
        })
    return {
        "metadata": {"bpm": 120, "duration": n_frames / fps, "fps": fps},
        "frames": frames,
    }


class TestFractalKaleidoscopeRenderer:
    def test_single_frame(self):
        config = RenderConfig(width=160, height=120, fps=30, glow_enabled=False, aberration_enabled=False)
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=1)
        frame = renderer.render_frame(manifest["frames"][0], 0)
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8

    def test_multiple_frames_different(self):
        config = RenderConfig(width=80, height=60, fps=30, glow_enabled=False, aberration_enabled=False)
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=3)

        frames = list(renderer.render_manifest(manifest))
        assert len(frames) == 3
        # Consecutive frames should differ (animation is progressing)
        assert not np.array_equal(frames[0], frames[1])

    def test_generator_yields(self):
        config = RenderConfig(width=80, height=60, fps=30, glow_enabled=False, aberration_enabled=False)
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=5)

        count = 0
        for frame in renderer.render_manifest(manifest):
            assert frame.shape == (60, 80, 3)
            count += 1
        assert count == 5

    def test_julia_mode(self):
        config = RenderConfig(
            width=80, height=60, fps=30,
            fractal_mode="julia",
            glow_enabled=False, aberration_enabled=False,
        )
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=2)
        frames = list(renderer.render_manifest(manifest))
        assert len(frames) == 2

    def test_mandelbrot_mode(self):
        config = RenderConfig(
            width=80, height=60, fps=30,
            fractal_mode="mandelbrot",
            glow_enabled=False, aberration_enabled=False,
        )
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=2)
        frames = list(renderer.render_manifest(manifest))
        assert len(frames) == 2

    def test_progress_callback(self):
        config = RenderConfig(width=80, height=60, fps=30, glow_enabled=False, aberration_enabled=False)
        renderer = FractalKaleidoscopeRenderer(config)
        manifest = _mock_manifest(n_frames=3)

        progress = []
        for _ in renderer.render_manifest(manifest, progress_callback=lambda c, t: progress.append((c, t))):
            pass
        assert len(progress) == 3
        assert progress[-1] == (3, 3)
