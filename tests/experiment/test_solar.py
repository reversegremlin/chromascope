"""
Tests for the solar renderer.
"""
import numpy as np

from chromascope.experiment.solar import SolarRenderer
from chromascope.experiment.solar_cli import RenderConfig

def test_solar_renderer_runs():
    """
    Test that the solar renderer runs for a few frames without errors.
    """
    config = RenderConfig(width=100, height=100, fps=60)
    renderer = SolarRenderer(config)
    
    manifest = {
        "frames": [
            {"global_energy": 0.5, "low_energy": 0.5},
            {"global_energy": 0.6, "low_energy": 0.4},
            {"global_energy": 0.7, "low_energy": 0.3},
        ]
    }
    
    frames = list(renderer.render_manifest(manifest))
    
    assert len(frames) == 3
    for frame in frames:
        assert frame.shape == (100, 100, 3)
        # Check that the frame is not all black
        assert np.sum(frame) > 0
