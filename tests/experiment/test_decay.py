import numpy as np
import pytest
from chromascope.experiment.decay import DecayRenderer, DecayConfig

def test_decay_renderer_init():
    config = DecayConfig(width=640, height=480, fps=30)
    renderer = DecayRenderer(config)
    assert renderer.cfg.width == 640
    assert renderer.trail_buffer.shape == (480, 640)
    assert len(renderer.particles) == 0

def test_decay_renderer_spawn():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    renderer.spawn_particle("alpha")
    assert len(renderer.particles) == 1
    assert renderer.particles[0].type == "alpha"
    
    renderer.spawn_particle("beta")
    assert len(renderer.particles) == 2
    assert renderer.particles[1].type == "beta"

def test_render_frame():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    frame_data = {
        "global_energy": 0.8,
        "percussive_impact": 0.9,
        "low_energy": 0.6,
        "high_energy": 0.4,
        "is_beat": True,
        "spectral_flux": 1.0,
        "sub_bass": 1.0, # High bass for fission
        "brilliance": 0.8,
        "spectral_centroid": 0.7
    }
    renderer._smooth_sub_bass = 1.0
    frame = renderer.render_frame(frame_data, 0)
    assert frame.shape == (100, 100, 3)
    assert frame.dtype == np.uint8
    # Particles should have been spawned
    assert len(renderer.particles) > 0
    # Zoom should have reacted
    assert renderer.view_zoom > 1.0
    # Fission cores should have been created on high-bass beat
    assert len(renderer.fission_cores) > 0

def test_render_manifest():
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    manifest = {
        "frames": [
            {"global_energy": 0.1, "is_beat": False, "sub_bass": 0.1},
            {"global_energy": 0.9, "is_beat": True, "sub_bass": 0.9}
        ]
    }
    frames = list(renderer.render_manifest(manifest))
    assert len(frames) == 2
    assert frames[0].shape == (100, 100, 3)

def test_styles():
    config = DecayConfig(width=10, height=10, style="uranium")
    renderer = DecayRenderer(config)
    buffer = np.ones((10, 10), dtype=np.float32)
    cherenkov = np.zeros((10, 10), dtype=np.float32)
    rgb = renderer._apply_styles(buffer, cherenkov)
    assert rgb.shape == (10, 10, 3)
    # Uranium should be greenish: G should be strong
    assert rgb[0, 0, 1] > 200

    renderer.cfg.style = "lab"
    rgb_lab = renderer._apply_styles(buffer, cherenkov)
    assert np.all(rgb_lab[0, 0, 0] == rgb_lab[0, 0, 1] == rgb_lab[0, 0, 2])

def test_weak_force_mutation_probabilistic():
    """Run multiple times to catch the probabilistic mutation."""
    renderer = DecayRenderer(DecayConfig(width=100, height=100))
    
    mutated = False
    for _ in range(100):
        # Manually add an Alpha particle with low life to trigger mutation
        renderer.spawn_particle("alpha", x=50, y=50, vx=1, vy=1)
        renderer.particles[-1].life = 0.4
        
        # Mock high flux to increase probability
        renderer._smooth_flux = 1.0
        
        renderer.update_particles(1.0/60.0)
        if any(p.type == "beta" for p in renderer.particles):
            mutated = True
            break
        renderer.particles = [] # Reset for next attempt
        
    assert mutated, "Mutation didn't trigger in 100 attempts at 5% prob"
