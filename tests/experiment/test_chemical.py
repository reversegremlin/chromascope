"""Tests for ChemicalRenderer and ChemicalConfig."""

import numpy as np
import pytest

from chromascope.experiment.chemical import ChemicalConfig, ChemicalRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(
    energy: float = 0.5,
    percussive: float = 0.3,
    low: float = 0.4,
    high: float = 0.2,
    sub_bass: float = 0.2,
    harmonic: float = 0.3,
    brilliance: float = 0.1,
    flux: float = 0.2,
    flatness: float = 0.4,
    centroid: float = 0.5,
    is_beat: bool = False,
) -> dict:
    return {
        "global_energy": energy,
        "percussive_impact": percussive,
        "low_energy": low,
        "high_energy": high,
        "sub_bass": sub_bass,
        "harmonic_energy": harmonic,
        "brilliance": brilliance,
        "spectral_flux": flux,
        "spectral_flatness": flatness,
        "spectral_centroid": centroid,
        "is_beat": is_beat,
        "pitch_hue": 0.0,
        "sharpness": 0.1,
    }


def _renderer(w=64, h=64, seed=42, **kwargs) -> ChemicalRenderer:
    cfg = ChemicalConfig(width=w, height=h, fps=30, **kwargs)
    return ChemicalRenderer(cfg, seed=seed)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestChemicalConfigDefaults:
    def test_default_style(self):
        cfg = ChemicalConfig()
        assert cfg.style == "neon_lab"

    def test_default_palette(self):
        cfg = ChemicalConfig()
        assert cfg.chem_palette == "mixed"

    def test_inherits_base(self):
        cfg = ChemicalConfig(width=1280, height=720)
        assert cfg.width == 1280
        assert cfg.fps == 60


class TestChemicalRendererInit:
    def test_sim_grid_allocated(self):
        r = _renderer(w=128, h=64)
        assert r._field_a.shape == (r._sim_h, r._sim_w)
        assert r._field_b.shape == (r._sim_h, r._sim_w)
        assert r._field_heat.shape == (r._sim_h, r._sim_w)
        assert r._field_crystal.shape == (r._sim_h, r._sim_w)

    def test_sim_grid_quarter_resolution(self):
        r = _renderer(w=128, h=128)
        assert r._sim_w == 32
        assert r._sim_h == 32

    def test_gs_seed_creates_b_patches(self):
        r = _renderer(w=64, h=64, seed=7)  # noqa: seed goes to renderer, not config
        # After init_gs_seed, field_b should have some non-zero values
        assert np.any(r._field_b > 0)

    def test_field_a_starts_near_one(self):
        r = _renderer()
        # Most of field_a should be near 1 (dilute solution)
        assert r._field_a.mean() > 0.5


# ---------------------------------------------------------------------------
# Single frame rendering
# ---------------------------------------------------------------------------

class TestRenderFrame:
    def test_output_shape(self):
        r = _renderer(w=64, h=64)
        frame = r.render_frame(_frame(), 0)
        assert frame.shape == (64, 64, 3)

    def test_output_dtype(self):
        r = _renderer(w=64, h=64)
        frame = r.render_frame(_frame(), 0)
        assert frame.dtype == np.uint8

    def test_not_all_black(self):
        # After a beat frame, some pixels should be non-zero
        r = _renderer(w=64, h=64)
        for i in range(5):
            frame = r.render_frame(
                _frame(energy=0.9, percussive=0.9, sub_bass=0.9, is_beat=True), i
            )
        assert np.sum(frame) > 0

    def test_output_values_in_range(self):
        r = _renderer(w=64, h=64)
        frame = r.render_frame(_frame(), 0)
        assert frame.min() >= 0
        assert frame.max() <= 255


# ---------------------------------------------------------------------------
# Manifest rendering
# ---------------------------------------------------------------------------

class TestRenderManifest:
    def test_frame_count(self):
        r = _renderer(w=32, h=32)
        manifest = {"frames": [_frame() for _ in range(4)]}
        frames = list(r.render_manifest(manifest))
        assert len(frames) == 4

    def test_frame_shapes_consistent(self):
        r = _renderer(w=32, h=32)
        manifest = {"frames": [_frame(is_beat=(i % 2 == 0)) for i in range(3)]}
        for frame in r.render_manifest(manifest):
            assert frame.shape == (32, 32, 3)

    def test_progress_callback(self):
        r = _renderer(w=16, h=16)
        manifest = {"frames": [_frame() for _ in range(3)]}
        log = []
        list(r.render_manifest(manifest, progress_callback=lambda c, t: log.append((c, t))))
        assert log == [(1, 3), (2, 3), (3, 3)]


# ---------------------------------------------------------------------------
# Simulation behaviour
# ---------------------------------------------------------------------------

class TestSimulationBehaviour:
    def test_crystal_grows_on_beat(self):
        r = _renderer(w=64, h=64, nucleation_threshold=0.1)
        initial_crystal = r._field_crystal.sum()
        for i in range(10):
            r.update(_frame(percussive=0.95, sub_bass=0.9, energy=0.8, is_beat=True))
        assert r._field_crystal.sum() > initial_crystal

    def test_heat_field_activated_by_reaction(self):
        r = _renderer(w=64, h=64)
        for i in range(5):
            r.update(_frame(energy=0.7, sub_bass=0.8, is_beat=True))
        assert r._field_heat.max() > 0.0

    def test_edge_field_follows_crystal(self):
        r = _renderer(w=64, h=64, nucleation_threshold=0.1)
        for i in range(15):
            r.update(_frame(percussive=0.9, sub_bass=0.9, energy=0.9, is_beat=True))
        # If crystal has grown, edge field should be non-zero
        if r._field_crystal.max() > 0.01:
            assert r._field_edge.max() > 0.0

    def test_crystal_dissolves_on_quiet(self):
        r = _renderer(w=64, h=64, nucleation_threshold=0.1)
        # First grow crystals with sustained high energy
        for _ in range(20):
            r.update(_frame(percussive=0.9, energy=0.9, sub_bass=0.9, is_beat=True))
        grown = r._field_crystal.sum()
        # Then sustained silence — heat dies, dissolution takes over
        for _ in range(60):
            r.update(_frame(energy=0.0, percussive=0.0, sub_bass=0.0))
        dissolved = r._field_crystal.sum()
        assert dissolved < grown

    def test_gs_fields_stay_bounded(self):
        r = _renderer(w=64, h=64)
        for i in range(20):
            r.update(_frame(energy=1.0, sub_bass=1.0, is_beat=True))
        assert r._field_a.min() >= 0.0
        assert r._field_a.max() <= 1.0
        assert r._field_b.min() >= 0.0
        assert r._field_b.max() <= 1.0


# ---------------------------------------------------------------------------
# Palette and style
# ---------------------------------------------------------------------------

class TestPaletteStyles:
    @pytest.mark.parametrize("palette", ["iron", "copper", "sodium", "potassium", "mixed"])
    def test_named_palette_renders(self, palette):
        r = ChemicalRenderer(ChemicalConfig(width=32, height=32, chem_palette=palette), seed=0)
        for i in range(3):
            frame = r.render_frame(_frame(energy=0.7, is_beat=True), i)
        assert frame.shape == (32, 32, 3)

    @pytest.mark.parametrize("style", ["neon_lab", "plasma_beaker", "midnight_fluor", "synth_chem"])
    def test_named_style_renders(self, style):
        r = ChemicalRenderer(ChemicalConfig(width=32, height=32, style=style), seed=0)
        for i in range(3):
            frame = r.render_frame(_frame(energy=0.7, is_beat=True), i)
        assert frame.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_fixed_seed_same_output(self):
        frames_a = []
        frames_b = []
        frame_inputs = [_frame(energy=0.5 + i * 0.05, is_beat=(i % 3 == 0)) for i in range(5)]

        for seed_frames, store in [(frames_a, True), (frames_b, True)]:
            r = ChemicalRenderer(ChemicalConfig(width=32, height=32, fps=30), seed=99)
            for i, fd in enumerate(frame_inputs):
                f = r.render_frame(fd, i)
                if store:
                    seed_frames.append(f.copy())

        for fa, fb in zip(frames_a, frames_b):
            np.testing.assert_array_equal(fa, fb)


# ---------------------------------------------------------------------------
# get_raw_field
# ---------------------------------------------------------------------------

class TestGetRawField:
    def test_shape_is_sim_grid(self):
        r = _renderer(w=64, h=64)
        r.update(_frame())
        field = r.get_raw_field()
        assert field.shape == (r._sim_h, r._sim_w)

    def test_dtype_float32(self):
        r = _renderer(w=64, h=64)
        r.update(_frame())
        field = r.get_raw_field()
        assert field.dtype == np.float32

    def test_values_in_01(self):
        r = _renderer(w=64, h=64)
        for i in range(5):
            r.update(_frame(energy=0.8, is_beat=True))
        field = r.get_raw_field()
        assert field.min() >= 0.0
        assert field.max() <= 1.0
