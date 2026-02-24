# CLAUDE.md — chromascope

## What This Repo Is

`chromascope` is an audio analysis engine + web-based visualizer. The Python
backend extracts musical features from audio files and writes a time-aligned
JSON/NumPy manifest. The JavaScript frontend (`frontend/app.js`) reads that
manifest and drives a Canvas 2D kaleidoscope renderer with 11 visual styles.

The experiment renderers (attractor, bloom, chemical, decay, fractal) live in the
sibling private repo `~/chromascope-experiment/`.

---

## Environment

```bash
# Virtual environment
/home/nadiabellamorris/chromascope/.venv/

# Install (required before running tests)
.venv/bin/python -m pip install -e .

# Run tests
.venv/bin/python -m pytest tests/ -v

# 63 core tests pass. SKIP test_kaleidoscope.py in headless env —
# pygame.init() hangs without a display.
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_kaleidoscope.py
```

**Current schema version:** 2.0. `ANALYSIS_VERSION = "2.0"` in `analyzer.py`.
Bumping this invalidates all cached manifests.

---

## Key Files

```
src/chromascope/
├── core/
│   ├── decomposer.py    HPSS: harmonic + percussive separation
│   ├── analyzer.py      Feature extraction (RMS, chroma, MFCC, pitch, structure, harmony)
│   ├── polisher.py      Attack/release envelopes, normalization → [0, 1]
│   └── stream.py        RealtimeAnalyzer stub (Phase 3)
├── io/
│   └── exporter.py      Writes per-frame JSON manifest + numpy arrays
├── pipeline.py          AudioPipeline: orchestrates all four stages
frontend/
├── app.js               KaleidoscopeStudio class (~6500 lines), all 11 styles
├── server.py            Flask API: /api/analyze, /api/render
docs/
├── project/
│   ├── AUDIO_INTELLIGENCE_PROPOSAL.md   Full Phase 1/2/3 roadmap (read this)
│   ├── roadmap.md                        High-level roadmap
│   └── AGENTS.md                         Cognitive role reference (human docs)
├── MANIFEST.md          Manifest field reference
└── SCHEMA_CHANGELOG.md  v1.1 → v2.0 diff
```

---

## Pipeline Architecture

```
Audio file
  → decomposer.py   HPSS → DecomposedAudio (harmonic, percussive, original)
  → analyzer.py     ExtractedFeatures (per-frame arrays: RMS, chroma, MFCC,
                    pitch, structure, key, beats, onsets, spectral)
  → polisher.py     PolishedFeatures (normalized, enveloped → [0, 1] scalars)
  → exporter.py     Manifest JSON + .npy cache
  → frontend/app.js KaleidoscopeStudio reads manifest, drives Canvas 2D render
```

Each stage is independently testable. All manifest changes are **additive**
(new fields only — old consumers keep working).

---

## Manifest Schema (v2.0 per-frame fields)

```
Energy:    percussive_impact, harmonic_energy, global_energy, spectral_flux
Frequency: sub_bass, bass, low_mid, mid, high_mid, presence, brilliance
           sub_bass_cqt, bass_cqt  [opt-in: use_cqt=True]
Texture:   spectral_brightness, spectral_flatness, spectral_rolloff,
           zero_crossing_rate, spectral_bandwidth, spectral_contrast (7 bins)
Tonality:  chroma_values (12), dominant_chroma
           mfcc (13), mfcc_delta (13), mfcc_delta2 (13), timbre_velocity
Pitch:     f0_hz, f0_voiced, f0_probs, pitch_register
Timing:    is_beat, is_onset, is_downbeat, beat_position, bar_index, bar_progress
           onset_type ("transient"/"percussive"/"harmonic"), onset_sharpness
Structure: section_index, section_novelty, section_progress, section_change
Harmony:   key_stability
Primitives: impact, fluidity, brightness, pitch_hue, texture, sharpness
```

Metadata block includes `structure` (section boundaries) and `key` (root,
mode, confidence).

---

## Frontend: 11 Visual Styles

All styles live in `frontend/app.js` as methods on `KaleidoscopeStudio`.
The Python backend only renders Geometric; all other styles are JS-only.

Each style follows this pattern:
- `render*Style(ctx, centerX, centerY, radius, numSides, hue, thickness)`
- `render*Background(ctx, centerX, centerY, radius, hue)` — 3-layer parallax
- `renderStyledBackground()` dispatches to the per-style background method

Key shared state:
- `smoothedValues` — `{percussiveImpact, harmonicEnergy, spectralBrightness}`
- `accumulatedRotation` — time-based rotation accumulator
- `_bgFractalRotation` — background parallax rotation
- `seededRandom(seed)` — deterministic variation within styles

---

## Workflow

1. **Read before editing.** Never propose changes to code you haven't read.
2. **Plan before coding** on anything non-trivial — use `EnterPlanMode` and
   get approval before writing code. The Maestro role: scope → read → plan →
   approval → implement.
3. **Match scope to request.** A bug fix doesn't need surrounding refactors.
   Don't add error handling, helpers, or abstractions beyond what the task
   requires. Three similar lines is better than a premature abstraction.
4. **Debug systematically.** Trace specific values (one frame, one field).
   Disable optional features (e.g., `use_cqt=False`) to isolate variables.
   The first hypothesis is often wrong — verify before fixing.
5. **Parity first, speed second.** Add a correctness test before an
   optimization.
6. **Update memory.** If architecture decisions, key file locations, or
   non-obvious patterns change, update
   `~/.claude/projects/-home-nadiabellamorris-chromascope/memory/MEMORY.md`.

---

## Audio Intelligence Phases

See `docs/project/AUDIO_INTELLIGENCE_PROPOSAL.md` for full detail.

| Phase | Status | What it adds |
|---|---|---|
| Phase 1 (Crawl) | **Complete** | MFCC export, structure, pitch, key, downbeats, CQT, spectral bandwidth/contrast — schema 2.0 |
| Phase 2 (Walk) | W4 done, W1–W3 stubs | Demucs stems (W1), madmom beats (W2), chord detection (W3) — schema 3.0 |
| Phase 3 (Run) | All stubs | CREPE pitch (R1), emotion embeddings (R2), real-time streaming (R3), Ableton (R4) — schema 4.0 |

**Recommended next step:** W2 (madmom neural beats) — 12–16h, lowest risk,
immediately improves every renderer. See proposal §W2.

---

## Test Notes

- `test_kaleidoscope.py` — **skip in headless env** (pygame.init() hangs)
- `tests/test_phase1_audio_intelligence.py` — 85 tests for Phase 1 features
- OOM fix: `librosa.pyin` runs at `hop_length * 4`; output is upsampled back
  via nearest-neighbour. `chroma_stft` (not `chroma_cqt`) in structure
  extraction. Do not revert either of these.
- K-S key detection on pure C-major chord (C, E, G) is ambiguous with
  A-minor — tests use diatonic set membership, not exact key label.
