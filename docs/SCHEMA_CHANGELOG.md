# Manifest Schema Changelog

The manifest schema version is stored in `manifest.metadata.schema_version`.
The engine version that produced it is in `manifest.metadata.version` (same value).
Both are set by `ANALYSIS_VERSION` in `pipeline.py`.

All schema changes are **additive** — new fields only, never removed or renamed.
Old consumers work against new manifests; new consumers gracefully degrade on old ones.

---

## Schema 1.1 (baseline)

The original production schema. Contained everything needed for a beat-reactive
visualizer driven by frequency bands and spectral texture.

### Metadata block
```
metadata.bpm
metadata.duration
metadata.fps
metadata.n_frames
```
No `structure` or `key` blocks.

### Per-frame fields
```
frame_index, time

# Triggers
is_beat, is_onset

# Core energy  (all [0,1])
percussive_impact, harmonic_energy, global_energy, spectral_flux

# 7-band frequency energy  (all [0,1])
sub_bass, bass, low_mid, mid, high_mid, presence, brilliance

# Spectral texture  (all [0,1])
spectral_brightness, spectral_flatness, spectral_rolloff, zero_crossing_rate

# Tonality
dominant_chroma, chroma_values{C…B}

# Semantic primitives  (all [0,1])
impact, fluidity, brightness, pitch_hue, texture, sharpness
```

---

## Schema 2.0 — current

**`ANALYSIS_VERSION = "2.0"` set in `pipeline.py:35`**
**`schema_version = "2.0"` set in `exporter.py:30`**

Phase 1 of the Audio Intelligence upgrade (features C1–C7 + W4).
All new fields are `null` when the feature was not computed; use `.get(key, default)`.

### New metadata blocks

```json
"structure": {
  "n_sections": 8,
  "section_boundaries": [0.0, 26.4, ...],
  "section_durations":  [26.4, 26.4, ...]
},
"key": {
  "root": "F",
  "root_index": 5,
  "mode": "minor",
  "confidence": 0.84,
  "relative_major": "Ab"
}
```

### New per-frame fields

#### C1 — MFCC / Timbre velocity
| Field | Type | Range | Notes |
|---|---|---|---|
| `timbre_velocity` | float\|null | [0,1] | L2-norm of mfcc_delta. High = morphing texture. |

Full MFCC arrays (`mfcc`, `mfcc_delta`, `mfcc_delta2`, 13 × n_frames each) are in
the companion `.npz` archive, not in the JSON manifest.

#### C2 — Song structure
| Field | Type | Range | Notes |
|---|---|---|---|
| `section_index` | int\|null | 0…n-1 | Which detected section this frame is in |
| `section_novelty` | float\|null | [0,1] | Peaks at section boundaries, decays away |
| `section_progress` | float\|null | [0,1] | How far through the current section |
| `section_change` | bool\|null | | True only on the boundary frame itself |

Implemented via `librosa.segment.agglomerative`. `chroma_stft` used (not `chroma_cqt`)
to avoid OOM on long tracks.

#### C3 — Pitch tracking (pyin)
| Field | Type | Range | Notes |
|---|---|---|---|
| `f0_hz` | float\|null | 65–2093 | Fundamental frequency in Hz. null = unvoiced. |
| `f0_confidence` | float\|null | [0,1] | pyin voicing probability |
| `f0_voiced` | bool\|null | | Whether a pitched note was detected this frame |
| `pitch_register` | float\|null | [0,1] | log(f0) mapped to C2–C7 range. 0 = low, 1 = high. |

pyin runs at `hop_length × 4` to avoid OOM on long tracks, then output is
upsampled back to frame-rate via nearest-neighbour.

#### C4 — Key stability
| Field | Type | Range | Notes |
|---|---|---|---|
| `key_stability` | float\|null | [0,1] | Frame-level Krumhansl-Schmuckler confidence. Drops near key changes. |

Global key (root, mode, confidence) is in `metadata.key`.

#### C5 — Downbeat / bar grid
| Field | Type | Range | Notes |
|---|---|---|---|
| `is_downbeat` | bool\|null | | True on beat 1 of each 4-beat bar |
| `beat_position` | float\|null | [0,1) | Phase within the current beat |
| `bar_index` | int\|null | 0… | Bar number since track start |
| `bar_progress` | float\|null | [0,1] | Phase within the 4-beat bar. 0 on downbeat, 1 just before next. |

Downbeats are heuristic (every 4th beat). Correct on 4/4 material; unreliable otherwise.

#### C6 — CQT bands (opt-in)
| Field | Type | Range | Notes |
|---|---|---|---|
| `sub_bass_cqt` | float\|null | [0,1] | CQT-based sub-bass (better low-frequency resolution) |
| `bass_cqt` | float\|null | [0,1] | CQT-based bass |

Both are `null` by default. Non-null only when `AudioPipeline(use_cqt=True)`.

#### C7 — Spectral bandwidth
| Field | Type | Range | Notes |
|---|---|---|---|
| `spectral_bandwidth` | float\|null | [0,1] | Spectral spread around centroid. Narrow = tonal/laser; wide = diffuse/cloud. |

Also added to semantic primitives as `bandwidth_norm` (same value).

#### W4 — Onset shape classification
| Field | Type | Notes |
|---|---|---|
| `onset_type` | string\|null | `"transient"` (hard kick/snare), `"percussive"` (beat, no sharp attack), `"harmonic"` (tonal entry) |
| `onset_sharpness` | float\|null | [0,1] — 1.0 = instantaneous, 0.0 = slow swell |

---

## Cache invalidation

Manifests are cached at `~/.cache/chromascope/manifests/` by a filename that
encodes:

```
manifest_{sha256_of_audio}_{md5_of_config}.json
```

The config hash includes `ANALYSIS_VERSION` as a field. Bumping the version
automatically invalidates every cached manifest and forces re-analysis.

Old v1.x caches (config hash `2f7ab7fe…` and similar) are silently ignored;
the engine re-analyzes and writes a new v2.0 file. To clear manually:

```python
pipeline.clear_cache()
# or
rm -rf ~/.cache/chromascope/manifests/
```

---

## Backward compatibility guarantee

Every manifest field is consumed via `.get(key, default)` — never direct
`frame["key"]` access. This means:

| Situation | Result |
|---|---|
| Old manifest (1.1) + new renderer | New fields missing → renderer falls back to defaults. Zero breakage. |
| New manifest (2.0) + old renderer | Old renderer ignores unknown fields. Zero breakage. |
| New manifest (2.0) + new renderer | All new fields consumed, full behavior. |

**Invariant:** every `update()` / render call in every consumer must access
manifest fields via `.get()` with a sensible default.

### Null-safe patterns

Python:
```python
section  = frame_data.get("section_index", 0) or 0
downbeat = frame_data.get("is_downbeat", False) or False
register = frame_data.get("pitch_register", 0.5) or 0.5
onset    = frame_data.get("onset_type", "percussive") or "percussive"
bar_prog = frame_data.get("bar_progress", 0.0) or 0.0
```

JavaScript:
```javascript
const schemaVer  = manifest.metadata?.schema_version ?? "1.1";
const hasPhase1  = schemaVer >= "2.0";
const section    = frame.section_index    ?? 0;
const downbeat   = frame.is_downbeat      ?? false;
const register   = frame.pitch_register   ?? 0.5;
const onset      = frame.onset_type       ?? "percussive";
const barProg    = frame.bar_progress     ?? 0.0;
const stability  = frame.key_stability    ?? 1.0;
```

---

## Planned future versions

| Version | Planned additions |
|---|---|
| `3.0` | Stem energy fields (`drums_energy`, `bass_energy`, `vocals_energy`, `other_energy`) via Demucs; neural beat tracking; chord identity (`chord_root`, `chord_quality`, `chord_tension`, `chord_is_change`) |
| `4.0` | CREPE pitch spectrum; per-section arousal/valence; MERT audio embedding (`embedding_3d`) |

`3.0` requires `pip install -e ".[separation]"` (htdemucs, ~1–2 GB model weights).
`4.0` requires `pip install -e ".[neural-beats,harmony"]`.
