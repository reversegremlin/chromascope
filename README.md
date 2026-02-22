# Chromascope: The Geometry of Sound

Chromascope is where **math, harmonics, and chroma** become living motion.
It is a visual instrument powered by a distinctive audio-to-visual engine: music is decomposed, interpreted, polished, and translated into frame-accurate visual control signals that can drive many renderers.

![Chromascope Teaser](docs/assets/preview.gif)

---

## Why Chromascope Feels Different

Most music visualizers map raw FFT bins straight to pixels.
Chromascope is built around a stronger idea: a **manifest-first engine design**.

1. **Decompose** audio into harmonic and percussive roles.
2. **Extract** meaningful musical features (rhythm, timbre, pitch, energy).
3. **Polish** those features for smooth cinematic behavior.
4. **Serialize** into a stable visual driver manifest any renderer can consume.

This gives you visuals that feel less like a scope and more like a choreography.

---

## Visual Showcase

### Motion Studies

| Circuit | Glass | Flower |
| :---: | :---: | :---: |
| ![Circuit demo](docs/assets/demos/preview_circuit.gif) | ![Glass demo](docs/assets/demos/preview_glass.gif) | ![Flower demo](docs/assets/demos/preview_flower.gif) |

| Fibonacci | Spiral | Geometric |
| :---: | :---: | :---: |
| ![Fibonacci demo](docs/assets/demos/preview_fibonacci.gif) | ![Spiral demo](docs/assets/demos/preview_spiral.gif) | ![Geometric demo](docs/assets/demos/preview_geometric.gif) |

### Style Gallery

| Geometric | Orrery | Quark | Sacred |
| :---: | :---: | :---: | :---: |
| ![Geometric style](docs/assets/demos/preview_geometric.png) | ![Orrery style](docs/assets/demos/preview_orrery.png) | ![Quark style](docs/assets/demos/preview_quark.png) | ![Sacred style](docs/assets/demos/preview_sacred.png) |

| DMT | Fluid | Mycelial | Circuit |
| :---: | :---: | :---: | :---: |
| ![DMT style](docs/assets/demos/preview_dmt.png) | ![Fluid style](docs/assets/demos/preview_fluid.png) | ![Mycelial style](docs/assets/demos/preview_mycelial.png) | ![Circuit style](docs/assets/demos/preview_circuit.png) |

---

## The Engine Architecture

### Phase A — Harmonic/Percussive Separation
HPSS splits the waveform into distinct musical roles:
- **Percussive channel** drives impact, impulses, and transients.
- **Harmonic channel** drives tonal flow, color, and sustained motion.

### Phase B — Visual Driver Extraction
Chromascope extracts frame-aligned controls such as:
- Beat + onset timing
- Harmonic/percussive/global energy
- Multi-band energy (sub-bass → brilliance)
- Chroma + dominant note
- Spectral flux, centroid, flatness, rolloff, ZCR, MFCC timbre

### Phase C — Signal Polishing
Raw audio features are reshaped for aesthetics:
- Attack/release envelopes
- Normalization to `[0, 1]`
- Optional BPM-adaptive envelope timing

Result: less flicker, more continuity, better visual musicality.

### Phase D — Manifest Contract
All polished features are emitted as a versioned visual manifest with semantic primitives:
- `impact`
- `fluidity`
- `brightness`
- `pitch_hue`
- `texture`
- `sharpness`

This lets real-time and offline renderers share the same musical control language.

---

## Build With It

Chromascope supports both immediate exploration and cinematic export:

- **Web Studio (`frontend/`)** for interactive style exploration.
- **Python experiment renderers (`src/chromascope/experiment/`)** for high-fidelity offline output.

The same audio analysis can drive both.

![Chromascope Studio Screenshot](docs/assets/studio-screenshot.png)

---

## Quick Start

### Installation
```bash
git clone https://github.com/nadiabellamorris/chromascope.git
cd chromascope
pip install -e .
```

### Generate an audio visual driver manifest
```bash
chromascope path/to/audio.wav --output manifest.json
```

### Run the web interface
```bash
cd frontend
python server.py
# open http://localhost:8000
```

### Render a cinematic experiment
```bash
chromascope-decay my_track.wav --output video.mp4 --style dark_nebula
```

---

## Design Philosophy

- **Beauty from structure:** rhythm, harmony, and timbre each have distinct visual roles.
- **Manifest-first architecture:** analysis and rendering stay decoupled.
- **Musical coherence over noisy reactivity:** polish signals before mapping.
- **Modular by design:** each stage can evolve independently.

Chromascope is not just audio-reactive graphics.
It is a composable engine for turning sound into visual meaning.
