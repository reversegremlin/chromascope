# Fractal Kaleidoscope Experiment — Implementation Plan

## Creative Goal

An audio-reactive kaleidoscopic fractal video renderer. Audio in, HD/4K MP4 out.
Patterns mirror symmetrically, recursively zoom inward, generate complex fractal
geometry, and feel alive, organic, hypnotic, and cinematic.

**Think:** infinite zoom + symmetry + fractal math + rich color gradients + audio reactivity.

---

## 1. Architecture Overview

```
audio.mp3
    │
    ▼
┌───────────────────────┐
│  AudioPipeline        │  ← existing, unchanged
│  (decompose → analyze │
│   → polish → manifest)│
└───────────┬───────────┘
            │ manifest dict
            ▼
┌───────────────────────┐
│  FractalKaleidoscope  │  ← NEW pure-Python renderer
│  Renderer             │
│                       │
│  ┌─────────────────┐  │
│  │ Fractal Engine   │  │  Julia sets, Mandelbrot zooms,
│  │ (numpy vectorized│  │  noise fields — all GPU-free
│  │  complex-plane)  │  │  but heavily vectorized
│  └────────┬────────┘  │
│           ▼           │
│  ┌─────────────────┐  │
│  │ Kaleidoscope     │  │  Polar transform → radial
│  │ Mirror + Zoom    │  │  mirror → infinite zoom via
│  │ (numpy/Pillow)   │  │  feedback accumulation
│  └────────┬────────┘  │
│           ▼           │
│  ┌─────────────────┐  │
│  │ Color Grading &  │  │  Jewel tones, iridescence,
│  │ Post-Processing  │  │  glow, chromatic aberration
│  └────────┬────────┘  │
│           ▼           │
│  raw RGB frame array  │
└───────────┬───────────┘
            │ numpy array per frame
            ▼
┌───────────────────────┐
│  FFmpeg pipe          │  ← raw frames piped to ffmpeg
│  (subprocess stdin)   │    H.264/H.265, muxed with
│                       │    original audio
└───────────┬───────────┘
            │
            ▼
        output.mp4
```

**Key decisions:**
- Pure Python + numpy + Pillow. No pygame, no Node.js, no OpenGL.
- Pillow only for per-frame post-processing (gaussian blur for glow, compositing).
- Frames piped to ffmpeg via stdin — no temp files, no frame PNGs on disk.
- Audio muxed into final MP4 by ffmpeg in the same pass.
- Reuses the existing `AudioPipeline` for manifest generation — zero changes to core.

---

## 2. New Files

All new code lives under `src/chromascope/experiment/`:

```
src/chromascope/experiment/
├── __init__.py
├── fractal.py          # Fractal texture generators (Julia, Mandelbrot, noise)
├── kaleidoscope.py     # Polar transform, radial mirror, infinite zoom
├── colorgrade.py       # Palette, glow, chromatic aberration, color cycling
├── renderer.py         # Frame-by-frame orchestrator (manifest → numpy arrays)
├── encoder.py          # FFmpeg pipe: frames + audio → MP4
└── cli.py              # CLI entry point
```

No existing files are modified.

---

## 3. Module Specifications

### 3.1 `fractal.py` — Fractal Texture Engine

Generates a 2D texture array from fractal math. All operations vectorized with numpy.

**Functions:**

```python
def julia_set(
    width: int, height: int,
    c: complex,              # Julia constant (audio-driven)
    center: complex,         # Viewport center (drift)
    zoom: float,             # Zoom level (infinite zoom driver)
    max_iter: int = 256,
) -> np.ndarray:
    """Render Julia set escape-time values as float32 array [0,1]."""

def mandelbrot_zoom(
    width: int, height: int,
    center: complex,         # Zoom target point
    zoom: float,
    max_iter: int = 256,
) -> np.ndarray:
    """Render Mandelbrot set at given zoom/center."""

def noise_fractal(
    width: int, height: int,
    time: float,
    octaves: int = 4,
    scale: float = 3.0,
) -> np.ndarray:
    """Multi-octave sine-based fractal noise field."""
```

**Audio mapping:**
| Manifest field | Fractal parameter |
|---|---|
| `harmonic_energy` | Julia constant `c` (smooth drift through interesting c-values) |
| `spectral_brightness` | `max_iter` (brighter → more detail) |
| `global_energy` | Zoom speed (energy accelerates the zoom) |
| `low_energy` | Warp/distortion amplitude |
| `is_beat` | Momentary zoom punch + fractal type blend |

### 3.2 `kaleidoscope.py` — Symmetry & Zoom Engine

Takes a raw fractal texture and applies the kaleidoscope transform.

**Functions:**

```python
def polar_mirror(
    texture: np.ndarray,      # H×W float texture
    num_segments: int = 8,    # Symmetry order (6, 8, 12)
    rotation: float = 0.0,    # Current rotation angle
) -> np.ndarray:
    """Apply radial kaleidoscope mirror via polar coordinate remapping."""

def infinite_zoom_blend(
    current_frame: np.ndarray,
    feedback_buffer: np.ndarray,
    zoom_factor: float = 1.02,     # Per-frame zoom-in amount
    feedback_alpha: float = 0.85,  # Retention of previous frame
) -> np.ndarray:
    """Blend zoomed-in previous frame with new frame for infinite zoom effect."""
```

**How the infinite zoom works:**
1. Take previous frame's output, scale it inward by `zoom_factor` (affine transform via Pillow or scipy.ndimage).
2. Alpha-blend the scaled previous frame behind the new fractal layer.
3. New detail always appears at the edges; the center is always falling inward.
4. `zoom_factor` modulated by `global_energy` — faster zoom on high energy.

**How the kaleidoscope mirror works:**
1. Convert Cartesian (x,y) to polar (r, θ) relative to center.
2. Fold θ into a single segment: `θ_folded = θ % (2π / N)`.
3. Reflect: if `θ_folded > π/N`, reflect it.
4. Sample the source texture at the remapped coordinates.
5. Result: N-way radially symmetric image from a single fractal wedge.

### 3.3 `colorgrade.py` — Color & Post-Processing

Transforms the monochrome fractal escape-time values into rich color.

**Functions:**

```python
def apply_palette(
    escape_values: np.ndarray,   # H×W float [0,1]
    hue_base: float,             # From dominant_chroma → pitch_hue
    time: float,                 # For color cycling
    saturation: float = 0.85,
    contrast: float = 1.5,
) -> np.ndarray:
    """Map escape-time scalars to RGB via jewel-tone HSV palette with cycling."""

def add_glow(
    frame: np.ndarray,          # H×W×3 uint8
    intensity: float = 0.3,
    radius: int = 15,
) -> np.ndarray:
    """Screen-blend a gaussian-blurred copy for bloom/glow effect."""

def chromatic_aberration(
    frame: np.ndarray,
    offset: int = 3,
) -> np.ndarray:
    """Shift R and B channels by ±offset pixels for fringe effect."""
```

**Palette strategy — "jewel tones":**
- Base hue from `pitch_hue` (chroma-driven).
- Saturation high (0.8–1.0), value modulated by escape-time.
- Color cycling: hue rotates slowly with time, shifts sharply on beats.
- High contrast: apply gamma curve to escape values before palette lookup.
- Glow: gaussian blur + screen blend for "neon stained glass" edges.

### 3.4 `renderer.py` — Frame Orchestrator

The main render loop. Reads the manifest, drives fractal/kaleidoscope/color per frame.

```python
class FractalKaleidoscopeRenderer:
    def __init__(self, width=1920, height=1080, fps=60, num_segments=8):
        ...
        self.feedback_buffer = None  # For infinite zoom
        self.accumulated_rotation = 0.0
        self.zoom_level = 1.0
        self.julia_c_path = [...]  # Precomputed interesting c-values

    def render_frame(self, frame_data: dict, frame_index: int) -> np.ndarray:
        """Render one frame → H×W×3 uint8 numpy array."""
        # 1. Extract audio drivers
        # 2. Update zoom_level, rotation, julia_c
        # 3. Generate fractal texture
        # 4. Apply kaleidoscope mirror
        # 5. Infinite zoom blend with feedback buffer
        # 6. Color grade
        # 7. Post-process (glow, aberration)
        # 8. Update feedback buffer
        # 9. Return RGB array

    def render_manifest(self, manifest: dict) -> Iterator[np.ndarray]:
        """Yield frames one at a time (generator for memory efficiency)."""
```

**Motion design (from experiment.txt):**
- Slow base rotation: `accumulated_rotation += 0.01 * (1 + harmonic_energy)`
- Pulsing scale: zoom punch on beats (`zoom_level *= 1.05` on `is_beat`)
- Color cycling: hue drifts with time, jumps on onsets
- Warp: `low_energy` drives a sinusoidal radial warp before mirroring
- Camera drift: Lissajous drift of the fractal viewport center

**Fractal blending / layering:**
- Primary layer: Julia set (driven by harmonic content)
- On high `percussive_impact` (>0.7): blend in Mandelbrot detail
- Noise fractal adds organic texture at low opacity (0.1–0.2)
- Layers composited before kaleidoscope mirror (so mirror multiplies complexity)

### 3.5 `encoder.py` — FFmpeg Video Encoder

Pipes raw RGB frames to ffmpeg, muxes with original audio.

```python
def encode_video(
    frame_iterator: Iterator[np.ndarray],
    audio_path: Path,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 60,
    quality: str = "high",       # high=H.264 slow/crf18, medium=crf23, fast=ultrafast
    duration: float = None,      # Trim audio to match
    progress_callback: callable = None,
    total_frames: int = None,
) -> Path:
    """Pipe frames to ffmpeg, mux with audio, produce MP4."""
```

**FFmpeg command (high quality):**
```
ffmpeg -y \
  -f rawvideo -pix_fmt rgb24 -s {width}x{height} -r {fps} -i pipe:0 \
  -i {audio_path} \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv444p \
  -c:a aac -b:a 192k \
  -shortest \
  {output_path}
```

**4K support:** width=3840, height=2160, same pipeline — numpy handles it.

### 3.6 `cli.py` — Command-Line Interface

```
Usage: python -m chromascope.experiment.cli <audio_file> [options]

Options:
  -o, --output PATH        Output MP4 path (default: <input>_fractal.mp4)
  --width INT              Video width (default: 1920)
  --height INT             Video height (default: 1080)
  -f, --fps INT            Frames per second (default: 60)
  -s, --segments INT       Kaleidoscope symmetry segments (default: 8)
  --max-duration FLOAT     Limit output duration in seconds
  -q, --quality STR        Encoding quality: high|medium|fast (default: high)
  --fractal STR            Fractal type: julia|mandelbrot|blend (default: blend)
  --zoom-speed FLOAT       Base zoom speed multiplier (default: 1.0)
  --rotation-speed FLOAT   Base rotation speed (default: 1.0)
  --no-glow                Disable glow post-processing
  --no-aberration          Disable chromatic aberration
```

---

## 4. Audio-to-Visual Mapping Summary

| Audio Feature | Visual Effect |
|---|---|
| `percussive_impact` | Zoom punch, fractal complexity burst, glow intensity |
| `harmonic_energy` | Rotation speed, Julia c-parameter drift, color saturation |
| `spectral_brightness` | Max iterations (detail level), aberration intensity |
| `global_energy` | Zoom speed, overall visual intensity |
| `low_energy` | Radial warp amplitude ("breathing"), line thickness |
| `mid_energy` | Noise fractal blend opacity |
| `high_energy` | Color cycling speed, segment count modulation |
| `dominant_chroma` / `pitch_hue` | Base hue for color palette |
| `is_beat` | Instantaneous zoom punch + hue shift + glow flash |
| `is_onset` | Minor hue rotation, fractal type crossfade |

---

## 5. Dependencies

**New required packages:**
- `Pillow>=10.0.0` — image transforms (affine zoom, gaussian blur, compositing)

**Already available:**
- `numpy` — all fractal math, array operations
- `scipy` — optional `ndimage.zoom` for subpixel accuracy
- `ffmpeg` — system binary, already installed

**Not needed:**
- pygame (existing renderer only)
- Node.js / canvas (existing web renderer only)
- OpenCV, OpenGL, GPU libraries

---

## 6. Testing Strategy

Test with assets in `tests/assets/`:
- `tests/assets/madonna.mp3` — pop track (beats, melody)
- `tests/assets/flume-getu.mp3` — electronic (heavy bass, textures)
- `tests/assets/flume-voices.mp3` — ambient/vocal (harmonic richness)

**Test plan:**

```
tests/experiment/
├── test_fractal.py       # Unit: Julia/Mandelbrot output shapes, value ranges
├── test_kaleidoscope.py  # Unit: polar mirror symmetry verification
├── test_colorgrade.py    # Unit: palette output ranges, glow dimensions
├── test_renderer.py      # Integration: full frame render from mock manifest
└── test_encoder.py       # Integration: short encode produces valid MP4
```

**Key assertions:**
- Fractal outputs are float32 arrays in [0, 1], correct dimensions
- Kaleidoscope mirror produces N-way symmetry (sample points at equal angles, verify equality)
- Color grading outputs uint8 RGB arrays in [0, 255]
- Encoder produces a valid MP4 file (check file exists, size > 0, ffprobe validates)
- Full pipeline: audio in → MP4 out, file plays without errors

**Quick smoke test (for dev iteration):**
```bash
python -m chromascope.experiment.cli tests/assets/madonna.mp3 \
  --max-duration 5 -q fast --width 640 --height 360 -f 30
```

---

## 7. Implementation Order

1. **`fractal.py`** — Julia set + Mandelbrot core (can test independently with saved PNGs)
2. **`kaleidoscope.py`** — Polar mirror transform (test with static fractal input)
3. **`colorgrade.py`** — Palette mapping + glow + aberration
4. **`encoder.py`** — FFmpeg pipe (test with solid-color frames first)
5. **`renderer.py`** — Wire everything together: manifest → fractal → mirror → color → frame
6. **`cli.py`** — CLI wrapper, argument parsing
7. **Tests** — Unit + integration tests
8. **Tuning** — Adjust audio mapping constants for best visual result

---

## 8. Performance Notes

- **Julia/Mandelbrot at 1080p:** ~50–100ms per frame with numpy vectorized complex arithmetic (no per-pixel Python loops).
- **4K (3840×2160):** ~200–400ms per frame. A 60s clip at 60fps = 3,600 frames → ~12–24 minutes render time. Acceptable for offline video export.
- **Memory:** One frame at 4K RGB = ~24 MB. Feedback buffer doubles that. Total working set ~50–100 MB.
- **FFmpeg pipe** avoids disk I/O for intermediate frames — significant speedup vs writing PNGs.

---

## 9. Scope Boundaries

**In scope:**
- Audio-reactive fractal kaleidoscope video generation
- Julia sets, Mandelbrot zooms, noise fractals
- 8-way (configurable) radial symmetry
- Infinite recursive zoom via feedback buffer
- Jewel-tone color grading with glow and aberration
- HD (1080p) and 4K output
- Audio muxed into final MP4
- CLI interface

**Out of scope (future):**
- Real-time preview / interactive mode
- GPU/shader acceleration
- 3D depth / parallax
- Particle dust overlay
- Seamless loop detection
- Integration with existing frontend/web UI
