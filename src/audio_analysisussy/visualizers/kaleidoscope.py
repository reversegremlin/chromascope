"""
Kaleidoscope visualization renderer.

Maps audio features to geometric transformations:
- Percussive Impact → Scale & Thickness (pulse)
- Harmonic Energy → Rotation & Orbit speed
- Spectral Centroid → Polygon complexity (sides)
- Chroma → Hue/Color
"""

import colorsys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pygame


@dataclass
class KaleidoscopeConfig:
    """Configuration for the kaleidoscope renderer."""

    width: int = 1920
    height: int = 1080
    fps: int = 60
    num_mirrors: int = 8  # Radial symmetry count
    trail_alpha: int = 40  # Frame persistence (0-255)
    base_radius: float = 150.0  # Base shape size
    max_scale: float = 1.8  # Maximum pulse scale
    base_thickness: int = 3  # Base line thickness
    max_thickness: int = 12  # Maximum line thickness
    orbit_radius: float = 200.0  # Base orbit distance
    rotation_speed: float = 2.0  # Base rotation multiplier
    min_sides: int = 3  # Triangle
    max_sides: int = 12  # Dodecagon (approaches circle)
    background_color: tuple[int, int, int] = (5, 5, 15)


class KaleidoscopeRenderer:
    """
    Renders kaleidoscopic visuals driven by audio manifest data.

    Creates a geometric visualization where shapes pulse, rotate, and
    change complexity based on the character of the music.
    """

    # Map chroma indices to hue values (0-1 range)
    # C=red, progressing through spectrum
    CHROMA_TO_HUE = {
        0: 0.0,    # C - Red
        1: 0.083,  # C# - Orange-red
        2: 0.167,  # D - Orange
        3: 0.25,   # D# - Yellow
        4: 0.333,  # E - Yellow-green
        5: 0.417,  # F - Green
        6: 0.5,    # F# - Cyan
        7: 0.583,  # G - Light blue
        8: 0.667,  # G# - Blue
        9: 0.75,   # A - Purple
        10: 0.833, # A# - Magenta
        11: 0.917, # B - Pink
    }

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, config: KaleidoscopeConfig | None = None):
        """
        Initialize the renderer.

        Args:
            config: Rendering configuration. Uses defaults if None.
        """
        self.config = config or KaleidoscopeConfig()
        self.accumulated_rotation = 0.0
        self.surface: pygame.Surface | None = None

    def _note_to_hue(self, note: str) -> float:
        """Convert note name to hue value."""
        try:
            idx = self.NOTE_NAMES.index(note)
            return self.CHROMA_TO_HUE[idx]
        except ValueError:
            return 0.5  # Default cyan

    def _hue_to_rgb(self, hue: float, saturation: float = 0.8, value: float = 0.9) -> tuple[int, int, int]:
        """Convert HSV to RGB color tuple."""
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _compute_polygon_points(
        self,
        center: tuple[float, float],
        radius: float,
        num_sides: int,
        rotation: float,
    ) -> list[tuple[float, float]]:
        """Compute vertices of a regular polygon."""
        points = []
        for i in range(num_sides):
            angle = rotation + (2 * math.pi * i / num_sides)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        return points

    def _draw_polygon(
        self,
        surface: pygame.Surface,
        center: tuple[float, float],
        radius: float,
        num_sides: int,
        rotation: float,
        color: tuple[int, int, int],
        thickness: int,
    ):
        """Draw a single polygon."""
        points = self._compute_polygon_points(center, radius, num_sides, rotation)
        if len(points) >= 3:
            pygame.draw.polygon(surface, color, points, thickness)

    def _draw_kaleidoscope(
        self,
        surface: pygame.Surface,
        frame_data: dict[str, Any],
        center: tuple[float, float],
    ):
        """Draw the full kaleidoscope pattern for a single frame."""
        cfg = self.config

        # Extract values from frame
        percussive = frame_data.get("percussive_impact", 0.0)
        harmonic = frame_data.get("harmonic_energy", 0.0)
        brightness = frame_data.get("spectral_brightness", 0.5)
        dominant_chroma = frame_data.get("dominant_chroma", "C")
        is_beat = frame_data.get("is_beat", False)

        # Map audio to visual properties
        # Scale: base + percussive boost
        scale = 1.0 + (percussive * (cfg.max_scale - 1.0))
        if is_beat:
            scale *= 1.1  # Extra pop on beats

        radius = cfg.base_radius * scale

        # Thickness: driven by percussive impact
        thickness = int(cfg.base_thickness + percussive * (cfg.max_thickness - cfg.base_thickness))

        # Rotation: accumulate based on harmonic energy
        rotation_delta = harmonic * cfg.rotation_speed * (math.pi / 30)  # Smoother rotation
        self.accumulated_rotation += rotation_delta

        # Polygon sides: brightness controls complexity
        num_sides = int(cfg.min_sides + brightness * (cfg.max_sides - cfg.min_sides))
        num_sides = max(cfg.min_sides, min(cfg.max_sides, num_sides))

        # Orbit distance: modulated by harmonic
        orbit = cfg.orbit_radius * (0.5 + harmonic * 0.5)

        # Color from chroma
        hue = self._note_to_hue(dominant_chroma)
        base_color = self._hue_to_rgb(hue, 0.85, 0.95)

        # Secondary color (complementary)
        secondary_hue = (hue + 0.5) % 1.0
        secondary_color = self._hue_to_rgb(secondary_hue, 0.7, 0.8)

        # Draw radial mirrors
        for i in range(cfg.num_mirrors):
            mirror_angle = (2 * math.pi * i / cfg.num_mirrors) + self.accumulated_rotation * 0.3

            # Calculate orbit position
            orbit_x = center[0] + orbit * math.cos(mirror_angle)
            orbit_y = center[1] + orbit * math.sin(mirror_angle)

            # Draw outer polygon
            self._draw_polygon(
                surface,
                (orbit_x, orbit_y),
                radius * 0.8,
                num_sides,
                self.accumulated_rotation + mirror_angle,
                base_color,
                thickness,
            )

            # Draw inner polygon (counter-rotating)
            self._draw_polygon(
                surface,
                (orbit_x, orbit_y),
                radius * 0.4,
                max(3, num_sides - 2),
                -self.accumulated_rotation * 1.5 + mirror_angle,
                secondary_color,
                max(1, thickness // 2),
            )

        # Central shape
        self._draw_polygon(
            surface,
            center,
            radius * 0.6,
            num_sides,
            self.accumulated_rotation * 0.5,
            self._hue_to_rgb(hue, 0.9, 1.0),
            thickness + 2,
        )

    def render_frame(
        self,
        frame_data: dict[str, Any],
        previous_surface: pygame.Surface | None = None,
    ) -> pygame.Surface:
        """
        Render a single frame.

        Args:
            frame_data: Frame data from manifest.
            previous_surface: Previous frame for trail effect.

        Returns:
            Rendered pygame Surface.
        """
        cfg = self.config

        # Create or reuse surface
        surface = pygame.Surface((cfg.width, cfg.height))

        # Trail effect: blend with previous frame
        if previous_surface is not None and cfg.trail_alpha > 0:
            # Darken previous frame
            fade = pygame.Surface((cfg.width, cfg.height))
            fade.fill(cfg.background_color)
            fade.set_alpha(255 - cfg.trail_alpha)
            previous_surface.blit(fade, (0, 0))
            surface.blit(previous_surface, (0, 0))
        else:
            surface.fill(cfg.background_color)

        # Draw kaleidoscope
        center = (cfg.width / 2, cfg.height / 2)
        self._draw_kaleidoscope(surface, frame_data, center)

        return surface

    def render_manifest(
        self,
        manifest: dict[str, Any],
        progress_callback: callable = None,
    ) -> list[pygame.Surface]:
        """
        Render all frames from a manifest.

        Args:
            manifest: Complete manifest dictionary.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            List of rendered pygame Surfaces.
        """
        frames = manifest.get("frames", [])
        surfaces = []
        previous = None

        self.accumulated_rotation = 0.0  # Reset rotation

        for i, frame_data in enumerate(frames):
            surface = self.render_frame(frame_data, previous)
            surfaces.append(surface.copy())
            previous = surface

            if progress_callback:
                progress_callback(i + 1, len(frames))

        return surfaces

    def surface_to_array(self, surface: pygame.Surface) -> np.ndarray:
        """Convert pygame surface to numpy array for video encoding."""
        # pygame uses (width, height) but numpy expects (height, width)
        arr = pygame.surfarray.array3d(surface)
        # Transpose from (width, height, 3) to (height, width, 3)
        arr = np.transpose(arr, (1, 0, 2))
        return arr
