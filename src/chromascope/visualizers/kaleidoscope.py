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

from chromascope.visualizers.styles import get_kaleidoscope_style


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
    background_color2: tuple[int, int, int] = (26, 10, 46)  # Secondary gradient color
    dynamic_background: bool = True  # Enable dynamic background effects
    bg_reactivity: float = 0.7  # Background reactivity to audio (0-1)
    bg_particles: bool = True  # Enable particle effects
    bg_pulse: bool = True  # Enable beat pulse rings
    style: str = "geometric"  # Visualization style (geometric, glass, flower, spiral, circuit, fibonacci)


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

        # Apply shared style presets as defaults — only fill in values
        # that the caller didn't explicitly set (i.e. still at class default).
        style_overrides = get_kaleidoscope_style(self.config.style)
        if style_overrides:
            defaults = KaleidoscopeConfig()
            for key, value in style_overrides.items():
                if hasattr(self.config, key) and getattr(self.config, key) == getattr(defaults, key):
                    setattr(self.config, key, value)
        self.accumulated_rotation = 0.0
        self.surface: pygame.Surface | None = None

        # Dynamic background state
        self.gradient_angle = 0.0
        self.pulse_intensity = 0.0
        self.particles = self._init_particles()

        # Smoothed values for fluid animation
        self.smoothed_percussive = 0.0
        self.smoothed_harmonic = 0.3
        self.smoothed_brightness = 0.5

    def _init_particles(self) -> list[dict]:
        """Initialize background particles."""
        import random
        particles = []
        for _ in range(80):
            particles.append({
                'x': random.random() * self.config.width,
                'y': random.random() * self.config.height,
                'size': random.random() * 2 + 0.5,
                'speed': random.random() * 0.5 + 0.1,
                'angle': random.random() * math.pi * 2,
                'brightness': random.random() * 0.5 + 0.3,
                'pulse': random.random() * math.pi * 2
            })
        return particles

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

    def _lerp(self, current: float, target: float, factor: float) -> float:
        """Linear interpolation helper."""
        return current + (target - current) * factor

    def _render_dynamic_background(
        self,
        surface: pygame.Surface,
        frame_data: dict[str, Any],
    ):
        """Render dynamic, music-reactive background."""
        cfg = self.config
        width, height = cfg.width, cfg.height
        reactivity = cfg.bg_reactivity

        # Update gradient angle
        self.gradient_angle += 0.002 * (1 + self.smoothed_harmonic)

        # Update pulse intensity
        is_beat = frame_data.get("is_beat", False)
        if is_beat:
            self.pulse_intensity = self._lerp(self.pulse_intensity, 1.0, 0.8)
        else:
            self.pulse_intensity = self._lerp(self.pulse_intensity, 0.0, 0.05)

        # Calculate gradient blend
        blend_phase = math.sin(self.gradient_angle * 2) * 0.5 + 0.5
        energy_blend = self.smoothed_harmonic * reactivity
        blend = blend_phase * 0.3 + energy_blend * 0.4

        # Interpolate background colors
        c1 = cfg.background_color
        c2 = cfg.background_color2
        mid_color = (
            int(c1[0] + (c2[0] - c1[0]) * blend),
            int(c1[1] + (c2[1] - c1[1]) * blend),
            int(c1[2] + (c2[2] - c1[2]) * blend),
        )

        # Add brightness boost on beats
        brightness_boost = int(self.pulse_intensity * 30 * reactivity)
        boosted_color = (
            min(255, mid_color[0] + brightness_boost),
            min(255, mid_color[1] + brightness_boost),
            min(255, mid_color[2] + brightness_boost),
        )

        # Create gradient effect using concentric circles
        center_x = width // 2 + int(math.sin(self.gradient_angle * 3) * width * 0.1 * reactivity)
        center_y = height // 2 + int(math.cos(self.gradient_angle * 2) * height * 0.1 * reactivity)

        # Draw radial gradient approximation
        max_radius = int(max(width, height) * 0.9 + self.pulse_intensity * 200 * reactivity)
        steps = 20

        for i in range(steps, 0, -1):
            ratio = i / steps
            radius = int(max_radius * ratio)
            # Interpolate from boosted center to dark edge
            step_color = (
                int(boosted_color[0] * ratio + c1[0] * (1 - ratio)),
                int(boosted_color[1] * ratio + c1[1] * (1 - ratio)),
                int(boosted_color[2] * ratio + c1[2] * (1 - ratio)),
            )
            pygame.draw.circle(surface, step_color, (center_x, center_y), radius)

        # Render particles
        if cfg.bg_particles:
            self._render_particles(surface)

        # Render pulse rings
        if cfg.bg_pulse and self.pulse_intensity > 0.1:
            self._render_pulse_rings(surface)

    def _render_particles(self, surface: pygame.Surface):
        """Render floating particle effects."""
        cfg = self.config
        reactivity = cfg.bg_reactivity
        energy_boost = 1 + self.smoothed_harmonic * 2 * reactivity

        for particle in self.particles:
            # Update position
            particle['x'] += math.cos(particle['angle']) * particle['speed'] * energy_boost
            particle['y'] += math.sin(particle['angle']) * particle['speed'] * energy_boost

            # Wrap around edges
            if particle['x'] < 0:
                particle['x'] = cfg.width
            if particle['x'] > cfg.width:
                particle['x'] = 0
            if particle['y'] < 0:
                particle['y'] = cfg.height
            if particle['y'] > cfg.height:
                particle['y'] = 0

            # Pulse brightness
            particle['pulse'] += 0.05
            pulse_brightness = math.sin(particle['pulse']) * 0.3 + 0.7
            beat_brightness = 1 + self.pulse_intensity * 0.5

            # Calculate alpha and size
            alpha = particle['brightness'] * pulse_brightness * beat_brightness * reactivity
            size = particle['size'] * (1 + self.smoothed_percussive * 0.5)

            # Draw particle
            color_val = int(255 * min(1, alpha))
            pygame.draw.circle(
                surface,
                (color_val, color_val, color_val),
                (int(particle['x']), int(particle['y'])),
                max(1, int(size))
            )

            # Add glow on high energy
            if self.smoothed_percussive > 0.5:
                glow_alpha = (self.smoothed_percussive - 0.5) * alpha * 0.3
                glow_val = int(255 * min(1, glow_alpha))
                if glow_val > 10:
                    pygame.draw.circle(
                        surface,
                        (glow_val, glow_val, glow_val),
                        (int(particle['x']), int(particle['y'])),
                        max(1, int(size * 3))
                    )

    def _render_pulse_rings(self, surface: pygame.Surface):
        """Render expanding pulse rings on beats."""
        cfg = self.config
        center = (cfg.width // 2, cfg.height // 2)
        max_radius = int(max(cfg.width, cfg.height) * 0.6)

        # Get accent color
        accent = (245, 158, 11)  # Default amber

        for i in range(3):
            phase = (1 - self.pulse_intensity + i * 0.2) % 1
            radius = int(phase * max_radius)
            alpha = self.pulse_intensity * 0.15 * (1 - phase)

            if alpha > 0.01 and radius > 0:
                color_val = int(255 * min(1, alpha))
                ring_color = (
                    int(accent[0] * alpha),
                    int(accent[1] * alpha),
                    int(accent[2] * alpha)
                )
                pygame.draw.circle(surface, ring_color, center, radius, 2)

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

        if cfg.style == "fractal":
            self._draw_fractal_kaleidoscope(
                surface=surface,
                center=center,
                radius=radius,
                hue=hue,
                thickness=thickness,
                percussive=percussive,
                harmonic=harmonic,
                brightness=brightness,
            )
            return

        if cfg.style == "glass":
            self._draw_glass_kaleidoscope(
                surface=surface,
                center=center,
                radius=radius,
                orbit=orbit,
                num_sides=num_sides,
                thickness=thickness,
                hue=hue,
                percussive=percussive,
                harmonic=harmonic,
                brightness=brightness,
                base_color=base_color,
                secondary_color=secondary_color,
                is_beat=is_beat,
            )
            return

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

    def _draw_fractal_kaleidoscope(
        self,
        surface: pygame.Surface,
        center: tuple[float, float],
        radius: float,
        hue: float,
        thickness: int,
        percussive: float,
        harmonic: float,
        brightness: float,
    ):
        """Draw a recursive mirrored fractal distinct from geometric style."""
        cfg = self.config
        mirror_count = max(6, cfg.num_mirrors)
        max_depth = max(4, min(7, cfg.max_sides // 3))
        base_length = radius * (0.55 + percussive * 0.35)
        branch_scale = 0.62 + harmonic * 0.16

        def branch(start: tuple[float, float], length: float, angle: float, depth: int, hue_shift: float):
            if depth <= 0 or length < 2:
                return

            bend = math.sin(self.accumulated_rotation * (1.5 + depth * 0.35) + depth + hue_shift) * (0.15 + percussive * 0.12)
            end = (
                start[0] + math.cos(angle + bend) * length,
                start[1] + math.sin(angle + bend) * length,
            )
            line_hue = (hue + hue_shift) % 1.0
            line_color = self._hue_to_rgb(line_hue, 0.95, 0.45 + depth * 0.08 + brightness * 0.2)
            pygame.draw.line(
                surface,
                line_color,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                max(1, int(thickness * (0.18 + depth * 0.12))),
            )

            if depth <= 2:
                node = self._hue_to_rgb((line_hue + 0.08) % 1.0, 1.0, 0.95)
                pygame.draw.circle(surface, node, (int(end[0]), int(end[1])), max(1, int(1.2 + (max_depth - depth) * 0.8)))

            split = 0.42 + percussive * 0.18
            branch(end, length * branch_scale, angle + split, depth - 1, hue_shift + 0.06)
            branch(end, length * branch_scale, angle - split, depth - 1, hue_shift - 0.05)
            if depth > 3:
                branch(end, length * (branch_scale - 0.08), angle + bend * 0.6, depth - 2, hue_shift + 0.12)

        for i in range(mirror_count):
            a = (2 * math.pi * i / mirror_count) + self.accumulated_rotation * 0.35
            branch(center, base_length, a - math.pi / 2, max_depth, i / mirror_count)

        spikes = cfg.min_sides + 6
        pts = []
        for i in range(spikes):
            a = (2 * math.pi * i / spikes) + self.accumulated_rotation * 0.8
            rr = radius * (0.08 + (0.06 if i % 2 else 0.12)) * (1 + percussive * 0.25)
            pts.append((center[0] + math.cos(a) * rr, center[1] + math.sin(a) * rr))

        core_fill = self._hue_to_rgb((hue + 0.53) % 1.0, 1.0, 0.75 + brightness * 0.2)
        core_edge = self._hue_to_rgb((hue + 0.08) % 1.0, 1.0, 0.9)
        pygame.draw.polygon(surface, core_fill, pts, 0)
        pygame.draw.polygon(surface, core_edge, pts, max(1, int(thickness * 0.35)))

    def _draw_glass_kaleidoscope(
        self,
        surface: pygame.Surface,
        center: tuple[float, float],
        radius: float,
        orbit: float,
        num_sides: int,
        thickness: int,
        hue: float,
        percussive: float,
        harmonic: float,
        brightness: float,
        base_color: tuple[int, int, int],
        secondary_color: tuple[int, int, int],
        is_beat: bool,
    ):
        """Draw a densely layered psychedelic kaleidoscope inspired by stained-glass mandalas."""
        cfg = self.config
        width, height = cfg.width, cfg.height
        mirror_count = max(10, cfg.num_mirrors)
        max_reach = max(width, height) * 0.76
        beat_boost = 1.0 + (0.22 if is_beat else 0.0) + percussive * 0.28
        rings = max(7, min(14, cfg.max_sides // 2 + 3))

        fx = pygame.Surface((width, height), pygame.SRCALPHA)

        violet_hue = (hue + 0.68) % 1.0
        cyan_hue = (hue + 0.48) % 1.0
        magenta_hue = (hue + 0.87) % 1.0
        amber_hue = (hue + 0.10) % 1.0

        # Stained wash to keep the whole frame alive.
        for i in range(9, 0, -1):
            rp = i / 9
            rr = int(max_reach * (0.15 + rp * 0.92))
            wash_hue = [violet_hue, cyan_hue, magenta_hue, amber_hue][i % 4]
            wash = self._hue_to_rgb(wash_hue, 0.78, 0.28 + rp * 0.35)
            alpha = int(8 + rp * 22)
            pygame.draw.circle(
                fx,
                (wash[0], wash[1], wash[2], alpha),
                (int(center[0]), int(center[1])),
                max(2, rr),
                0,
            )

        # Layer 1: mirrored lace filaments across the frame.
        wedge = (2 * math.pi) / mirror_count
        filament_segments = max(14, min(40, cfg.max_sides * 2))
        for ring in range(rings):
            rp = (ring + 1) / rings
            ring_r = max_reach * (0.12 + rp * 0.9) * beat_boost
            ring_spin = self.accumulated_rotation * (0.24 + rp * 1.4)
            line_thickness = max(1, int(thickness * (0.2 + rp * 0.32)))

            for i in range(mirror_count):
                axis = i * wedge + ring_spin
                mirror_sign = -1 if i % 2 else 1
                points: list[tuple[float, float]] = []

                for s in range(filament_segments + 1):
                    t = s / filament_segments
                    wave = math.sin(
                        self.accumulated_rotation * (2.5 + rp)
                        + t * math.pi * (8 + rp * 6)
                        + i * 0.9
                    )
                    flutter = math.cos(
                        self.accumulated_rotation * (1.7 + harmonic)
                        + t * math.pi * (5 + rp * 7)
                        + ring * 0.6
                    )
                    theta = axis + mirror_sign * (t - 0.5) * wedge * 1.04 + flutter * 0.08
                    radial = ring_r * (0.14 + 0.96 * t + wave * 0.08)
                    points.append((center[0] + math.cos(theta) * radial, center[1] + math.sin(theta) * radial))

                if len(points) > 1:
                    filament_hue = [violet_hue, cyan_hue, magenta_hue, amber_hue][(ring + i) % 4]
                    glow = self._hue_to_rgb(filament_hue, 0.9, 0.55 + rp * 0.25)
                    core = self._hue_to_rgb((filament_hue + 0.04) % 1.0, 0.98, 0.72 + rp * 0.22)
                    pygame.draw.lines(
                        fx,
                        (glow[0], glow[1], glow[2], int(35 + rp * 60)),
                        False,
                        points,
                        max(1, line_thickness * 2),
                    )
                    pygame.draw.lines(
                        fx,
                        (core[0], core[1], core[2], int(95 + rp * 120)),
                        False,
                        points,
                        line_thickness,
                    )

                spark_count = max(4, min(10, cfg.min_sides + 2))
                for n in range(1, spark_count + 1):
                    idx = int((n / (spark_count + 1)) * (len(points) - 1))
                    sx, sy = points[idx]
                    spark_hue = [cyan_hue, amber_hue, magenta_hue, violet_hue][(n + i) % 4]
                    spark = self._hue_to_rgb(spark_hue, 1.0, 0.9)
                    pygame.draw.circle(
                        fx,
                        (spark[0], spark[1], spark[2], 170),
                        (int(sx), int(sy)),
                        max(1, int(1 + rp * 2 + percussive * 1.5)),
                    )

        # Layer 2: cluster clouds + shard sparks for organic texture.
        clusters = mirror_count * 2
        for i in range(clusters):
            axis = (2 * math.pi * i) / clusters + self.accumulated_rotation * 0.35
            cluster_r = max_reach * (0.34 + (i % 7) * 0.08)
            cx = center[0] + math.cos(axis) * cluster_r
            cy = center[1] + math.sin(axis) * cluster_r

            dots = 24 + min(40, cfg.max_sides * 2)
            for j in range(dots):
                seed = i * 97 + j * 37
                rnd = math.sin(seed * 12.9898 + 78.233) * 43758.5453
                rnd = rnd - math.floor(rnd)
                rnd2 = math.sin((seed + 1) * 12.9898 + 78.233) * 43758.5453
                rnd2 = rnd2 - math.floor(rnd2)
                rr = (8 + rnd * (32 + brightness * 30)) * (0.7 + percussive * 0.45)
                aa = rnd2 * math.pi * 2
                px = cx + math.cos(aa) * rr
                py = cy + math.sin(aa) * rr
                dot_hue = [cyan_hue, amber_hue, magenta_hue, violet_hue][(j + i) % 4]
                dot = self._hue_to_rgb((dot_hue + rnd * 0.05) % 1.0, 0.98, 0.58 + rnd * 0.35)
                dot_size = max(1, int(1 + rnd2 * (2 + brightness * 2)))
                pygame.draw.circle(fx, (dot[0], dot[1], dot[2], int(65 + rnd * 120)), (int(px), int(py)), dot_size)

            shard_count = 9 + int(percussive * 8)
            for s in range(shard_count):
                sa = axis + (s / shard_count - 0.5) * 1.2
                length = 14 + brightness * 18 + (s % 3) * 6
                sx = cx + math.cos(sa) * (20 + (s % 4) * 8)
                sy = cy + math.sin(sa) * (20 + (s % 4) * 8)
                ex = sx + math.cos(sa + 0.15) * length
                ey = sy + math.sin(sa + 0.15) * length
                shard_hue = (amber_hue + s * 0.025) % 1.0
                shard = self._hue_to_rgb(shard_hue, 1.0, 0.78 + brightness * 0.18)
                pygame.draw.line(
                    fx,
                    (shard[0], shard[1], shard[2], 95),
                    (int(sx), int(sy)),
                    (int(ex), int(ey)),
                    max(1, int(thickness * 0.12)),
                )

        # Layer 3: ornate compact center (no giant opaque disk).
        center_layers = 7
        for layer in range(center_layers):
            lp = (layer + 1) / center_layers
            points_count = 14 + layer * 4
            base_r = radius * (0.06 + lp * 0.28) * beat_boost
            points = []
            for k in range(points_count):
                a = (2 * math.pi * k) / points_count + self.accumulated_rotation * (0.4 + lp * 1.1)
                pulse = 1 + math.sin(a * (3 + layer) + self.accumulated_rotation * 3.2) * (0.11 + lp * 0.05)
                rr = base_r * pulse
                points.append((center[0] + math.cos(a) * rr, center[1] + math.sin(a) * rr))

            center_hue = [cyan_hue, amber_hue, magenta_hue, violet_hue][layer % 4]
            fill = self._hue_to_rgb(center_hue, 0.95, 0.45 + lp * 0.3)
            edge = self._hue_to_rgb((center_hue + 0.08) % 1.0, 1.0, 0.68 + lp * 0.18)
            pygame.draw.polygon(fx, (fill[0], fill[1], fill[2], int(30 + lp * 50)), points, 0)
            pygame.draw.polygon(fx, (edge[0], edge[1], edge[2], int(60 + lp * 70)), points, max(1, int(thickness * 0.16)))

        nucleus_hue = amber_hue
        nucleus = self._hue_to_rgb(nucleus_hue, 1.0, 0.96)
        nucleus_radius = int(max(6, radius * (0.028 + percussive * 0.018)))
        pygame.draw.circle(
            fx,
            (nucleus[0], nucleus[1], nucleus[2], 235),
            (int(center[0]), int(center[1])),
            nucleus_radius,
        )

        # Halo bloom around the nucleus.
        for i in range(5, 0, -1):
            rp = i / 5
            rr = int(nucleus_radius * (1.5 + rp * 5.5))
            bloom = self._hue_to_rgb((cyan_hue + rp * 0.18) % 1.0, 1.0, 0.86 - rp * 0.25)
            pygame.draw.circle(
                fx,
                (bloom[0], bloom[1], bloom[2], int(20 + rp * 35)),
                (int(center[0]), int(center[1])),
                max(2, rr),
                0,
            )

        surface.blit(fx, (0, 0))

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

        # Smooth incoming values for fluid animation
        percussive = frame_data.get("percussive_impact", 0.1)
        harmonic = frame_data.get("harmonic_energy", 0.3)
        brightness = frame_data.get("spectral_brightness", 0.5)
        is_beat = frame_data.get("is_beat", False)

        smooth_factor = 0.15
        self.smoothed_percussive = self._lerp(
            self.smoothed_percussive, percussive,
            0.5 if is_beat else smooth_factor
        )
        self.smoothed_harmonic = self._lerp(
            self.smoothed_harmonic, harmonic, smooth_factor * 0.5
        )
        self.smoothed_brightness = self._lerp(
            self.smoothed_brightness, brightness, smooth_factor * 0.3
        )

        # Create or reuse surface
        surface = pygame.Surface((cfg.width, cfg.height))

        # Trail effect: blend with previous frame
        if previous_surface is not None and cfg.trail_alpha > 0:
            # Darken previous frame
            fade = pygame.Surface((cfg.width, cfg.height))
            fade.fill(cfg.background_color)
            fade_alpha = int((100 - cfg.trail_alpha) / 100 * 80) + 5
            fade.set_alpha(fade_alpha)
            previous_surface.blit(fade, (0, 0))
            surface.blit(previous_surface, (0, 0))
        else:
            surface.fill(cfg.background_color)

        # Render dynamic background if enabled
        if cfg.dynamic_background:
            self._render_dynamic_background(surface, frame_data)

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
