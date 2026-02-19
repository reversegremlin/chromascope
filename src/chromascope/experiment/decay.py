"""
Audio-reactive cloud chamber decay renderer.

Translates music into a field of radioactive trails:
- Alpha: short, thick, bright (starts fast, slows down rapidly)
- Beta: long, thin, fast (sustained velocity, subtle drag)
- Gamma: sparse flashes/speck events
- Central Ore: Pulsating radioactive core spawning decay events.

Aesthetic focus:
- Drag Dynamics: Particles decelerate as they lose energy in the vapor.
- Smokey Trails: Dual-buffer system for sharp tracks and diffuse lingering vapor.
- Harmonic Sync: Diffusion and trail "wobble" tied to musical harmonics.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates

from chromascope.experiment.colorgrade import (
    add_glow,
    tone_map_soft,
    vignette,
)


@dataclass
class Particle:
    """Represents a single decay event trail with drag dynamics."""
    x: float
    y: float
    vx: float
    vy: float
    life: float  # 1.0 to 0.0
    decay_rate: float
    thickness: float
    intensity: float
    type: str  # "alpha", "beta", "gamma"
    last_x: float = 0.0
    last_y: float = 0.0
    drag: float = 0.95
    generation: int = 0


@dataclass
class DecayConfig:
    """Configuration for the 'smokey' decay renderer."""
    width: int = 1920
    height: int = 1080
    fps: int = 60
    
    # Decay-specific
    base_cpm: int = 12000 
    trail_persistence: float = 0.95
    vapor_persistence: float = 0.98 # Vapor lingers longer
    base_diffusion: float = 0.08
    ionization_gain: float = 1.2
    style: str = "uranium"
    
    # Post-processing
    glow_enabled: bool = True
    vignette_strength: float = 0.4
    distortion_strength: float = 0.15
    
    # Performance
    max_particles: int = 6000


class DecayRenderer:
    """
    Renders organic, smokey decay trails with drag and harmonic reactivity.
    """

    def __init__(self, config: DecayConfig | None = None):
        self.cfg = config or DecayConfig()
        
        # State
        self.particles: List[Particle] = []
        self.time = 0.0
        
        # Buffers: track (sharp) and vapor (diffuse)
        self.track_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.vapor_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        
        # Smoothed audio
        self._smooth_energy = 0.1
        self._smooth_percussive = 0.0
        self._smooth_harmonic = 0.2
        self._smooth_low = 0.1
        self._smooth_high = 0.1
        self._smooth_flux = 0.0
        self._smooth_sharpness = 0.0
        self._smooth_centroid = 0.5
        self._smooth_sub_bass = 0.0
        self._smooth_brilliance = 0.0

        # Dynamics
        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0
        
        self._distortion_offsets = np.mgrid[0:self.cfg.height, 0:self.cfg.width].astype(np.float32)

    def _lerp(self, current: float, target: float, factor: float) -> float:
        return current + (target - current) * factor

    def _smooth_audio(self, frame_data: dict[str, Any]):
        """Update smoothed audio values."""
        is_beat = frame_data.get("is_beat", False)
        fast = 0.35 if is_beat else 0.18
        slow = 0.06

        self._smooth_energy = self._lerp(self._smooth_energy, frame_data.get("global_energy", 0.1), slow)
        self._smooth_percussive = self._lerp(self._smooth_percussive, frame_data.get("percussive_impact", 0.0), fast)
        self._smooth_harmonic = self._lerp(self._smooth_harmonic, frame_data.get("harmonic_energy", 0.2), slow)
        self._smooth_low = self._lerp(self._smooth_low, frame_data.get("low_energy", 0.1), slow)
        self._smooth_high = self._lerp(self._smooth_high, frame_data.get("high_energy", 0.1), slow)
        self._smooth_flux = self._lerp(self._smooth_flux, frame_data.get("spectral_flux", 0.0), fast)
        self._smooth_sharpness = self._lerp(self._smooth_sharpness, frame_data.get("sharpness", 0.0), slow)
        self._smooth_centroid = self._lerp(self._smooth_centroid, frame_data.get("spectral_centroid", 0.5), slow)
        self._smooth_sub_bass = self._lerp(self._smooth_sub_bass, frame_data.get("sub_bass", 0.0), fast)
        self._smooth_brilliance = self._lerp(self._smooth_brilliance, frame_data.get("brilliance", 0.0), fast)

    def spawn_particle(self, p_type: str, x: float = None, y: float = None, 
                       vx: float = None, vy: float = None, gen: int = 0):
        if len(self.particles) >= self.cfg.max_particles:
            return

        if x is None:
            center_x, center_y = self.cfg.width / 2, self.cfg.height / 2
            base_radius = 50.0 * self.ore_scale
            spawn_angle = random.uniform(0, 2 * math.pi)
            r = base_radius * random.uniform(0.7, 1.1)
            x = center_x + math.cos(spawn_angle) * r
            y = center_y + math.sin(spawn_angle) * r
            angle = spawn_angle + random.uniform(-0.4, 0.4) + self.drift_angle
        else:
            angle = math.atan2(vy, vx) + random.uniform(-0.3, 0.3)

        # Initial velocity "kick" on beats
        kick = 1.0 + self._smooth_percussive * 1.5
        
        if p_type == "alpha":
            speed = random.uniform(5.0, 15.0) * kick
            life = 1.0
            decay_rate = random.uniform(0.02, 0.05)
            thickness = random.uniform(8.0, 16.0)
            intensity = random.uniform(1.0, 1.5)
            drag = 0.88 # Heavy drag for Alpha
        elif p_type == "beta":
            speed = random.uniform(30.0, 60.0) * kick
            life = 1.0
            decay_rate = random.uniform(0.005, 0.015)
            thickness = random.uniform(2.0, 4.0)
            intensity = random.uniform(0.7, 1.1)
            drag = 0.98 # Light drag for Beta
        else: # gamma
            speed = random.uniform(0, 10.0)
            life = 1.0
            decay_rate = 0.1
            thickness = random.uniform(4.0, 10.0)
            intensity = random.uniform(1.0, 2.0)
            drag = 0.95

        nvx = math.cos(angle) * speed
        nvy = math.sin(angle) * speed
        
        p = Particle(
            x=x, y=y, last_x=x, last_y=y,
            vx=nvx, vy=nvy, 
            life=life, decay_rate=decay_rate, 
            thickness=thickness, intensity=intensity,
            type=p_type, drag=drag, generation=gen
        )
        self.particles.append(p)

    def update_particles(self, dt: float):
        new_particles = []
        # Harmonic resonance wobble
        h_wobble = self._smooth_harmonic * 4.0
        
        for p in self.particles:
            p.life -= p.decay_rate
            if p.life > 0:
                p.last_x, p.last_y = p.x, p.y
                p.x += p.vx
                p.y += p.vy
                
                # Apply drag (fast-then-slow)
                # Alpha particles slow down much faster
                p.vx *= p.drag
                p.vy *= p.drag
                
                # Harmonic wobble
                p.vx += random.uniform(-h_wobble, h_wobble)
                p.vy += random.uniform(-h_wobble, h_wobble)
                
                # Secondary branching logic (kept but simplified)
                if p.generation < 1 and random.random() < (0.01 * self._smooth_flux):
                    self.spawn_particle(p.type, p.x, p.y, p.vx, p.vy, p.generation + 1)

                new_particles.append(p)
        self.particles = new_particles

    def _draw_ore(self, draw: ImageDraw.Draw, center: tuple[float, float], scale: float):
        cx, cy = center
        num_points = 16
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points) + self.ore_rotation
            r = (50.0 + random.uniform(-10, 10) * self._smooth_energy) * scale
            points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
        
        # Ore glows with harmonics
        fill = 0.6 + self._smooth_harmonic * 0.4
        draw.polygon(points, fill=fill)
        # Core
        for _ in range(3):
            hr = random.uniform(5, 15) * scale
            draw.ellipse([cx-hr, cy-hr, cx+hr, cy+hr], fill=1.0)

    def _apply_vapor_distortion(self, buffer: np.ndarray) -> np.ndarray:
        strength = self.cfg.distortion_strength * (1.0 + self._smooth_flux * 2.0)
        if strength <= 0.01: return buffer
        h, w = buffer.shape
        t = self.time * 2.5
        dx = np.sin(t + np.linspace(0, 12, w)) * strength * 8
        dy = np.cos(t * 1.1 + np.linspace(0, 12, h)) * strength * 8
        y, x = self._distortion_offsets
        coords = np.array([y + dy[:, None], x + dx[None, :]])
        return map_coordinates(buffer, coords, order=1, mode='reflect')

    def _apply_styles(self, track: np.ndarray, vapor: np.ndarray) -> np.ndarray:
        style = self.cfg.style
        h, w = self.cfg.height, self.cfg.width
        cx, cy = w/2, h/2
        
        # Calculate distance field for color transitions
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Core size baseline
        core_radius = 50.0 * self.ore_scale
        # Transition starts at 1.0x core radius, fully shifts by 2.0x
        # This captures "half or more of the size of the center"
        dist_norm = dist / core_radius
        shift_mask = np.clip((dist_norm - 1.2) / 1.0, 0, 1)
        
        # Harmonic modulation of the color shift
        h_mod = self._smooth_harmonic * 0.5
        shift_mask = np.clip(shift_mask + h_mod, 0, 1)

        if style == "uranium":
            # Inner: Radioactive Emerald
            r_in = track * 0.35 + vapor * 0.1
            g_in = track * 1.0 + vapor * 0.8
            b_in = track * 0.2 + vapor * 0.1
            
            # Outer Tips: Ghostly Cyan/Blue (Harmonious with Emerald)
            r_out = track * 0.1 + vapor * 0.05
            g_out = track * 0.7 + vapor * 0.5
            b_out = track * 1.0 + vapor * 0.9
            
            r = r_in * (1 - shift_mask) + r_out * shift_mask
            g = g_in * (1 - shift_mask) + g_out * shift_mask
            b = b_in * (1 - shift_mask) + b_out * shift_mask
            
            # Sub-bass core heat (Red-shifted green)
            heat = np.exp(-dist / (60.0 * self.ore_scale))
            r = np.maximum(r, (track + vapor) * heat * 0.9)
            rgb = np.stack([r, g, b], axis=-1)
            
        elif style == "neon":
            # Inner: Electric Magenta
            # Cycle base hue slightly with time
            hue_cycle = math.sin(self.time * 0.5) * 0.1
            
            r_in = track * 1.0 + vapor * (0.8 + hue_cycle)
            g_in = track * 0.2 + vapor * 0.1
            b_in = track * 0.9 + vapor * 0.7
            
            # Outer Tips: Deep Violet/Electric Blue
            r_out = track * 0.4 + vapor * 0.2
            g_out = track * 0.1 + vapor * 0.05
            b_out = track * 1.0 + vapor * 1.0
            
            r = r_in * (1 - shift_mask) + r_out * shift_mask
            g = g_in * (1 - shift_mask) + g_out * shift_mask
            b = b_in * (1 - shift_mask) + b_out * shift_mask
            rgb = np.stack([r, g, b], axis=-1)
            
        else: # lab / noir
            val = np.clip(track * 1.2 + vapor * 0.6, 0, 1)
            if style == "noir": 
                val = np.power(val, 1.5)
                # Noir gets a subtle blue tint at the tips for "cold" radioactivity
                r = val * (1 - shift_mask * 0.2)
                g = val * (1 - shift_mask * 0.1)
                b = val
                rgb = np.stack([r, g, b], axis=-1)
            else:
                rgb = np.stack([val, val, val], axis=-1)
            
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def render_frame(
        self,
        frame_data: dict[str, Any],
        frame_index: int,
    ) -> np.ndarray:
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)
        
        # State updates
        self.ore_rotation += (0.04 + self._smooth_harmonic * 0.3)
        self.ore_scale = self._lerp(self.ore_scale, 1.0 + self._smooth_sub_bass * 1.2, 0.4)
        self.drift_angle += (self._smooth_centroid - 0.5) * dt * 6.0
        self.view_zoom = self._lerp(self.view_zoom, 1.0 + self._smooth_sub_bass * 0.3, 0.1)

        # 1. Spawning
        cpm = cfg.base_cpm * (1.0 + self._smooth_energy * 4.0 + self._smooth_flux * 3.0)
        spawn_prob = (cpm / 60.0) * dt
        num_spawns = np.random.poisson(spawn_prob)
        if frame_data.get("is_beat", False):
            num_spawns += int(60 * self._smooth_percussive * cfg.ionization_gain)

        alpha_prob = self._smooth_low / (self._smooth_low + self._smooth_high + 1e-6)
        for _ in range(num_spawns):
            r = random.random()
            if r < alpha_prob * 0.5: self.spawn_particle("alpha")
            elif r < 0.92: self.spawn_particle("beta")
            else: self.spawn_particle("gamma")

        # 2. Update Buffers
        # Sharp tracks fade faster than lingering vapor
        self.track_buffer *= cfg.trail_persistence
        self.vapor_buffer *= cfg.vapor_persistence
        
        # Diffusion pulses with harmonics
        sigma = cfg.base_diffusion * 6 * (0.5 + self._smooth_harmonic)
        self.vapor_buffer = gaussian_filter(self.vapor_buffer, sigma=sigma)

        # 3. Draw
        self.update_particles(dt)
        
        track_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        vapor_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        draw_t = ImageDraw.Draw(track_img)
        draw_v = ImageDraw.Draw(vapor_img)
        
        cx, cy = cfg.width / 2, cfg.height / 2
        self._draw_ore(draw_t, (cx, cy), self.ore_scale)
        
        for p in self.particles:
            color = float(p.intensity * p.life)
            thickness = int(p.thickness * (0.7 + p.life * 0.3))
            
            def transform(px, py):
                return (cx + (px - cx) * self.view_zoom, cy + (py - cy) * self.view_zoom)

            tx, ty = transform(p.x, p.y)
            tlx, tly = transform(p.last_x, p.last_y)

            if p.type == "gamma":
                draw_t.ellipse([tx - thickness, ty - thickness, tx + thickness, ty + thickness], fill=color)
            else:
                # Track is sharp
                draw_t.line([(tlx, tly), (tx, ty)], fill=color, width=max(1, thickness // 2))
                # Vapor is wider/smokey
                draw_v.line([(tlx, tly), (tx, ty)], fill=color * 0.6, width=thickness)

        # 4. Composite
        current_t = self._apply_vapor_distortion(np.array(track_img))
        current_v = self._apply_vapor_distortion(np.array(vapor_img))
        
        self.track_buffer = np.maximum(self.track_buffer, current_t)
        self.vapor_buffer = np.maximum(self.vapor_buffer, current_v)

        # 5. Grading
        rgb = self._apply_styles(self.track_buffer, self.vapor_buffer)
        
        if cfg.glow_enabled:
            g_int = 0.35 + self._smooth_flux * 0.5 + self._smooth_harmonic * 0.3
            rgb = add_glow(rgb, intensity=min(g_int, 0.9), radius=18)
            
        if cfg.vignette_strength > 0:
            v_str = cfg.vignette_strength * (1.0 + self._smooth_sub_bass * 1.5)
            rgb = vignette(rgb, strength=v_str)
            
        return tone_map_soft(rgb)

    def render_manifest(self, manifest: dict[str, Any], progress_callback: callable = None) -> Iterator[np.ndarray]:
        frames = manifest.get("frames", [])
        total = len(frames)
        # Reset State
        self.particles = []
        self.track_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.vapor_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.time = 0.0
        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0

        for i, frame_data in enumerate(frames):
            frame = self.render_frame(frame_data, i)
            yield frame
            if progress_callback: progress_callback(i + 1, total)
