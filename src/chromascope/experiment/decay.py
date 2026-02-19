"""
Audio-reactive cloud chamber decay renderer.

Translates music into a field of radioactive trails:
- Alpha: short, thick, bright (starts fast, slows down rapidly)
- Beta: long, thin, fast (sustained velocity, subtle drag)
- Gamma: sparse flashes/speck events
- Central Ore: Pulsating radioactive core spawning decay events.

Mirror & Interference Features:
- Multi-Instance: Independent simulations with local random states.
- Spatial Pan: Ores can be panned to create complex overlapping fields.
- Mode Cycling: Smooth, music-driven transitions between split and interference modes.
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
    vapor_persistence: float = 0.98 
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
    Truly independent via local random state.
    """

    def __init__(self, config: DecayConfig | None = None, seed: int | None = None, 
                 center_pos: Tuple[float, float] | None = None):
        self.cfg = config or DecayConfig()
        # Local random state for multi-instance independence
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
        self.center_pos = center_pos or (self.cfg.width / 2, self.cfg.height / 2)
        
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
            cx, cy = self.center_pos
            base_radius = 50.0 * self.ore_scale
            spawn_angle = self.rng.uniform(0, 2 * math.pi)
            r = base_radius * self.rng.uniform(0.7, 1.1)
            x = cx + math.cos(spawn_angle) * r
            y = cy + math.sin(spawn_angle) * r
            angle = spawn_angle + self.rng.uniform(-0.4, 0.4) + self.drift_angle
        else:
            angle = math.atan2(vy, vx) + self.rng.uniform(-0.3, 0.3)

        # Initial velocity "kick" on beats
        kick = 1.0 + self._smooth_percussive * 1.5
        
        if p_type == "alpha":
            speed = self.rng.uniform(5.0, 15.0) * kick
            life = 1.0
            decay_rate = self.rng.uniform(0.02, 0.05)
            thickness = self.rng.uniform(8.0, 16.0)
            intensity = self.rng.uniform(1.0, 1.5)
            drag = 0.88 
        elif p_type == "beta":
            speed = self.rng.uniform(30.0, 60.0) * kick
            life = 1.0
            decay_rate = self.rng.uniform(0.005, 0.015)
            thickness = self.rng.uniform(2.0, 4.0)
            intensity = self.rng.uniform(0.7, 1.1)
            drag = 0.98 
        else: # gamma
            speed = self.rng.uniform(0, 10.0)
            life = 1.0
            decay_rate = 0.1
            thickness = self.rng.uniform(4.0, 10.0)
            intensity = self.rng.uniform(1.0, 2.0)
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
        h_wobble = self._smooth_harmonic * 4.0
        
        for p in self.particles:
            p.life -= p.decay_rate
            if p.life > 0:
                p.last_x, p.last_y = p.x, p.y
                p.x += p.vx
                p.y += p.vy
                
                p.vx *= p.drag
                p.vy *= p.drag
                
                p.vx += self.rng.uniform(-h_wobble, h_wobble)
                p.vy += self.rng.uniform(-h_wobble, h_wobble)
                
                if p.generation < 1 and self.rng.random() < (0.01 * self._smooth_flux):
                    self.spawn_particle(p.type, p.x, p.y, p.vx, p.vy, p.generation + 1)

                new_particles.append(p)
        self.particles = new_particles

    def _draw_ore(self, draw: ImageDraw.Draw, center: tuple[float, float], scale: float):
        cx, cy = center
        num_points = 16
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points) + self.ore_rotation
            r = (50.0 + self.rng.uniform(-10, 10) * self._smooth_energy) * scale
            points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
        
        fill = 0.6 + self._smooth_harmonic * 0.4
        draw.polygon(points, fill=fill)
        for _ in range(3):
            hr = self.rng.uniform(5, 15) * scale
            hx = cx + self.rng.uniform(-10, 10) * scale
            hy = cy + self.rng.uniform(-10, 10) * scale
            draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=1.0)

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

    def get_raw_buffers(self, frame_data: dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)
        
        self.ore_rotation += (0.04 + self._smooth_harmonic * 0.3)
        self.ore_scale = self._lerp(self.ore_scale, 1.0 + self._smooth_sub_bass * 1.2, 0.4)
        self.drift_angle += (self._smooth_centroid - 0.5) * dt * 6.0
        self.view_zoom = self._lerp(self.view_zoom, 1.0 + self._smooth_sub_bass * 0.3, 0.1)

        cpm = cfg.base_cpm * (1.0 + self._smooth_energy * 4.0 + self._smooth_flux * 3.0)
        spawn_prob = (cpm / 60.0) * dt
        num_spawns = self.np_rng.poisson(spawn_prob)
        if frame_data.get("is_beat", False):
            num_spawns += int(60 * self._smooth_percussive * cfg.ionization_gain)

        alpha_prob = self._smooth_low / (self._smooth_low + self._smooth_high + 1e-6)
        for _ in range(num_spawns):
            r = self.rng.random()
            if r < alpha_prob * 0.5: self.spawn_particle("alpha")
            elif r < 0.92: self.spawn_particle("beta")
            else: self.spawn_particle("gamma")

        self.track_buffer *= cfg.trail_persistence
        self.vapor_buffer *= cfg.vapor_persistence
        sigma = cfg.base_diffusion * 6 * (0.5 + self._smooth_harmonic)
        self.vapor_buffer = gaussian_filter(self.vapor_buffer, sigma=sigma)

        self.update_particles(dt)
        track_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        vapor_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        draw_t = ImageDraw.Draw(track_img)
        draw_v = ImageDraw.Draw(vapor_img)
        
        # Ores are centered at instance's center_pos
        self._draw_ore(draw_t, self.center_pos, self.ore_scale)
        
        cx, cy = cfg.width / 2, cfg.height / 2
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
                draw_t.line([(tlx, tly), (tx, ty)], fill=color, width=max(1, thickness // 2))
                draw_v.line([(tlx, tly), (tx, ty)], fill=color * 0.6, width=thickness)

        current_t = self._apply_vapor_distortion(np.array(track_img))
        current_v = self._apply_vapor_distortion(np.array(vapor_img))
        self.track_buffer = np.maximum(self.track_buffer, current_t)
        self.vapor_buffer = np.maximum(self.vapor_buffer, current_v)
        return self.track_buffer, self.vapor_buffer

    def _apply_styles(self, track: np.ndarray, vapor: np.ndarray) -> np.ndarray:
        style = self.cfg.style
        h, w = self.cfg.height, self.cfg.width
        cx, cy = w/2, h/2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        core_radius = 50.0 * self.ore_scale
        dist_norm = dist / core_radius
        shift_mask = np.clip((dist_norm - 1.2) / 1.5, 0, 1)
        
        hue_base = (self.time * 0.05 + self._smooth_centroid * 0.5) % 1.0
        hue_tip = (hue_base + 0.33 + self._smooth_harmonic * 0.2) % 1.0

        if style in ["uranium", "neon"]:
            sat_in = 0.8 + self._smooth_energy * 0.2
            sat_out = 0.6 + self._smooth_brilliance * 0.4
            val_in = np.clip(track * 1.5 + vapor * 0.8, 0, 1)
            val_out = np.clip(track * 1.0 + vapor * 1.2, 0, 1)
            rgb_in = self._hsv_to_rgb(hue_base, sat_in, val_in)
            rgb_out = self._hsv_to_rgb(hue_tip, sat_out, val_out)
            rgb = rgb_in * (1 - shift_mask[:,:,None]) + rgb_out * shift_mask[:,:,None]
            heat_mask = np.exp(-dist / (60.0 * self.ore_scale))
            heat_hue = (hue_base - 0.1) % 1.0
            rgb_heat = self._hsv_to_rgb(heat_hue, 1.0, heat_mask * (0.8 + self._smooth_sub_bass))
            rgb = np.maximum(rgb, rgb_heat)
        elif style == "noir":
            val = np.clip(track * 1.3 + vapor * 0.7, 0, 1)
            val = np.power(val, 1.5)
            rgb = self._hsv_to_rgb(hue_base, 0.2, val)
        else: # lab
            val = np.clip(track * 1.2 + vapor * 0.6, 0, 1)
            rgb = np.stack([val, val, val], axis=-1)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def _hsv_to_rgb(self, h: float, s: float, v: np.ndarray) -> np.ndarray:
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        sector = int(h * 6) % 6
        if sector == 0: r, g, b = c, x, 0
        elif sector == 1: r, g, b = x, c, 0
        elif sector == 2: r, g, b = 0, c, x
        elif sector == 3: r, g, b = 0, x, c
        elif sector == 4: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        return np.stack([r + m, g + m, b + m], axis=-1)

    def render_frame(self, frame_data: dict[str, Any], frame_index: int) -> np.ndarray:
        track, vapor = self.get_raw_buffers(frame_data)
        rgb = self._apply_styles(track, vapor)
        if self.cfg.glow_enabled:
            g_int = 0.35 + self._smooth_flux * 0.5 + self._smooth_harmonic * 0.3
            rgb = add_glow(rgb, intensity=min(g_int, 0.9), radius=18)
        if self.cfg.vignette_strength > 0:
            v_str = self.cfg.vignette_strength * (1.0 + self._smooth_sub_bass * 1.5)
            rgb = vignette(rgb, strength=v_str)
        return tone_map_soft(rgb)

    def render_manifest(self, manifest: dict[str, Any], progress_callback: callable = None) -> Iterator[np.ndarray]:
        frames = manifest.get("frames", [])
        total = len(frames)
        self.particles = []
        self.track_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.vapor_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.time = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0
        for i, frame_data in enumerate(frames):
            yield self.render_frame(frame_data, i)
            if progress_callback: progress_callback(i + 1, total)


class MirrorRenderer:
    """
    Manages two independent DecayRenderer instances and composites them
    using spatial splits and interference logic. Supports smooth music-driven cycling.
    """
    MIRROR_MODES = ["vertical", "horizontal", "diagonal", "circular"]
    INT_MODES = ["resonance", "constructive", "destructive", "sweet_spot"]

    def __init__(self, config: DecayConfig, split_mode: str = "vertical", 
                 interference_mode: str = "resonance"):
        self.cfg = config
        self.requested_split = split_mode
        self.requested_int = interference_mode
        self.curr_split_idx = 0 if split_mode == "cycle" else self.MIRROR_MODES.index(split_mode)
        self.next_split_idx = self.curr_split_idx
        self.curr_int_idx = 0 if interference_mode == "cycle" else self.INT_MODES.index(interference_mode)
        self.next_int_idx = self.curr_int_idx
        self.transition_alpha = 0.0 
        self.change_potential = 0.0 
        
        # Pan the simulation centers away from each other for visible overlap
        pan_x = self.cfg.width / 6
        pan_y = self.cfg.height / 6
        self.instance_a = DecayRenderer(config, seed=42, center_pos=(self.cfg.width/2 - pan_x, self.cfg.height/2 - pan_y))
        self.instance_b = DecayRenderer(config, seed=1337, center_pos=(self.cfg.width/2 + pan_x, self.cfg.height/2 + pan_y))
        
        h, w = config.height, config.width
        self.y, self.x = np.ogrid[:h, :w]

    def _get_mask(self, mode: str, pulse: float) -> np.ndarray:
        h, w = self.cfg.height, self.cfg.width
        cx, cy = w/2, h/2
        # Wider gradients for more overlap
        grad = 200.0 
        if mode == "vertical":
            return np.clip(0.5 - (self.x - (cx + pulse)) / grad, 0, 1)
        elif mode == "horizontal":
            return np.clip(0.5 - (self.y - (cy + pulse)) / grad, 0, 1)
        elif mode == "diagonal":
            dist = (self.x - cx) - (self.y - cy) + pulse
            return np.clip(0.5 - dist / grad, 0, 1)
        else: # circular
            r = np.sqrt((self.x - cx)**2 + (self.y - cy)**2)
            boundary = 200.0 * self.instance_a.ore_scale + pulse
            return np.clip(0.5 - (r - boundary) / grad, 0, 1)

    def _apply_interference(self, a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
        if mode == "resonance":
            return (a * b) * 5.0 # High energy where they meet
        elif mode == "constructive":
            return (a + b) * 0.6
        elif mode == "destructive":
            return np.abs(a - b) * 1.5
        else: # sweet_spot
            return np.maximum(a, b) + (a * b * 3.0)

    def render_frame(self, frame_data: dict[str, Any], frame_index: int) -> np.ndarray:
        dt = 1.0 / self.cfg.fps
        energy = frame_data.get("global_energy", 0.1)
        
        if self.requested_split == "cycle" or self.requested_int == "cycle":
            self.change_potential += energy * dt * 1.5 # 3x faster potential growth
            if self.change_potential > 1.0 and self.transition_alpha <= 0:
                self.change_potential = 0
                if self.requested_split == "cycle":
                    self.next_split_idx = (self.curr_split_idx + 1) % len(self.MIRROR_MODES)
                if self.requested_int == "cycle":
                    self.next_int_idx = (self.curr_int_idx + 1) % len(self.INT_MODES)
            if self.next_split_idx != self.curr_split_idx or self.next_int_idx != self.curr_int_idx:
                self.transition_alpha += dt * 0.6 # Slightly faster cross-fade (~1.6s)
                if self.transition_alpha >= 1.0:
                    self.curr_split_idx = self.next_split_idx
                    self.curr_int_idx = self.next_int_idx
                    self.transition_alpha = 0.0

        t_a, v_a = self.instance_a.get_raw_buffers(frame_data)
        t_b, v_b = self.instance_b.get_raw_buffers(frame_data)
        
        pulse = math.sin(self.instance_a.time * 2) * 30 * self.instance_a._smooth_energy
        mask_a_curr = self._get_mask(self.MIRROR_MODES[self.curr_split_idx], pulse)
        mask_a_next = self._get_mask(self.MIRROR_MODES[self.next_split_idx], pulse)
        mask_a = mask_a_curr * (1 - self.transition_alpha) + mask_a_next * self.transition_alpha
        mask_b = 1.0 - mask_a
        
        def blended_interference(a, b):
            int_curr = self._apply_interference(a, b, self.INT_MODES[self.curr_int_idx])
            int_next = self._apply_interference(a, b, self.INT_MODES[self.next_int_idx])
            return int_curr * (1 - self.transition_alpha) + int_next * self.transition_alpha

        # Significant overlap zone
        overlap = np.clip(1.0 - np.abs(mask_a - 0.5) * 1.5, 0, 1)
        
        track_final = (t_a * mask_a * (1-overlap)) + (t_b * mask_b * (1-overlap)) + blended_interference(t_a, t_b) * overlap
        vapor_final = (v_a * mask_a * (1-overlap)) + (v_b * mask_b * (1-overlap)) + blended_interference(v_a, v_b) * overlap
        
        rgb = self.instance_a._apply_styles(track_final, vapor_final)
        cfg = self.cfg
        if cfg.glow_enabled:
            g_int = 0.35 + self.instance_a._smooth_flux * 0.5 + self.instance_a._smooth_harmonic * 0.3
            rgb = add_glow(rgb, intensity=min(g_int, 0.9), radius=18)
        if cfg.vignette_strength > 0:
            v_str = cfg.vignette_strength * (1.0 + self.instance_a._smooth_sub_bass * 1.5)
            rgb = vignette(rgb, strength=v_str)
        return tone_map_soft(rgb)

    def render_manifest(self, manifest: dict[str, Any], progress_callback: callable = None) -> Iterator[np.ndarray]:
        frames = manifest.get("frames", [])
        total = len(frames)
        # Re-init instances to ensure clean state
        pan_x = self.cfg.width / 6
        pan_y = self.cfg.height / 6
        self.instance_a = DecayRenderer(self.cfg, seed=42, center_pos=(self.cfg.width/2 - pan_x, self.cfg.height/2 - pan_y))
        self.instance_b = DecayRenderer(self.cfg, seed=1337, center_pos=(self.cfg.width/2 + pan_x, self.cfg.height/2 + pan_y))
        for i, frame_data in enumerate(frames):
            yield self.render_frame(frame_data, i)
            if progress_callback: progress_callback(i + 1, total)
