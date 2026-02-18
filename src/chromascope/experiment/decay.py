"""
Audio-reactive cloud chamber decay renderer.

Translates music into a field of radioactive trails:
- Alpha: short, thick, bright
- Beta: long, thin, fast
- Gamma: sparse flashes/speck events
- Central Ore: Pulsating radioactive core spawning decay events.

ULTRA-INTENSE FEATURES:
- Quantum Fission: Core shatters into multiple sub-emitters on transients.
- Cherenkov Radiation: High-speed blue shockwaves following beta particles.
- Weak Force Mutation: Trails decay into other particle types mid-flight (flavor change).
- Quantum Foam: A living, stochastic background of virtual events.
- Spacetime Warping: Massive distortion around the core.
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
    """Represents a single decay event trail."""
    x: float
    y: float
    vx: float
    vy: float
    life: float  # 1.0 to 0.0
    decay_rate: float
    thickness: float
    intensity: float
    type: str  # "alpha", "beta", "gamma", "boson"
    last_x: float = 0.0
    last_y: float = 0.0
    generation: int = 0
    mass: float = 1.0


@dataclass
class DecayConfig:
    """Configuration for the ultra-intense decay renderer."""
    width: int = 1920
    height: int = 1080
    fps: int = 60
    
    # Decay-specific
    base_cpm: int = 15000 
    trail_persistence: float = 0.97
    diffusion: float = 0.1
    ionization_gain: float = 2.0
    style: str = "uranium"
    
    # Post-processing
    glow_enabled: bool = True
    vignette_strength: float = 0.6
    distortion_strength: float = 0.4
    
    # Performance
    max_particles: int = 8000


class DecayRenderer:
    """
    The ultimate expression of radioactive chaos.
    """

    def __init__(self, config: DecayConfig | None = None):
        self.cfg = config or DecayConfig()
        
        # State
        self.particles: List[Particle] = []
        self.fission_cores: List[Tuple[float, float, float]] = [] # x, y, scale
        self.time = 0.0
        
        # Buffers
        self.trail_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.cherenkov_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        
        # Smoothed audio
        self._smooth_energy = 0.1
        self._smooth_percussive = 0.0
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
        fast = 0.45 if is_beat else 0.25
        slow = 0.05

        self._smooth_energy = self._lerp(self._smooth_energy, frame_data.get("global_energy", 0.1), slow)
        self._smooth_percussive = self._lerp(self._smooth_percussive, frame_data.get("percussive_impact", 0.0), fast)
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
            # Randomly pick a fission core or the main center
            if self.fission_cores and random.random() < 0.7:
                center_x, center_y, c_scale = random.choice(self.fission_cores)
            else:
                center_x, center_y, c_scale = self.cfg.width / 2, self.cfg.height / 2, self.ore_scale
            
            base_radius = 60.0 * c_scale
            spawn_angle = random.uniform(0, 2 * math.pi)
            r = base_radius * random.uniform(0.4, 1.3)
            x = center_x + math.cos(spawn_angle) * r
            y = center_y + math.sin(spawn_angle) * r
            angle = spawn_angle + random.uniform(-0.6, 0.6) + self.drift_angle
        else:
            angle = math.atan2(vy, vx) + random.uniform(-0.4, 0.4)

        speed_scale = 0.3 + self._smooth_energy * 0.7
        
        if p_type == "alpha":
            speed = random.uniform(2.0, 6.0) * (1.0 + self._smooth_low * 2) * speed_scale
            life = 1.0
            decay_rate = random.uniform(0.01, 0.03)
            thickness = random.uniform(12.0, 25.0)
            intensity = random.uniform(1.0, 2.0)
            mass = 4.0
        elif p_type == "beta":
            speed = random.uniform(30.0, 80.0) * (0.6 + self._smooth_centroid * 1.2) * speed_scale
            life = 1.0
            decay_rate = random.uniform(0.004, 0.012)
            thickness = random.uniform(2.5, 6.0)
            intensity = random.uniform(0.7, 1.4)
            mass = 0.5
        elif p_type == "boson":
            speed = random.uniform(10.0, 20.0) * speed_scale
            life = 1.0
            decay_rate = 0.15 # Very short lived
            thickness = random.uniform(20.0, 40.0)
            intensity = 2.0
            mass = 10.0
        else: # gamma
            speed = random.uniform(0, 20.0) * speed_scale
            life = 1.0
            decay_rate = 0.08
            thickness = random.uniform(6.0, 18.0)
            intensity = random.uniform(1.2, 2.5)
            mass = 0.0

        nvx = math.cos(angle) * speed
        nvy = math.sin(angle) * speed
        
        p = Particle(
            x=x, y=y, last_x=x, last_y=y,
            vx=nvx, vy=nvy, 
            life=life, decay_rate=decay_rate, 
            thickness=thickness, intensity=intensity,
            type=p_type, generation=gen, mass=mass
        )
        self.particles.append(p)

    def update_particles(self, dt: float):
        new_particles = []
        # Weak force field strength
        weak_force = self._smooth_flux * 15.0
        
        for p in self.particles:
            p.life -= p.decay_rate
            if p.life > 0:
                p.last_x, p.last_y = p.x, p.y
                p.x += p.vx
                p.y += p.vy
                
                # Weak Force Mutation (Flavor Change)
                # Alpha decays into multiple Betas
                if p.type == "alpha" and p.life < 0.5 and random.random() < 0.05:
                    for _ in range(3):
                        self.spawn_particle("beta", p.x, p.y, p.vx, p.vy, p.generation + 1)
                    continue # Alpha is gone
                
                # Boson creates Beta explosion
                if p.type == "boson" and p.life < 0.2:
                    for _ in range(8):
                        self.spawn_particle("beta", p.x, p.y, random.uniform(-1,1), random.uniform(-1,1))
                    continue

                # Chaotic deviation
                p.vx += (math.sin(self.time * 5 + p.y * 0.02) * weak_force * 0.2)
                p.vy += (math.cos(self.time * 4 + p.x * 0.02) * weak_force * 0.2)
                
                # Drag
                p.vx *= 0.985
                p.vy *= 0.985
                new_particles.append(p)
        self.particles = new_particles

    def _draw_fission_ore(self, draw: ImageDraw.Draw, center: tuple[float, float], scale: float, rot_offset: float):
        cx, cy = center
        num_points = 18
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points) + self.ore_rotation + rot_offset
            r = (50.0 + random.uniform(-15, 15) * self._smooth_flux) * scale
            points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
        
        draw.polygon(points, fill=0.8 + self._smooth_brilliance * 0.4)
        # Hot spots
        for _ in range(3):
            hx = cx + random.uniform(-20, 20) * scale
            hy = cy + random.uniform(-20, 20) * scale
            hr = random.uniform(5, 20) * scale
            draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=1.0)

    def _apply_quantum_foam(self, buffer: np.ndarray):
        """Add virtual particle noise (Quantum Foam)."""
        foam = np.random.poisson(0.001 * (1.0 + self._smooth_energy * 5), buffer.shape).astype(np.float32)
        return np.maximum(buffer, foam)

    def _apply_vapor_distortion(self, buffer: np.ndarray) -> np.ndarray:
        h, w = buffer.shape
        strength = self.cfg.distortion_strength * (1.0 + self._smooth_flux * 4.0 + self._smooth_sub_bass * 2.0)
        if strength <= 0.02: return buffer
        
        t = self.time * 4.0
        # Multi-scale distortion
        dx = (np.sin(t + np.linspace(0, 15, w)) * strength * 12 + 
              np.sin(t*2 + np.linspace(0, 40, w)) * strength * 4)
        dy = (np.cos(t*1.3 + np.linspace(0, 15, h)) * strength * 12 + 
              np.cos(t*2.5 + np.linspace(0, 40, h)) * strength * 4)
        
        y, x = self._distortion_offsets
        coords = np.array([y + dy[:, None], x + dx[None, :]])
        return map_coordinates(buffer, coords, order=1, mode='reflect')

    def _apply_styles(self, buffer: np.ndarray, cherenkov: np.ndarray) -> np.ndarray:
        style = self.cfg.style
        
        if style == "uranium":
            # Uncanny spectral uranium: emerald green + violet fringes
            r = buffer * 0.2 + cherenkov * 0.4
            g = buffer * 1.0
            b = buffer * 0.1 + cherenkov * 1.0
            
            # Plasma core
            y, x = np.ogrid[:self.cfg.height, :self.cfg.width]
            for cx, cy, cs in ([(self.cfg.width/2, self.cfg.height/2, self.ore_scale)] + self.fission_cores):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                glow = np.exp(-dist / (90.0 * cs * self.view_zoom))
                r = np.maximum(r, buffer * glow * 1.5)
                b = np.maximum(b, buffer * glow * 0.8)
            
            rgb = np.stack([r, g, b], axis=-1)
        elif style == "neon":
            shift = self.time * 2.0 + self._smooth_flux * 8.0
            r = buffer * (0.6 + 0.4 * math.sin(shift)) + cherenkov * 0.5
            g = buffer * (0.3 + 0.3 * math.cos(shift * 0.7))
            b = buffer * 0.8 + cherenkov * 1.0
            rgb = np.stack([r, g, b], axis=-1)
        else: # noir
            val = np.power(buffer + cherenkov * 0.5, 1.5)
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
        
        # Quantum Fission Logic
        if frame_data.get("is_beat", False) and self._smooth_sub_bass > 0.6:
            # Core Shatters
            self.fission_cores = []
            num_fragments = random.randint(2, 5)
            for _ in range(num_fragments):
                fx = cfg.width / 2 + random.uniform(-200, 200) * self._smooth_sub_bass
                fy = cfg.height / 2 + random.uniform(-200, 200) * self._smooth_sub_bass
                fs = random.uniform(0.3, 0.7) * self.ore_scale
                self.fission_cores.append((fx, fy, fs))
        else:
            # Fragments re-merge
            new_cores = []
            for cx, cy, cs in self.fission_cores:
                nx = self._lerp(cx, cfg.width / 2, 0.1)
                ny = self._lerp(cy, cfg.height / 2, 0.1)
                new_cores.append((nx, ny, cs * 0.98))
            self.fission_cores = [c for c in new_cores if c[2] > 0.1]

        self.ore_rotation += (0.06 + self._smooth_flux * 1.2)
        self.ore_scale = self._lerp(self.ore_scale, 1.0 + self._smooth_sub_bass * 1.8, 0.4)
        self.drift_angle += (self._smooth_centroid - 0.5) * dt * 12.0
        self.view_zoom = self._lerp(self.view_zoom, 1.0 + self._smooth_sub_bass * 0.4 + self._smooth_flux * 0.2, 0.15)

        # 1. Spawning - ABSOLUTE CHAOS
        cpm = cfg.base_cpm * (1.0 + self._smooth_energy * 8.0 + self._smooth_flux * 15.0)
        spawn_prob = (cpm / 60.0) * dt
        num_spawns = np.random.poisson(spawn_prob)
        
        if frame_data.get("is_beat", False):
            num_spawns += int(150 * self._smooth_percussive * cfg.ionization_gain)
            # Spawn a Boson carrier on beats
            self.spawn_particle("boson")

        alpha_prob = self._smooth_low / (self._smooth_low + self._smooth_high + 1e-6)
        for _ in range(num_spawns):
            r = random.random()
            if r < alpha_prob * 0.5: self.spawn_particle("alpha")
            elif r < 0.9: self.spawn_particle("beta")
            else: self.spawn_particle("gamma")

        # 2. Buffers
        p_fade = cfg.trail_persistence * (1.0 - self._smooth_flux * 0.08)
        self.trail_buffer *= p_fade
        self.cherenkov_buffer *= (p_fade * 0.9) # Faster fade for shockwaves
        
        if cfg.diffusion > 0:
            sigma = cfg.diffusion * 10 * (0.4 + self._smooth_energy)
            self.trail_buffer = gaussian_filter(self.trail_buffer, sigma=sigma)

        # 3. Draw
        self.update_particles(dt)
        
        deposit_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        cherenkov_img = Image.new("F", (cfg.width, cfg.height), 0.0)
        draw = ImageDraw.Draw(deposit_img)
        draw_c = ImageDraw.Draw(cherenkov_img)
        
        cx, cy = cfg.width / 2, cfg.height / 2
        
        # Draw Main Ore
        self._draw_fission_ore(draw, (cx, cy), self.ore_scale, 0)
        # Draw Fragments
        for fx, fy, fs in self.fission_cores:
            self._draw_fission_ore(draw, (fx, fy), fs, fx * 0.01)
        
        for p in self.particles:
            f_boost = (1.0 + self._smooth_flux * 2.0)
            color = float(p.intensity * p.life * f_boost)
            thickness = int(p.thickness * (0.6 + p.life * 0.4) * f_boost)
            
            def transform(px, py):
                return (cx + (px - cx) * self.view_zoom, cy + (py - cy) * self.view_zoom)

            tx, ty = transform(p.x, p.y)
            tlx, tly = transform(p.last_x, p.last_y)

            if p.type == "gamma":
                draw.ellipse([tx - thickness, ty - thickness, tx + thickness, ty + thickness], fill=color)
            else:
                draw.line([(tlx, tly), (tx, ty)], fill=color, width=thickness)
                # Cherenkov Shockwave for high-speed Beta
                if p.type == "beta" and (p.vx**2 + p.vy**2) > 2000:
                    draw_c.line([(tlx, tly), (tx, ty)], fill=color * 0.8, width=thickness * 3)

        # Accumulate & Shader
        current_dep = self._apply_vapor_distortion(np.array(deposit_img))
        current_cher = self._apply_vapor_distortion(np.array(cherenkov_img))
        
        self.trail_buffer = np.maximum(self.trail_buffer, current_dep)
        self.cherenkov_buffer = np.maximum(self.cherenkov_buffer, current_cher)
        
        # Quantum Foam background
        self.trail_buffer = self._apply_quantum_foam(self.trail_buffer)

        # 4. Grading
        rgb = self._apply_styles(self.trail_buffer, self.cherenkov_buffer)
        
        if cfg.glow_enabled:
            g_int = 0.55 + self._smooth_flux * 1.0 + self._smooth_sub_bass * 0.6
            rgb = add_glow(rgb, intensity=min(g_int, 0.99), radius=25)
            
        if cfg.vignette_strength > 0:
            v_str = cfg.vignette_strength * (1.8 + self._smooth_sub_bass * 2.5)
            rgb = vignette(rgb, strength=v_str)
            
        return tone_map_soft(rgb)

    def render_manifest(self, manifest: dict[str, Any], progress_callback: callable = None) -> Iterator[np.ndarray]:
        frames = manifest.get("frames", [])
        total = len(frames)
        # Reset State
        self.particles = []
        self.fission_cores = []
        self.trail_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.cherenkov_buffer = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        self.time = 0.0
        self.drift_angle = 0.0
        self.ore_rotation = 0.0
        self.ore_scale = 1.0
        self.view_zoom = 1.0

        for i, frame_data in enumerate(frames):
            frame = self.render_frame(frame_data, i)
            yield frame
            if progress_callback: progress_callback(i + 1, total)
