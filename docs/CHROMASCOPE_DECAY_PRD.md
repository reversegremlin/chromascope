# Chromascope Decay: PRD & Specification
## "Radioactive Cloud Chamber" Audio-Reactive Renderer

### 1. Vision
An organic, high-energy particle simulation that translates music into a radioactive decay field. It mimics the aesthetics of a Wilson Cloud Chamber but introduces a "Matter-Antimatter" sliding mirror architecture where two independent worlds collide and annihilate in the center.

---

### 2. Core Physics: "The Smokey Engine"
*   **Dual-Buffer Vapor:**
    *   **Track Buffer:** Sharp, high-intensity lines representing the ionization paths.
    *   **Vapor Buffer:** Diffuse, persistent smoke that drifts and distorts over time.
*   **Particle Types:**
    *   **Alpha:** Short, thick, slow. Starts with a high-velocity kick and suffers heavy drag (0.88).
    *   **Beta:** Long, thin, fast. Maintains speed (0.98 drag) and creates sweeping arcs.
    *   **Gamma:** Sparse flashes and point-events (speckles).
*   **Drag Dynamics:** All particles obey a friction-based decay law, causing them to "smoke out" as they lose energy.

---

### 3. Mirror Architecture: "Sliding Planes"
*   **Symmetrical Opposites:** The visual is split into two independent simulations (Identity A and Identity B) that move in exactly opposite directions along a locked axis (Vertical, Horizontal, or Diagonal).
*   **Centered Intersection:** All panning is centered on the screen's midpoint, ensuring that the "collision" always happens in the center.
*   **Cinematic Motion:** Slow, sweeping pans (0.15 - 0.55 phase speed) driven by music energy.
*   **Interference Modes:**
    *   **Resonance:** Multiplicative overlap with a 12.0x intensity boost.
    *   **Constructive:** Additive merging.
    *   **Destructive:** Difference-based annihilation.
    *   **Sweet Spot:** Max-based merging with resonance "sparking" in the center.

---

### 4. Coloring: "Matter-Antimatter Annihilation"
*   **Dynamic Harmony:** Uses triadic offsets to shift trail "tips" based on their distance to the nearest ore.
*   **Annihilation Palette:** When Simulation A and B overlap, the color engine detects the collision mask and triggers a high-contrast shift to **Cyan / Magenta / White-Hot**.
*   **White-Hot Core:** Particle tracks in the overlap zone are boosted to brilliant white, simulating an annihilation reaction.

---

### 5. CLI & Controls
*   `--style`: `uranium` (green/yellow), `neon` (blue/pink), `noir` (B&W).
*   `--mirror`: `vertical`, `horizontal`, `diagonal`, `circular`, `cycle`.
*   `--interference`: `resonance`, `constructive`, `destructive`, `sweet_spot`, `cycle`.
*   `--vapor-persistence`: Controls how long the smoke lingers (0.90 - 0.99).
*   `--distortion`: Controls the intensity of the "heat shimmer" vapor warp.

---

### 6. Technical Implementation
*   **Sub-Pixel Shifting:** Uses Bilinear Interpolation (`map_coordinates`) to ensure 100% jitter-free motion.
*   **Independent Seeds:** Every mirrored instance uses its own local `Random` and `np.random.Generator` states (Seeds: 42 and 1337).
*   **Coordinate Wrapping:** Motion utilizes "Wrap" mode, ensuring the screen is always filled even during large sweeps.
