# Software Development Life Cycle (SDLC)

To maintain high-fidelity audio processing, every change must move through these five gates.

## Stage 1: Analysis & Impact Assessment
- Review the requirement against `architecture.md`.
- Identify which DSP modules are affected (Decomposer, Analyzer, or Polisher).
- Check for potential synchronization drift (FPS vs Sample Rate).

## Stage 2: Test Specification (Red)
- Create a test file in `/tests` (e.g., `test_analyzer_rms.py`).
- Define "Golden Data": What should the output be for a 1kHz sine wave vs. a drum hit?
- **Goal:** The test must fail initially because the feature doesn't exist yet.

## Stage 3: Implementation (Green)
- Write the minimum amount of code required to satisfy the test.
- Follow the Google Python Style Guide.
- Ensure type hints are applied to all new functions.

## Stage 4: Validation & Polishing (Refactor)
- Run the full test suite: `pytest tests/`.
- Optimize the DSP math for performance (vectorize with NumPy where possible).
- Verify the "Visual Driver Manifest" JSON still validates against the schema.

## Stage 5: Documentation & Handover
- Update `DEVLOG.md` with the "Why" and "How."
- Update `roadmap.md`.
- Commit changes to Git with a descriptive message.
