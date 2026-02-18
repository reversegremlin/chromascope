"""
Signal smoothing and normalization module.

Applies aesthetic processing to raw audio features to prevent
visual flickering and ensure smooth, organic visuals.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from chromascope.core.analyzer import ExtractedFeatures


@dataclass
class EnvelopeParams:
    """Attack/Release envelope parameters in milliseconds."""

    attack_ms: float = 0.0  # Instant attack
    release_ms: float = 500.0  # 500ms decay


@dataclass
class PolishedFeatures:
    """Smoothed and normalized features ready for visualization."""

    # Per-frame boolean triggers
    is_beat: np.ndarray  # Shape: (n_frames,)
    is_onset: np.ndarray  # Shape: (n_frames,)

    # Smoothed continuous signals [0.0, 1.0]
    percussive_impact: np.ndarray
    harmonic_energy: np.ndarray
    global_energy: np.ndarray
    spectral_flux: np.ndarray

    # Frequency bands [0.0, 1.0]
    sub_bass: np.ndarray
    bass: np.ndarray
    low_mid: np.ndarray
    mid: np.ndarray
    high_mid: np.ndarray
    presence: np.ndarray
    brilliance: np.ndarray
    
    # Legacy bands (optional, but keeping for compatibility)
    low_energy: np.ndarray
    mid_energy: np.ndarray
    high_energy: np.ndarray

    # Tonality/Texture [0.0, 1.0]
    spectral_brightness: np.ndarray
    spectral_flatness: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    chroma: np.ndarray  # Shape: (12, n_frames), normalized

    # Dominant note per frame
    dominant_chroma_indices: np.ndarray

    # Metadata
    n_frames: int
    fps: int
    frame_times: np.ndarray


class SignalPolisher:
    """
    Applies aesthetic smoothing to raw audio features.

    Implements attack/release envelopes and normalization to create
    visually pleasing, flicker-free signals.
    """

    def __init__(
        self,
        fps: int = 60,
        impact_envelope: EnvelopeParams | None = None,
        energy_envelope: EnvelopeParams | None = None,
        adaptive_envelopes: bool = False,
    ):
        """
        Initialize the polisher.

        Args:
            fps: Target frames per second.
            impact_envelope: Envelope for percussive signals (default: 0ms attack, 200ms release).
            energy_envelope: Envelope for continuous energy signals (default: 50ms attack, 300ms release).
        """
        self.fps = fps
        self.impact_envelope = impact_envelope or EnvelopeParams(
            attack_ms=0.0,
            release_ms=200.0,
        )
        self.energy_envelope = energy_envelope or EnvelopeParams(
            attack_ms=50.0,
            release_ms=300.0,
        )
        # When enabled, envelope timings are adaptively scaled based on BPM.
        self.adaptive_envelopes = adaptive_envelopes

    def _ms_to_frames(self, ms: float) -> int:
        """Convert milliseconds to number of frames at current FPS."""
        return max(1, int((ms / 1000.0) * self.fps))

    def normalize(self, signal: np.ndarray, floor: float = 0.001) -> np.ndarray:
        """
        Normalize signal to [0.0, 1.0] range.

        Args:
            signal: Input signal.
            floor: Minimum value to prevent division by zero.

        Returns:
            Normalized signal in [0.0, 1.0].
        """
        min_val = np.min(signal)
        max_val = np.max(signal)
        range_val = max_val - min_val

        if range_val < floor:
            return np.zeros_like(signal)

        normalized = (signal - min_val) / range_val
        return np.clip(normalized, 0.0, 1.0)

    def apply_envelope(
        self,
        signal: np.ndarray,
        params: EnvelopeParams,
    ) -> np.ndarray:
        """
        Apply attack/release envelope to a signal.

        Creates smooth "glow" effects where values jump up quickly
        but fade down slowly.

        Args:
            signal: Input signal (should be normalized first).
            params: Envelope attack/release parameters.

        Returns:
            Envelope-smoothed signal.
        """
        attack_frames = self._ms_to_frames(params.attack_ms)
        release_frames = self._ms_to_frames(params.release_ms)

        output = np.zeros_like(signal)
        current = 0.0

        for i, target in enumerate(signal):
            if target > current:
                # Attack phase - rise towards target
                if attack_frames <= 1:
                    current = target
                else:
                    attack_rate = 1.0 / attack_frames
                    current = current + (target - current) * attack_rate
            else:
                # Release phase - decay towards target
                if release_frames <= 1:
                    current = target
                else:
                    release_rate = 1.0 / release_frames
                    current = current - (current - target) * release_rate

            output[i] = current

        return np.clip(output, 0.0, 1.0)

    def create_beat_array(
        self,
        n_frames: int,
        beat_frames: np.ndarray,
    ) -> np.ndarray:
        """
        Create boolean beat trigger array aligned to frame indices.

        Args:
            n_frames: Total number of output frames.
            beat_frames: Frame indices where beats occur.

        Returns:
            Boolean array with True at beat positions.
        """
        is_beat = np.zeros(n_frames, dtype=bool)
        valid_beats = beat_frames[beat_frames < n_frames]
        is_beat[valid_beats.astype(int)] = True
        return is_beat

    def create_onset_array(
        self,
        n_frames: int,
        onset_frames: np.ndarray,
    ) -> np.ndarray:
        """
        Create boolean onset trigger array.

        Args:
            n_frames: Total number of output frames.
            onset_frames: Frame indices where onsets occur.

        Returns:
            Boolean array with True at onset positions.
        """
        is_onset = np.zeros(n_frames, dtype=bool)
        valid_onsets = onset_frames[onset_frames < n_frames]
        is_onset[valid_onsets.astype(int)] = True
        return is_onset

    def smooth_spectral_centroid(
        self,
        centroid: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """
        Normalize spectral centroid to [0.0, 1.0] as "brightness".

        Maps typical music range (100Hz - 10000Hz) to 0-1.
        """
        # Typical range for music
        min_hz = 100.0
        max_hz = 10000.0

        brightness = (centroid - min_hz) / (max_hz - min_hz)
        brightness = np.clip(brightness, 0.0, 1.0)

        # Apply light smoothing
        return self.apply_envelope(brightness, self.energy_envelope)

    def polish(self, features: ExtractedFeatures) -> PolishedFeatures:
        """
        Apply full aesthetic processing to extracted features.

        Args:
            features: Raw features from FeatureAnalyzer.

        Returns:
            PolishedFeatures ready for visualization.
        """
        n_frames = features.n_frames

        # Boolean triggers
        is_beat = self.create_beat_array(n_frames, features.temporal.beat_frames)
        is_onset = self.create_onset_array(n_frames, features.temporal.onset_frames)

        # Optionally adapt envelope timings based on detected BPM.
        impact_env = self.impact_envelope
        energy_env = self.energy_envelope
        if self.adaptive_envelopes:
            bpm = getattr(features.temporal, "bpm", 120.0) or 120.0
            # Scale release inversely with tempo: slower songs -> longer glow.
            scale = 120.0 / max(bpm, 1.0)
            scale = float(np.clip(scale, 0.5, 2.0))

            impact_env = EnvelopeParams(
                attack_ms=impact_env.attack_ms,
                release_ms=impact_env.release_ms * scale,
            )
            energy_env = EnvelopeParams(
                attack_ms=energy_env.attack_ms,
                release_ms=energy_env.release_ms * scale,
            )

        # Energy signals with envelope smoothing
        percussive_impact = self.apply_envelope(
            self.normalize(features.energy.rms_percussive),
            impact_env,
        )

        harmonic_energy = self.apply_envelope(
            self.normalize(features.energy.rms_harmonic),
            energy_env,
        )

        global_energy = self.apply_envelope(
            self.normalize(features.energy.rms),
            energy_env,
        )

        spectral_flux = self.apply_envelope(
            self.normalize(features.energy.spectral_flux),
            impact_env,
        )

        # 7-band frequency energy
        fb = features.energy.frequency_bands
        sub_bass = self.apply_envelope(self.normalize(fb.sub_bass), energy_env)
        bass = self.apply_envelope(self.normalize(fb.bass), energy_env)
        low_mid = self.apply_envelope(self.normalize(fb.low_mid), energy_env)
        mid = self.apply_envelope(self.normalize(fb.mid), energy_env)
        high_mid = self.apply_envelope(self.normalize(fb.high_mid), energy_env)
        presence = self.apply_envelope(self.normalize(fb.presence), energy_env)
        brilliance = self.apply_envelope(self.normalize(fb.brilliance), energy_env)

        # Legacy bands
        low_energy = self.apply_envelope(self.normalize(fb.low), energy_env)
        mid_energy = self.apply_envelope(self.normalize(fb.mid_aggregate), energy_env)
        high_energy = self.apply_envelope(self.normalize(fb.high), energy_env)

        # Tonality/Texture
        spectral_brightness = self.smooth_spectral_centroid(
            features.tonality.spectral_centroid,
            features.sample_rate,
        )
        
        spectral_flatness = self.apply_envelope(
            self.normalize(features.tonality.spectral_flatness),
            energy_env,
        )
        
        spectral_rolloff = self.apply_envelope(
            self.normalize(features.tonality.spectral_rolloff),
            energy_env,
        )
        
        zero_crossing_rate = self.apply_envelope(
            self.normalize(features.tonality.zero_crossing_rate),
            energy_env,
        )

        # Normalize chroma (each bin independently)
        chroma_normalized = np.zeros_like(features.tonality.chroma)
        for i in range(12):
            chroma_normalized[i] = self.normalize(features.tonality.chroma[i])

        return PolishedFeatures(
            is_beat=is_beat,
            is_onset=is_onset,
            percussive_impact=percussive_impact,
            harmonic_energy=harmonic_energy,
            global_energy=global_energy,
            spectral_flux=spectral_flux,
            sub_bass=sub_bass,
            bass=bass,
            low_mid=low_mid,
            mid=mid,
            high_mid=high_mid,
            presence=presence,
            brilliance=brilliance,
            low_energy=low_energy,
            mid_energy=mid_energy,
            high_energy=high_energy,
            spectral_brightness=spectral_brightness,
            spectral_flatness=spectral_flatness,
            spectral_rolloff=spectral_rolloff,
            zero_crossing_rate=zero_crossing_rate,
            chroma=chroma_normalized,
            dominant_chroma_indices=features.tonality.dominant_chroma_indices,
            n_frames=n_frames,
            fps=self.fps,
            frame_times=features.frame_times,
        )
