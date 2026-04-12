"""
Audio augmentations for clinical speech SSL.

Augmentations are categorized as:
- CLINICAL: Augmentations that mimic clinical speech deviations (time-stretch, pitch-shift, formant-warp)
- SAFE: Augmentations that affect recording conditions but not clinical signal (noise, reverb, gain)

The augmentation prediction objective uses CLINICAL augmentations as targets.
The contrastive objective uses SAFE augmentations to create positive pairs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T


class AugmentationType(Enum):
    """Types of augmentations available."""
    # Clinical augmentations (affect speech production characteristics)
    TIME_STRETCH = "time_stretch"
    PITCH_SHIFT = "pitch_shift"
    FORMANT_SHIFT = "formant_shift"
    AMPLITUDE_MOD = "amplitude_mod"
    
    # Safe augmentations (affect recording conditions only)
    ADDITIVE_NOISE = "additive_noise"
    REVERB = "reverb"
    GAIN = "gain"
    LOW_PASS = "low_pass"
    HIGH_PASS = "high_pass"


CLINICAL_AUGMENTATIONS = (
    AugmentationType.TIME_STRETCH,
    AugmentationType.PITCH_SHIFT,
    AugmentationType.FORMANT_SHIFT,
    AugmentationType.AMPLITUDE_MOD,
)

CLINICAL_AUGMENTATION_INDEX = {
    aug: i for i, aug in enumerate(CLINICAL_AUGMENTATIONS)
}

SAFE_AUGMENTATIONS = (
    AugmentationType.ADDITIVE_NOISE,
    # REVERB omitted — not implemented; including it would produce identity
    # views in contrastive learning. TODO: implement with sampled RIRs.
    AugmentationType.GAIN,
    AugmentationType.LOW_PASS,
    AugmentationType.HIGH_PASS,
)


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters."""
    # Time stretch: factor (1.0 = no change, <1 = faster, >1 = slower)
    time_stretch_range: Tuple[float, float] = (0.8, 1.2)
    
    # Pitch shift: semitones
    pitch_shift_range: Tuple[float, float] = (-3.0, 3.0)
    
    # Formant shift: ratio (1.0 = no change)
    formant_shift_range: Tuple[float, float] = (0.9, 1.1)
    
    # Amplitude modulation: depth (0 = no mod, 1 = full mod)
    amplitude_mod_range: Tuple[float, float] = (0.0, 0.3)
    amplitude_mod_freq_range: Tuple[float, float] = (2.0, 8.0)  # Hz
    
    # Additive noise: SNR in dB
    noise_snr_range: Tuple[float, float] = (10.0, 30.0)
    
    # Reverb: room size (0-1)
    reverb_room_size_range: Tuple[float, float] = (0.1, 0.5)
    
    # Gain: dB change
    gain_range: Tuple[float, float] = (-6.0, 6.0)
    
    # Low-pass filter: cutoff frequency Hz
    low_pass_range: Tuple[float, float] = (4000.0, 8000.0)
    
    # High-pass filter: cutoff frequency Hz
    high_pass_range: Tuple[float, float] = (50.0, 200.0)


@dataclass
class RegionAugmentationLabel:
    """Label for a single augmentation applied to a single region."""
    start_sample: int
    end_sample: int
    aug_type: AugmentationType
    param_normalized: float
    is_transition: bool


@dataclass
class AugmentationResult:
    """Result of applying augmentations."""
    waveform: torch.Tensor
    labels: Dict[str, float]  # Aggregated augmentation type -> max abs magnitude (back-compat)
    region_labels: List[RegionAugmentationLabel] = field(default_factory=list)
    regions: Optional[List[Tuple[int, int]]] = None  # Which regions were augmented


class AudioAugmentor(nn.Module):
    """
    Applies augmentations to audio waveforms with known parameters.
    
    This module applies augmentations and returns both the augmented audio
    and the augmentation parameters, which serve as targets for the
    augmentation prediction objective.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        config: Optional[AugmentationConfig] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.config = config or AugmentationConfig()
        
    def _sample_param(self, param_range: Tuple[float, float]) -> float:
        """Sample a parameter uniformly from range."""
        return np.random.uniform(param_range[0], param_range[1])
    
    def _normalize_param(self, value: float, param_range: Tuple[float, float]) -> float:
        """Normalize parameter to [-1, 1] for regression target."""
        min_val, max_val = param_range
        mid = (min_val + max_val) / 2
        half_range = (max_val - min_val) / 2
        return (value - mid) / half_range if half_range > 0 else 0.0
    
    def time_stretch(
        self,
        waveform: torch.Tensor,
        factor: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply time stretching using phase vocoder.
        
        Args:
            waveform: Input waveform [C, T] or [T]
            factor: Stretch factor (>1 = slower, <1 = faster). If None, sampled randomly.
            
        Returns:
            Tuple of (stretched waveform, normalized factor)
        """
        if factor is None:
            factor = self._sample_param(self.config.time_stretch_range)
        
        # Ensure 2D input
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
            
        # Use torchaudio's stretch
        n_fft = 1024
        hop_length = n_fft // 4
        
        # Compute spectrogram
        spec = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
        )
        
        # Phase vocoder
        phase_advance = torch.linspace(
            0,
            np.pi * hop_length,
            spec.shape[-2],
            device=waveform.device,
        )[..., None]
        
        # Interpolate magnitude and phase
        time_steps = torch.arange(
            0,
            spec.shape[-1],
            factor,
            device=waveform.device,
        )
        
        # Simple linear interpolation (more sophisticated methods exist)
        stretched_len = int(spec.shape[-1] / factor)
        indices = torch.linspace(0, spec.shape[-1] - 1, stretched_len, device=waveform.device)
        indices_floor = indices.long().clamp(0, spec.shape[-1] - 2)
        indices_ceil = (indices_floor + 1).clamp(0, spec.shape[-1] - 1)
        weights = (indices - indices_floor.float()).unsqueeze(0).unsqueeze(0)
        
        spec_stretched = (
            spec[..., indices_floor] * (1 - weights) +
            spec[..., indices_ceil] * weights
        )
        
        # Inverse STFT
        stretched = torch.istft(
            spec_stretched,
            n_fft=n_fft,
            hop_length=hop_length,
            length=int(waveform.shape[-1] / factor),
        )
        
        if squeeze:
            stretched = stretched.squeeze(0)
            
        normalized = self._normalize_param(factor, self.config.time_stretch_range)
        return stretched, normalized
    
    def pitch_shift(
        self,
        waveform: torch.Tensor,
        semitones: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply pitch shifting.
        
        Args:
            waveform: Input waveform [C, T] or [T]
            semitones: Number of semitones to shift. If None, sampled randomly.
            
        Returns:
            Tuple of (pitch-shifted waveform, normalized semitones)
        """
        if semitones is None:
            semitones = self._sample_param(self.config.pitch_shift_range)
        
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        
        # Use torchaudio pitch shift
        shifted = AF.pitch_shift(
            waveform,
            self.sample_rate,
            n_steps=semitones,
        )
        
        if squeeze:
            shifted = shifted.squeeze(0)
            
        normalized = self._normalize_param(semitones, self.config.pitch_shift_range)
        return shifted, normalized
    
    def formant_shift(
        self,
        waveform: torch.Tensor,
        ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply formant shifting by resampling then pitch-correcting.
        
        This is a simplified formant shift - for production use, consider
        a proper vocoder-based approach (WORLD, STRAIGHT).
        
        Args:
            waveform: Input waveform [C, T] or [T]
            ratio: Formant shift ratio. If None, sampled randomly.
            
        Returns:
            Tuple of (formant-shifted waveform, normalized ratio)
        """
        if ratio is None:
            ratio = self._sample_param(self.config.formant_shift_range)
        
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
            
        # Resample to shift formants
        intermediate_sr = int(self.sample_rate * ratio)
        resampler_down = T.Resample(self.sample_rate, intermediate_sr)
        resampler_up = T.Resample(intermediate_sr, self.sample_rate)
        
        shifted = resampler_up(resampler_down(waveform))
        
        # Match length
        if shifted.shape[-1] > waveform.shape[-1]:
            shifted = shifted[..., :waveform.shape[-1]]
        elif shifted.shape[-1] < waveform.shape[-1]:
            shifted = F.pad(shifted, (0, waveform.shape[-1] - shifted.shape[-1]))
        
        if squeeze:
            shifted = shifted.squeeze(0)
            
        normalized = self._normalize_param(ratio, self.config.formant_shift_range)
        return shifted, normalized
    
    def amplitude_modulation(
        self,
        waveform: torch.Tensor,
        depth: Optional[float] = None,
        freq: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply amplitude modulation (tremolo-like effect).
        
        Args:
            waveform: Input waveform [C, T] or [T]
            depth: Modulation depth (0-1). If None, sampled randomly.
            freq: Modulation frequency in Hz. If None, sampled randomly.
            
        Returns:
            Tuple of (modulated waveform, normalized depth)
        """
        if depth is None:
            depth = self._sample_param(self.config.amplitude_mod_range)
        if freq is None:
            freq = self._sample_param(self.config.amplitude_mod_freq_range)
        
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
            
        t = torch.arange(waveform.shape[-1], device=waveform.device).float() / self.sample_rate
        modulator = 1 - depth * (1 - torch.cos(2 * np.pi * freq * t)) / 2
        modulated = waveform * modulator
        
        if squeeze:
            modulated = modulated.squeeze(0)
            
        normalized = self._normalize_param(depth, self.config.amplitude_mod_range)
        return modulated, normalized
    
    def add_noise(
        self,
        waveform: torch.Tensor,
        snr_db: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Add Gaussian noise at specified SNR.
        
        Args:
            waveform: Input waveform [C, T] or [T]
            snr_db: Target SNR in dB. If None, sampled randomly.
            
        Returns:
            Tuple of (noisy waveform, normalized SNR)
        """
        if snr_db is None:
            snr_db = self._sample_param(self.config.noise_snr_range)
        
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        noisy = waveform + noise
        
        normalized = self._normalize_param(snr_db, self.config.noise_snr_range)
        return noisy, normalized
    
    def apply_gain(
        self,
        waveform: torch.Tensor,
        gain_db: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply gain change in dB.
        
        Args:
            waveform: Input waveform [C, T] or [T]
            gain_db: Gain in dB. If None, sampled randomly.
            
        Returns:
            Tuple of (gained waveform, normalized gain)
        """
        if gain_db is None:
            gain_db = self._sample_param(self.config.gain_range)
        
        gain_linear = 10 ** (gain_db / 20)
        gained = waveform * gain_linear
        
        normalized = self._normalize_param(gain_db, self.config.gain_range)
        return gained, normalized
    
    def apply_low_pass(
        self,
        waveform: torch.Tensor,
        cutoff_hz: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply low-pass filter.
        
        Args:
            waveform: Input waveform [C, T] or [T]
            cutoff_hz: Cutoff frequency in Hz. If None, sampled randomly.
            
        Returns:
            Tuple of (filtered waveform, normalized cutoff)
        """
        if cutoff_hz is None:
            cutoff_hz = self._sample_param(self.config.low_pass_range)
        
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
            
        filtered = AF.lowpass_biquad(waveform, self.sample_rate, cutoff_hz)
        
        if squeeze:
            filtered = filtered.squeeze(0)
            
        normalized = self._normalize_param(cutoff_hz, self.config.low_pass_range)
        return filtered, normalized
    
    def apply_high_pass(
        self,
        waveform: torch.Tensor,
        cutoff_hz: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply high-pass filter.
        
        Args:
            waveform: Input waveform [C, T] or [T]
            cutoff_hz: Cutoff frequency in Hz. If None, sampled randomly.
            
        Returns:
            Tuple of (filtered waveform, normalized cutoff)
        """
        if cutoff_hz is None:
            cutoff_hz = self._sample_param(self.config.high_pass_range)
        
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
            
        filtered = AF.highpass_biquad(waveform, self.sample_rate, cutoff_hz)
        
        if squeeze:
            filtered = filtered.squeeze(0)
            
        normalized = self._normalize_param(cutoff_hz, self.config.high_pass_range)
        return filtered, normalized


class RegionAugmentor(nn.Module):
    """
    Applies augmentations to specific regions of audio based on phoneme alignments.
    
    This is the core module for the augmentation prediction objective, where
    augmentations are applied to specific phoneme regions (with transition bias)
    and the model must predict which augmentations were applied where.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        config: Optional[AugmentationConfig] = None,
        transition_bias: float = 2.0,
        clinical_only: bool = True,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            config: Augmentation configuration
            transition_bias: Factor by which to increase probability of augmenting transitions
            clinical_only: If True, only apply clinical augmentations (not safe ones)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.augmentor = AudioAugmentor(sample_rate, config)
        self.transition_bias = transition_bias
        self.clinical_only = clinical_only
        
        if clinical_only:
            self.aug_types = list(CLINICAL_AUGMENTATIONS)
        else:
            self.aug_types = list(AugmentationType)
    
    def identify_transitions(
        self,
        gamma: torch.Tensor,
        threshold: float = 0.3,
    ) -> torch.Tensor:
        """
        Identify transition frames from CTC gamma matrix.
        
        Args:
            gamma: CTC alignment posteriors [T, num_phonemes]
            threshold: Entropy threshold for transition detection
            
        Returns:
            Boolean tensor [T] indicating transition frames
        """
        # Compute entropy at each frame
        gamma_safe = gamma.clamp(min=1e-10)
        entropy = -(gamma_safe * gamma_safe.log()).sum(dim=-1)
        max_entropy = np.log(gamma.shape[-1])
        normalized_entropy = entropy / max_entropy
        
        # High entropy = transition (uncertain which phoneme)
        is_transition = normalized_entropy > threshold
        return is_transition
    
    def sample_regions(
        self,
        waveform_length: int,
        gamma: torch.Tensor,
        num_regions: int = 2,
        region_min_frames: int = 10,
    ) -> List[Tuple[int, int, bool]]:
        """
        Sample regions to augment, with bias toward transitions.
        
        Args:
            waveform_length: Length of waveform in samples
            gamma: CTC gamma matrix [T, num_phonemes]
            num_regions: Number of regions to augment
            region_min_frames: Minimum region size in frames
            
        Returns:
            List of (start_sample, end_sample, is_transition) tuples
        """
        num_frames = gamma.shape[0]
        samples_per_frame = waveform_length // num_frames
        
        is_transition = self.identify_transitions(gamma)
        
        # Create sampling weights with transition bias
        weights = torch.ones(num_frames)
        weights[is_transition] *= self.transition_bias
        weights = weights / weights.sum()
        
        regions = []
        for _ in range(num_regions):
            # Sample center frame
            center = torch.multinomial(weights, 1).item()
            
            # Determine region boundaries
            half_width = region_min_frames // 2
            start_frame = max(0, center - half_width)
            end_frame = min(num_frames, center + half_width)
            
            start_sample = start_frame * samples_per_frame
            end_sample = min(end_frame * samples_per_frame, waveform_length)
            
            regions.append((
                start_sample,
                end_sample,
                is_transition[center].item(),
            ))
        
        return regions
    
    def apply_to_region(
        self,
        waveform: torch.Tensor,
        start: int,
        end: int,
        aug_type: AugmentationType,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply augmentation to a specific region of the waveform.
        
        Args:
            waveform: Full waveform [T] or [C, T]
            start: Start sample index
            end: End sample index
            aug_type: Type of augmentation to apply
            
        Returns:
            Tuple of (augmented waveform, normalized parameter)
        """
        # Extract region
        is_1d = waveform.dim() == 1
        if is_1d:
            region = waveform[start:end]
        else:
            region = waveform[:, start:end]
        
        # Apply augmentation
        aug_fn_map = {
            AugmentationType.TIME_STRETCH: self.augmentor.time_stretch,
            AugmentationType.PITCH_SHIFT: self.augmentor.pitch_shift,
            AugmentationType.FORMANT_SHIFT: self.augmentor.formant_shift,
            AugmentationType.AMPLITUDE_MOD: self.augmentor.amplitude_modulation,
            AugmentationType.ADDITIVE_NOISE: self.augmentor.add_noise,
            AugmentationType.GAIN: self.augmentor.apply_gain,
            AugmentationType.LOW_PASS: self.augmentor.apply_low_pass,
            AugmentationType.HIGH_PASS: self.augmentor.apply_high_pass,
        }
        
        aug_fn = aug_fn_map[aug_type]
        augmented_region, param = aug_fn(region)
        
        # Handle length changes (time stretch)
        original_len = end - start
        if augmented_region.shape[-1] != original_len:
            if augmented_region.shape[-1] > original_len:
                augmented_region = augmented_region[..., :original_len]
            else:
                pad_len = original_len - augmented_region.shape[-1]
                augmented_region = F.pad(augmented_region, (0, pad_len))
        
        # Replace region in waveform
        result = waveform.clone()
        if is_1d:
            result[start:end] = augmented_region
        else:
            result[:, start:end] = augmented_region
        
        return result, param
    
    def forward(
        self,
        waveform: torch.Tensor,
        gamma: torch.Tensor,
        num_regions: int = 2,
        num_augs_per_region: int = 1,
    ) -> AugmentationResult:
        """
        Apply random augmentations to random regions.
        
        Args:
            waveform: Input waveform [T] or [C, T]
            gamma: CTC gamma matrix [T_frames, num_phonemes]
            num_regions: Number of regions to augment
            num_augs_per_region: Number of augmentation types per region
            
        Returns:
            AugmentationResult with augmented waveform and labels
        """
        waveform_length = waveform.shape[-1]
        regions = self.sample_regions(waveform_length, gamma, num_regions)

        result = waveform.clone()
        labels = {aug.value: 0.0 for aug in self.aug_types}
        region_labels = []

        for start, end, is_trans in regions:
            # Sample augmentation types for this region
            selected_augs = np.random.choice(
                self.aug_types,
                size=min(num_augs_per_region, len(self.aug_types)),
                replace=False,
            )

            for aug_type in selected_augs:
                result, param = self.apply_to_region(result, start, end, aug_type)
                region_labels.append(RegionAugmentationLabel(
                    start_sample=start,
                    end_sample=end,
                    aug_type=aug_type,
                    param_normalized=param,
                    is_transition=is_trans,
                ))
                # Aggregated view: keep max absolute magnitude per type
                if abs(param) > abs(labels[aug_type.value]):
                    labels[aug_type.value] = param

        return AugmentationResult(
            waveform=result,
            labels=labels,
            region_labels=region_labels,
            regions=[(s, e) for s, e, _ in regions],
        )


class SafeAugmentor(nn.Module):
    """
    Applies only safe augmentations for contrastive learning.
    
    These augmentations affect recording conditions but should not
    change clinical speech characteristics.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        config: Optional[AugmentationConfig] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.augmentor = AudioAugmentor(sample_rate, config)
    
    def forward(
        self,
        waveform: torch.Tensor,
        aug_types: Optional[List[AugmentationType]] = None,
    ) -> torch.Tensor:
        """
        Apply safe augmentations to create a positive pair.
        
        Args:
            waveform: Input waveform
            aug_types: Specific augmentations to apply. If None, randomly samples.
            
        Returns:
            Augmented waveform
        """
        if aug_types is None:
            # Randomly select 1-3 safe augmentations
            num_augs = np.random.randint(1, 4)
            aug_types = np.random.choice(
                list(SAFE_AUGMENTATIONS),
                size=min(num_augs, len(SAFE_AUGMENTATIONS)),
                replace=False,
            )
        
        result = waveform.clone()
        
        for aug_type in aug_types:
            if aug_type == AugmentationType.ADDITIVE_NOISE:
                result, _ = self.augmentor.add_noise(result)
            elif aug_type == AugmentationType.GAIN:
                result, _ = self.augmentor.apply_gain(result)
            elif aug_type == AugmentationType.LOW_PASS:
                result, _ = self.augmentor.apply_low_pass(result)
            elif aug_type == AugmentationType.HIGH_PASS:
                result, _ = self.augmentor.apply_high_pass(result)
        
        return result
