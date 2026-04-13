"""
Tests for data loading and augmentation components.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path

from clinical_speech_ssl.data.augmentations import (
    AugmentationType,
    AugmentationConfig,
    AugmentationResult,
    RegionAugmentationLabel,
    AudioAugmentor,
    RegionAugmentor,
    SafeAugmentor,
    CLINICAL_AUGMENTATIONS,
    CLINICAL_AUGMENTATION_INDEX,
    SAFE_AUGMENTATIONS,
)
from clinical_speech_ssl.data.dataset import (
    SpeechSample,
    ClinicalSpeechDataset,
    SSLCollator,
)


class TestAugmentationTypes:
    """Test augmentation type definitions."""
    
    def test_clinical_augmentations(self):
        """Test clinical augmentation set."""
        assert AugmentationType.TIME_STRETCH in CLINICAL_AUGMENTATIONS
        assert AugmentationType.PITCH_SHIFT in CLINICAL_AUGMENTATIONS
        assert AugmentationType.FORMANT_SHIFT in CLINICAL_AUGMENTATIONS
        assert AugmentationType.AMPLITUDE_MOD in CLINICAL_AUGMENTATIONS
    
    def test_safe_augmentations(self):
        """Test safe augmentation set."""
        assert AugmentationType.ADDITIVE_NOISE in SAFE_AUGMENTATIONS
        assert AugmentationType.GAIN in SAFE_AUGMENTATIONS
        assert AugmentationType.LOW_PASS in SAFE_AUGMENTATIONS
        assert AugmentationType.HIGH_PASS in SAFE_AUGMENTATIONS
    
    def test_no_overlap(self):
        """Test that clinical and safe augmentations don't overlap."""
        assert len(set(CLINICAL_AUGMENTATIONS) & set(SAFE_AUGMENTATIONS)) == 0

    def test_clinical_augmentations_are_ordered(self):
        """Test that CLINICAL_AUGMENTATIONS is an ordered tuple (not a set)."""
        assert isinstance(CLINICAL_AUGMENTATIONS, tuple)
        # Verify deterministic ordering
        assert list(CLINICAL_AUGMENTATIONS) == list(CLINICAL_AUGMENTATIONS)
        assert CLINICAL_AUGMENTATION_INDEX[AugmentationType.TIME_STRETCH] == 0

    def test_repetition_in_clinical_augmentations(self):
        """Test that REPETITION is a clinical augmentation (Phase 3b)."""
        assert AugmentationType.REPETITION in CLINICAL_AUGMENTATIONS
        assert AugmentationType.REPETITION in CLINICAL_AUGMENTATION_INDEX

    def test_safe_augmentations_are_ordered(self):
        """Test that SAFE_AUGMENTATIONS is an ordered tuple without REVERB."""
        assert isinstance(SAFE_AUGMENTATIONS, tuple)
        assert AugmentationType.REVERB not in SAFE_AUGMENTATIONS


class TestAugmentationConfig:
    """Test augmentation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AugmentationConfig()
        
        assert config.time_stretch_range == (0.8, 1.2)
        assert config.pitch_shift_range == (-3.0, 3.0)
        assert config.noise_snr_range == (10.0, 30.0)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AugmentationConfig(
            time_stretch_range=(0.9, 1.1),
            pitch_shift_range=(-1.0, 1.0),
        )
        
        assert config.time_stretch_range == (0.9, 1.1)
        assert config.pitch_shift_range == (-1.0, 1.0)


class TestAudioAugmentor:
    """Test audio augmentation operations."""
    
    @pytest.fixture
    def augmentor(self):
        return AudioAugmentor(sample_rate=16000)
    
    @pytest.fixture
    def sample_audio_1d(self):
        """1D audio tensor."""
        return torch.randn(16000)
    
    @pytest.fixture
    def sample_audio_2d(self):
        """2D audio tensor [C, T]."""
        return torch.randn(1, 16000)
    
    def test_time_stretch_slower(self, augmentor, sample_audio_1d):
        """Test time stretching (slower)."""
        stretched, param = augmentor.time_stretch(sample_audio_1d, factor=1.2)
        
        # Stretched audio should be shorter (fewer samples for same duration)
        assert stretched.shape[0] < sample_audio_1d.shape[0]
    
    def test_time_stretch_faster(self, augmentor, sample_audio_1d):
        """Test time stretching (faster)."""
        stretched, param = augmentor.time_stretch(sample_audio_1d, factor=0.8)
        
        # Compressed audio should be longer
        assert stretched.shape[0] > sample_audio_1d.shape[0]
    
    def test_time_stretch_random(self, augmentor, sample_audio_1d):
        """Test random time stretching."""
        stretched, param = augmentor.time_stretch(sample_audio_1d)
        
        assert -1 <= param <= 1  # Normalized parameter
    
    def test_pitch_shift(self, augmentor, sample_audio_1d):
        """Test pitch shifting."""
        shifted, param = augmentor.pitch_shift(sample_audio_1d, semitones=2.0)
        
        assert shifted.shape == sample_audio_1d.shape
        assert -1 <= param <= 1
    
    def test_pitch_shift_2d(self, augmentor, sample_audio_2d):
        """Test pitch shifting with 2D input."""
        shifted, param = augmentor.pitch_shift(sample_audio_2d, semitones=-1.0)
        
        assert shifted.shape == sample_audio_2d.shape
    
    def test_amplitude_modulation(self, augmentor, sample_audio_1d):
        """Test amplitude modulation."""
        modulated, param = augmentor.amplitude_modulation(
            sample_audio_1d,
            depth=0.5,
            freq=5.0,
        )
        
        assert modulated.shape == sample_audio_1d.shape
        # Modulation should change the signal
        assert not torch.allclose(modulated, sample_audio_1d)
    
    def test_add_noise(self, augmentor, sample_audio_1d):
        """Test noise addition."""
        noisy, param = augmentor.add_noise(sample_audio_1d, snr_db=20.0)
        
        assert noisy.shape == sample_audio_1d.shape
        # Signal should be different
        assert not torch.allclose(noisy, sample_audio_1d)
    
    def test_apply_gain(self, augmentor, sample_audio_1d):
        """Test gain application."""
        # Positive gain
        gained, _ = augmentor.apply_gain(sample_audio_1d, gain_db=6.0)
        assert gained.abs().mean() > sample_audio_1d.abs().mean()
        
        # Negative gain
        gained, _ = augmentor.apply_gain(sample_audio_1d, gain_db=-6.0)
        assert gained.abs().mean() < sample_audio_1d.abs().mean()
    
    def test_low_pass_filter(self, augmentor, sample_audio_1d):
        """Test low-pass filter."""
        filtered, param = augmentor.apply_low_pass(
            sample_audio_1d,
            cutoff_hz=4000.0,
        )
        
        assert filtered.shape == sample_audio_1d.shape
    
    def test_high_pass_filter(self, augmentor, sample_audio_1d):
        """Test high-pass filter."""
        filtered, param = augmentor.apply_high_pass(
            sample_audio_1d,
            cutoff_hz=100.0,
        )

        assert filtered.shape == sample_audio_1d.shape

    def test_repeat_region_preserves_length(self, augmentor, sample_audio_1d):
        """Test that repetition preserves region length."""
        repeated, param = augmentor.repeat_region(sample_audio_1d, count=2)
        assert repeated.shape == sample_audio_1d.shape
        assert -1 <= param <= 1

    def test_repeat_region_changes_content(self, augmentor):
        """Test that repetition produces content different from the input."""
        t = torch.linspace(0, 1, 16000)
        signal = torch.sin(2 * np.pi * 440 * t) * torch.linspace(0, 1, 16000)
        repeated, _ = augmentor.repeat_region(signal, count=2)
        assert not torch.allclose(repeated, signal)

    def test_repeat_region_has_repeated_structure(self, augmentor):
        """A repeated signal shows strong periodic self-similarity at the repeat lag."""
        torch.manual_seed(42)
        signal = torch.randn(3200)
        count = 3  # 4 total copies, each 800 samples pre-crossfade
        repeated, _ = augmentor.repeat_region(signal, count=count)
        assert repeated.shape == signal.shape

        # The signal repeats with period = piece_len - crossfade = 800 - 160 = 640
        # Scan lags to find the best match (test is robust to crossfade config)
        piece_len = 3200 // (count + 1)
        best_corr = -1.0
        for lag in range(piece_len // 2, piece_len + 50):
            a = repeated[:1600 - lag]
            b = repeated[lag:1600]
            if len(a) < 100:
                continue
            c = torch.sum(a * b) / (torch.norm(a) * torch.norm(b) + 1e-8)
            best_corr = max(best_corr, c.item())
        # After repetition there's a strong lag with high correlation (near 1)
        assert best_corr > 0.8, f"Expected strong repeat structure, got {best_corr}"
    
    def test_normalize_param(self, augmentor):
        """Test parameter normalization."""
        # Test normalization to [-1, 1]
        normalized = augmentor._normalize_param(
            1.0,  # Middle of range
            (0.8, 1.2),
        )
        assert abs(normalized) < 0.01  # Should be near 0
        
        normalized = augmentor._normalize_param(
            1.2,  # Max of range
            (0.8, 1.2),
        )
        assert abs(normalized - 1.0) < 0.01  # Should be near 1
        
        normalized = augmentor._normalize_param(
            0.8,  # Min of range
            (0.8, 1.2),
        )
        assert abs(normalized + 1.0) < 0.01  # Should be near -1


class TestRegionAugmentor:
    """Test region-based augmentation."""
    
    @pytest.fixture
    def region_augmentor(self):
        return RegionAugmentor(
            sample_rate=16000,
            transition_bias=2.0,
            clinical_only=True,
        )
    
    @pytest.fixture
    def sample_waveform(self):
        return torch.randn(16000)
    
    @pytest.fixture
    def sample_gamma(self):
        """Sample CTC gamma matrix."""
        # 50 frames, 10 phonemes
        gamma = torch.rand(50, 10)
        return torch.softmax(gamma, dim=-1)
    
    def test_identify_transitions_shape(self, region_augmentor, sample_gamma):
        """Test transition identification output shape."""
        transitions = region_augmentor.identify_transitions(sample_gamma)
        
        assert transitions.shape == (50,)
        assert transitions.dtype == torch.bool
    
    def test_identify_transitions_content(self, region_augmentor):
        """Test transition identification logic."""
        # Create gamma with clear transitions
        # Frame 0-9: phoneme 0 dominant
        # Frame 10-19: transitioning
        # Frame 20-29: phoneme 1 dominant
        gamma = torch.zeros(30, 5)
        gamma[0:10, 0] = 1.0
        gamma[20:30, 1] = 1.0
        # Transition region - mixed
        gamma[10:20, 0] = 0.5
        gamma[10:20, 1] = 0.5
        
        transitions = region_augmentor.identify_transitions(gamma, threshold=0.3)
        
        # Transition region should have high entropy
        assert transitions[10:20].any()
    
    def test_sample_regions(self, region_augmentor, sample_waveform, sample_gamma):
        """Test region sampling."""
        regions = region_augmentor.sample_regions(
            sample_waveform.shape[0],
            sample_gamma,
            num_regions=3,
        )
        
        assert len(regions) == 3
        
        for start, end, is_transition in regions:
            assert 0 <= start < end <= sample_waveform.shape[0]
            assert isinstance(is_transition, bool)
    
    def test_apply_to_region(self, region_augmentor, sample_waveform):
        """Test applying augmentation to a region."""
        start, end = 1000, 3000
        
        augmented, param = region_augmentor.apply_to_region(
            sample_waveform,
            start,
            end,
            AugmentationType.PITCH_SHIFT,
        )
        
        # Shape should be preserved
        assert augmented.shape == sample_waveform.shape
        
        # Region should be different
        assert not torch.allclose(
            augmented[start:end],
            sample_waveform[start:end],
        )
        
        # Outside region should be same
        assert torch.allclose(augmented[:start], sample_waveform[:start])
        assert torch.allclose(augmented[end:], sample_waveform[end:])
    
    def test_forward(self, region_augmentor, sample_waveform, sample_gamma):
        """Test full forward pass."""
        result = region_augmentor(
            sample_waveform,
            sample_gamma,
            num_regions=2,
            num_augs_per_region=1,
        )

        assert result.waveform.shape == sample_waveform.shape
        assert len(result.labels) == len(CLINICAL_AUGMENTATIONS)
        assert result.regions is not None
        assert len(result.regions) == 2

    def test_region_labels_are_populated(self, region_augmentor, sample_waveform, sample_gamma):
        """Test that per-region labels are stored properly (Fix 2)."""
        result = region_augmentor(
            sample_waveform,
            sample_gamma,
            num_regions=3,
            num_augs_per_region=1,
        )

        assert len(result.region_labels) >= 3
        for rl in result.region_labels:
            assert isinstance(rl, RegionAugmentationLabel)
            assert rl.start_sample >= 0
            assert rl.end_sample > rl.start_sample
            assert rl.aug_type in CLINICAL_AUGMENTATIONS
            assert isinstance(rl.is_transition, bool)

    def test_region_labels_preserve_duplicates(self, region_augmentor, sample_waveform, sample_gamma):
        """Test that if two regions get the same aug type, both are preserved."""
        np.random.seed(42)
        # Request many regions to increase chance of same-type collision
        result = region_augmentor(
            sample_waveform,
            sample_gamma,
            num_regions=8,
            num_augs_per_region=1,
        )

        # With 8 regions picking from 5 types (after REPETITION was added),
        # collisions are still likely
        from collections import Counter
        type_counts = Counter(rl.aug_type for rl in result.region_labels)
        assert any(c > 1 for c in type_counts.values()), (
            "Expected at least one augmentation type to appear in multiple regions"
        )

    def test_region_apply_repetition(self, region_augmentor, sample_waveform):
        """Test that REPETITION applied to a region preserves total length."""
        augmented, param = region_augmentor.apply_to_region(
            sample_waveform, 1000, 5000, AugmentationType.REPETITION,
        )
        assert augmented.shape == sample_waveform.shape
        assert not torch.allclose(augmented[1000:5000], sample_waveform[1000:5000])
        # Outside the region should be unchanged
        assert torch.allclose(augmented[:1000], sample_waveform[:1000])
        assert torch.allclose(augmented[5000:], sample_waveform[5000:])


class TestSafeAugmentor:
    """Test safe augmentation for contrastive learning."""
    
    @pytest.fixture
    def safe_augmentor(self):
        return SafeAugmentor(sample_rate=16000)
    
    def test_forward(self, safe_augmentor):
        """Test safe augmentation forward pass."""
        waveform = torch.randn(16000)
        augmented = safe_augmentor(waveform)
        
        assert augmented.shape == waveform.shape
        # Should be different (augmented)
        assert not torch.allclose(augmented, waveform)
    
    def test_specific_augmentations(self, safe_augmentor):
        """Test applying specific augmentations."""
        waveform = torch.randn(16000)
        
        augmented = safe_augmentor(
            waveform,
            aug_types=[AugmentationType.ADDITIVE_NOISE],
        )
        
        assert augmented.shape == waveform.shape


class TestSpeechSample:
    """Test SpeechSample dataclass."""
    
    def test_sample_creation(self):
        """Test creating a speech sample."""
        sample = SpeechSample(
            waveform=torch.randn(16000),
            sample_rate=16000,
            gamma=torch.randn(50, 10),
            patient_id="patient001",
        )
        
        assert sample.waveform.shape == (16000,)
        assert sample.sample_rate == 16000
        assert sample.patient_id == "patient001"
    
    def test_optional_fields(self):
        """Test optional fields."""
        sample = SpeechSample(
            waveform=torch.randn(16000),
            sample_rate=16000,
            gamma=torch.randn(50, 10),
        )
        
        assert sample.patient_id is None
        assert sample.clinical_labels is None


class TestSSLCollator:
    """Test SSL data collation."""
    
    @pytest.fixture
    def collator(self):
        return SSLCollator(pad_to_max=True)
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch of SpeechSamples."""
        return [
            SpeechSample(
                waveform=torch.randn(16000),
                sample_rate=16000,
                gamma=torch.randn(50, 10),
                patient_id="p1",
            ),
            SpeechSample(
                waveform=torch.randn(12000),
                sample_rate=16000,
                gamma=torch.randn(40, 10),
                patient_id="p2",
            ),
        ]
    
    def test_collate(self, collator, sample_batch):
        """Test batch collation."""
        batch = collator(sample_batch)
        
        assert 'waveforms' in batch
        assert 'lengths' in batch
        assert 'gammas' in batch
        
        # Should be padded to max length
        assert batch['waveforms'].shape == (2, 16000)
        assert batch['lengths'].tolist() == [16000, 12000]
    
    def test_collate_with_labels(self, collator):
        """Test collation with clinical labels."""
        samples = [
            SpeechSample(
                waveform=torch.randn(16000),
                sample_rate=16000,
                gamma=torch.randn(50, 10),
                clinical_labels={'severity': 0.5, 'diagnosis': 1},
            ),
            SpeechSample(
                waveform=torch.randn(16000),
                sample_rate=16000,
                gamma=torch.randn(50, 10),
                clinical_labels={'severity': 0.8, 'diagnosis': 2},
            ),
        ]
        
        batch = collator(samples)
        
        assert 'clinical_labels' in batch
        assert 'severity' in batch['clinical_labels']
        assert batch['clinical_labels']['severity'].shape == (2,)


class TestGammaValidation:
    """Test gamma shape validation (Fix 6) — unit tests on _validate_gamma directly."""

    def _make_dataset(self):
        """Build a minimal dataset to access _validate_gamma."""
        # Create with an empty manifest (no file I/O needed for unit tests)
        import types
        ds = ClinicalSpeechDataset.__new__(ClinicalSpeechDataset)
        ds.sample_rate = 16000
        ds.gamma_mismatch_policy = "truncate"
        ds.SAMPLES_PER_FRAME = 320
        return ds

    def test_gamma_validation_pass(self):
        """Correctly-sized gamma passes unchanged."""
        ds = self._make_dataset()
        ds.gamma_mismatch_policy = "raise"
        waveform = torch.randn(16000)  # 16000 / 320 = 50 frames
        gamma = torch.softmax(torch.randn(50, 10), dim=-1)
        result = ds._validate_gamma(gamma, waveform)
        assert result.shape[0] == 50

    def test_gamma_validation_small_slack(self):
        """Gamma within ±2 frame tolerance is silently fixed."""
        ds = self._make_dataset()
        ds.gamma_mismatch_policy = "raise"
        waveform = torch.randn(16000)  # 50 frames expected
        gamma = torch.softmax(torch.randn(52, 10), dim=-1)  # 2 extra — within tolerance
        result = ds._validate_gamma(gamma, waveform)
        assert result.shape[0] == 50

    def test_gamma_validation_truncate(self):
        """Large overshoot is truncated under 'truncate' policy."""
        ds = self._make_dataset()
        ds.gamma_mismatch_policy = "truncate"
        waveform = torch.randn(16000)  # 50 frames expected
        gamma = torch.softmax(torch.randn(80, 10), dim=-1)
        result = ds._validate_gamma(gamma, waveform)
        assert result.shape[0] == 50

    def test_gamma_validation_pad(self):
        """Large undershoot is zero-padded under 'truncate' policy."""
        ds = self._make_dataset()
        ds.gamma_mismatch_policy = "truncate"
        waveform = torch.randn(16000)  # 50 frames expected
        gamma = torch.softmax(torch.randn(30, 10), dim=-1)
        result = ds._validate_gamma(gamma, waveform)
        assert result.shape[0] == 50

    def test_gamma_validation_raise(self):
        """Large mismatch raises ValueError under 'raise' policy."""
        ds = self._make_dataset()
        ds.gamma_mismatch_policy = "raise"
        waveform = torch.randn(16000)
        gamma = torch.softmax(torch.randn(100, 10), dim=-1)
        with pytest.raises(ValueError, match="Gamma frame count mismatch"):
            ds._validate_gamma(gamma, waveform)


class TestDatasetIntegration:
    """Integration tests for dataset functionality."""
    
    def test_manifest_loading(self, tmp_path):
        """Test loading from manifest file."""
        import scipy.io.wavfile

        # Create dummy manifest
        manifest = {
            'metadata': {'version': '1.0'},
            'samples': [
                {
                    'audio_path': str(tmp_path / 'audio1.wav'),
                    'gamma_path': str(tmp_path / 'gamma1.pt'),
                    'patient_id': 'p1',
                },
            ]
        }

        # Create dummy files
        waveform = torch.randn(1, 16000)
        scipy.io.wavfile.write(str(tmp_path / 'audio1.wav'), 16000, waveform.squeeze(0).numpy())
        torch.save(torch.softmax(torch.randn(50, 10), dim=-1), str(tmp_path / 'gamma1.pt'))
        
        manifest_path = tmp_path / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        # Load dataset
        dataset = ClinicalSpeechDataset(manifest_path=manifest_path)
        
        assert len(dataset) == 1
        
        sample = dataset[0]
        assert sample.waveform is not None
        assert sample.gamma is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
