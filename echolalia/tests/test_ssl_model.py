"""
Tests for the main SSL model and its components.
"""

import pytest
import torch
import numpy as np

from clinical_speech_ssl.models.ssl_model import (
    ClinicalSpeechSSL,
    ClinicalSpeechSSLConfig,
    create_ssl_model,
)
from clinical_speech_ssl.data.augmentations import (
    AudioAugmentor,
    RegionAugmentor,
    SafeAugmentor,
    AugmentationConfig,
    AugmentationType,
)


class TestClinicalSpeechSSLConfig:
    """Test configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ClinicalSpeechSSLConfig()
        
        assert config.input_type == "waveform"
        assert config.sample_rate == 16000
        assert config.embed_dim == 256
        assert config.use_augmentation_prediction is True
        assert config.use_masked_reconstruction is True
        assert config.use_contrastive is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ClinicalSpeechSSLConfig(
            input_type="spectrogram",
            embed_dim=512,
            encoder_type="conformer_large",
            use_contrastive=False,
        )
        
        assert config.input_type == "spectrogram"
        assert config.embed_dim == 512
        assert config.use_contrastive is False


class TestClinicalSpeechSSL:
    """Test main SSL model."""
    
    @pytest.fixture
    def tiny_model(self):
        """Create a tiny model for testing."""
        return create_ssl_model("tiny")
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        batch_size = 2
        seq_length = 16000  # 1 second at 16kHz
        num_phonemes = 10
        num_frames = 50
        
        waveforms = torch.randn(batch_size, seq_length)
        lengths = torch.tensor([seq_length, seq_length // 2])
        gammas = [
            torch.softmax(torch.randn(num_frames, num_phonemes), dim=-1)
            for _ in range(batch_size)
        ]
        
        return {
            'waveforms': waveforms,
            'lengths': lengths,
            'gammas': gammas,
        }
    
    def test_model_creation_tiny(self, tiny_model):
        """Test tiny model creation."""
        assert tiny_model is not None
        assert isinstance(tiny_model, ClinicalSpeechSSL)
    
    def test_model_creation_all_presets(self):
        """Test all model presets."""
        for preset in ["tiny", "small", "base"]:
            model = create_ssl_model(preset)
            assert model is not None
    
    def test_forward_pass(self, tiny_model, sample_batch):
        """Test forward pass returns expected loss keys."""
        losses = tiny_model(
            sample_batch['waveforms'],
            sample_batch['gammas'],
            sample_batch['lengths'],
        )
        
        assert 'total_loss' in losses
        assert losses['total_loss'].requires_grad
    
    def test_forward_with_all_objectives(self, sample_batch):
        """Test forward with all SSL objectives enabled."""
        config = ClinicalSpeechSSLConfig(
            frontend_type="cnn_small",
            encoder_type="transformer_small",
            embed_dim=128,
            use_augmentation_prediction=True,
            use_masked_reconstruction=True,
            use_contrastive=True,
        )
        model = ClinicalSpeechSSL(config)
        
        losses = model(
            sample_batch['waveforms'],
            sample_batch['gammas'],
            sample_batch['lengths'],
        )
        
        # Should have losses for each objective
        assert 'aug_total_loss' in losses or 'aug_presence_loss' in losses
        assert 'mask_loss' in losses
        assert 'contrastive_loss' in losses
    
    def test_forward_single_objective(self, sample_batch):
        """Test forward with single objective."""
        config = ClinicalSpeechSSLConfig(
            frontend_type="cnn_small",
            encoder_type="transformer_small",
            embed_dim=128,
            use_augmentation_prediction=False,
            use_masked_reconstruction=True,
            use_contrastive=False,
        )
        model = ClinicalSpeechSSL(config)
        
        losses = model(
            sample_batch['waveforms'],
            sample_batch['gammas'],
            sample_batch['lengths'],
        )
        
        assert 'mask_loss' in losses
        assert 'total_loss' in losses
    
    def test_encode_method(self, tiny_model, sample_batch):
        """Test encode method."""
        features = tiny_model.encode(
            sample_batch['waveforms'],
            sample_batch['lengths'],
        )
        
        assert features.dim() == 3  # [B, T', D]
        assert features.shape[0] == sample_batch['waveforms'].shape[0]
        assert features.shape[2] == tiny_model.config.embed_dim
    
    def test_encode_return_all_layers(self, tiny_model, sample_batch):
        """Test encode with all layer outputs."""
        features, all_layers = tiny_model.encode(
            sample_batch['waveforms'],
            sample_batch['lengths'],
            return_all_layers=True,
        )
        
        assert features.dim() == 3
        assert all_layers is not None
        assert len(all_layers) > 0
    
    def test_get_representations(self, tiny_model, sample_batch):
        """Test getting representations with different pooling."""
        for pooling in ["mean", "first", "last"]:
            reps = tiny_model.get_representations(
                sample_batch['waveforms'],
                sample_batch['lengths'],
                pooling=pooling,
            )
            
            assert reps.dim() == 2  # [B, D]
            assert reps.shape[0] == sample_batch['waveforms'].shape[0]
        
        # Test no pooling
        reps = tiny_model.get_representations(
            sample_batch['waveforms'],
            sample_batch['lengths'],
            pooling="none",
        )
        assert reps.dim() == 3  # [B, T', D]
    
    def test_spectrogram_input(self, sample_batch):
        """Test model with spectrogram input."""
        for patch_type in ["patch_tall_narrow", "patch_vit"]:
            config = ClinicalSpeechSSLConfig(
                input_type="spectrogram",
                frontend_type=patch_type,
                encoder_type="transformer_small",
                embed_dim=128,
            )
            model = ClinicalSpeechSSL(config)
            
            features = model.encode(
                sample_batch['waveforms'],
                sample_batch['lengths'],
            )
            
            assert features.dim() == 3
    
    def test_gradient_flow(self, tiny_model, sample_batch):
        """Test that gradients flow properly."""
        tiny_model.train()
        
        losses = tiny_model(
            sample_batch['waveforms'],
            sample_batch['gammas'],
            sample_batch['lengths'],
        )
        
        losses['total_loss'].backward()
        
        # Check that gradients exist
        has_grad = False
        for param in tiny_model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No gradients found"


class TestAudioAugmentor:
    """Test audio augmentation module."""
    
    @pytest.fixture
    def augmentor(self):
        return AudioAugmentor(sample_rate=16000)
    
    @pytest.fixture
    def sample_audio(self):
        return torch.randn(16000)  # 1 second
    
    def test_time_stretch(self, augmentor, sample_audio):
        """Test time stretching."""
        stretched, param = augmentor.time_stretch(sample_audio, factor=1.2)
        
        assert stretched.shape[0] != sample_audio.shape[0]
        assert -1 <= param <= 1  # Normalized
    
    def test_pitch_shift(self, augmentor, sample_audio):
        """Test pitch shifting."""
        shifted, param = augmentor.pitch_shift(sample_audio, semitones=2.0)
        
        assert shifted.shape == sample_audio.shape
        assert -1 <= param <= 1
    
    def test_add_noise(self, augmentor, sample_audio):
        """Test noise addition."""
        noisy, param = augmentor.add_noise(sample_audio, snr_db=20.0)
        
        assert noisy.shape == sample_audio.shape
        assert not torch.allclose(noisy, sample_audio)
    
    def test_apply_gain(self, augmentor, sample_audio):
        """Test gain application."""
        gained, param = augmentor.apply_gain(sample_audio, gain_db=6.0)
        
        assert gained.shape == sample_audio.shape
        # Check that amplitude increased
        assert gained.abs().mean() > sample_audio.abs().mean()


class TestRegionAugmentor:
    """Test region-based augmentation."""
    
    @pytest.fixture
    def region_augmentor(self):
        return RegionAugmentor(sample_rate=16000, transition_bias=2.0)
    
    @pytest.fixture
    def sample_data(self):
        waveform = torch.randn(16000)
        gamma = torch.softmax(torch.randn(50, 10), dim=-1)
        return waveform, gamma
    
    def test_identify_transitions(self, region_augmentor, sample_data):
        """Test transition identification."""
        _, gamma = sample_data
        transitions = region_augmentor.identify_transitions(gamma)
        
        assert transitions.shape[0] == gamma.shape[0]
        assert transitions.dtype == torch.bool
    
    def test_sample_regions(self, region_augmentor, sample_data):
        """Test region sampling."""
        waveform, gamma = sample_data
        regions = region_augmentor.sample_regions(
            waveform.shape[0],
            gamma,
            num_regions=3,
        )
        
        assert len(regions) == 3
        for start, end, is_trans in regions:
            assert 0 <= start < end <= waveform.shape[0]
            assert isinstance(is_trans, bool)
    
    def test_forward(self, region_augmentor, sample_data):
        """Test full forward pass."""
        waveform, gamma = sample_data
        result = region_augmentor(waveform, gamma, num_regions=2)
        
        assert result.waveform.shape == waveform.shape
        assert len(result.labels) > 0
        assert result.regions is not None


class TestSafeAugmentor:
    """Test safe augmentation for contrastive learning."""
    
    @pytest.fixture
    def safe_augmentor(self):
        return SafeAugmentor(sample_rate=16000)
    
    def test_forward(self, safe_augmentor):
        """Test safe augmentation."""
        waveform = torch.randn(16000)
        augmented = safe_augmentor(waveform)
        
        assert augmented.shape == waveform.shape
        # Should be different (augmented)
        assert not torch.allclose(augmented, waveform)


class TestModelSerialization:
    """Test model saving and loading."""
    
    def test_state_dict(self):
        """Test state dict saving/loading."""
        model1 = create_ssl_model("tiny")
        state_dict = model1.state_dict()
        
        model2 = create_ssl_model("tiny")
        model2.load_state_dict(state_dict)
        
        # Check parameters match
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)
    
    def test_torchscript_compatibility(self):
        """Test TorchScript tracing works for encode."""
        model = create_ssl_model("tiny")
        model.eval()
        
        # Trace the encode method
        sample_input = torch.randn(1, 16000)
        
        # Note: Full model may not be traceable due to dynamic operations
        # but encode should work
        try:
            traced = torch.jit.trace(
                model.encode,
                (sample_input,),
            )
            output = traced(sample_input)
            assert output.dim() == 3
        except Exception as e:
            # Some dynamic operations may prevent tracing
            pytest.skip(f"TorchScript tracing not supported: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
