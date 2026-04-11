"""
Tests for frontend modules.

Tests waveform CNN and spectrogram patch frontends.
"""

import pytest
import torch
import numpy as np

from clinical_speech_ssl.models.frontends import (
    WaveformCNNFrontend,
    WaveformCNNFrontendSmall,
    WaveformCNNFrontendLarge,
    SpectrogramPatchFrontend,
    TallNarrowPatcher,
    ViTStylePatcher,
    MelSpectrogramTransform,
)


class TestWaveformCNNFrontend:
    """Tests for WaveformCNNFrontend."""
    
    @pytest.fixture
    def frontend_small(self):
        return WaveformCNNFrontendSmall(output_dim=256)
    
    @pytest.fixture
    def frontend_base(self):
        return WaveformCNNFrontend(output_dim=256)
    
    @pytest.fixture
    def sample_waveform(self):
        """Create sample waveform: batch of 4, 2 seconds at 16kHz."""
        return torch.randn(4, 32000)
    
    def test_forward_shape(self, frontend_small, sample_waveform):
        """Test output shape is correct."""
        features, lengths = frontend_small(sample_waveform)
        
        assert features.dim() == 3
        assert features.shape[0] == 4  # Batch size
        assert features.shape[2] == 256  # Output dim
        assert features.shape[1] > 0  # Some sequence length
    
    def test_forward_with_lengths(self, frontend_small, sample_waveform):
        """Test with explicit lengths."""
        lengths = torch.tensor([32000, 24000, 16000, 8000])
        features, out_lengths = frontend_small(sample_waveform, lengths)
        
        assert out_lengths is not None
        assert len(out_lengths) == 4
        # Longer input should give longer output
        assert out_lengths[0] > out_lengths[3]
    
    def test_output_length_computation(self, frontend_small):
        """Test get_output_length method."""
        input_length = 16000
        output_length = frontend_small.get_output_length(input_length)
        
        # Actually run forward to verify
        waveform = torch.randn(1, input_length)
        features, _ = frontend_small(waveform)
        
        # Should be close (may differ slightly due to padding)
        assert abs(features.shape[1] - output_length) <= 2
    
    def test_mono_input(self, frontend_small):
        """Test with 1D input."""
        waveform = torch.randn(16000)
        features, _ = frontend_small(waveform.unsqueeze(0))
        
        assert features.dim() == 3
        assert features.shape[0] == 1
    
    def test_gradient_flow(self, frontend_small, sample_waveform):
        """Test gradients flow through frontend."""
        features, _ = frontend_small(sample_waveform)
        loss = features.sum()
        loss.backward()
        
        # Check gradients exist
        for param in frontend_small.parameters():
            assert param.grad is not None
    
    def test_different_sizes(self):
        """Test different frontend sizes have different parameter counts."""
        small = WaveformCNNFrontendSmall(output_dim=256)
        large = WaveformCNNFrontendLarge(output_dim=256)
        
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        
        assert large_params > small_params


class TestSpectrogramPatchFrontend:
    """Tests for spectrogram-based frontends."""
    
    @pytest.fixture
    def mel_transform(self):
        return MelSpectrogramTransform(sample_rate=16000, n_mels=80)
    
    @pytest.fixture
    def tall_narrow_frontend(self):
        return SpectrogramPatchFrontend(
            patch_type="tall_narrow",
            n_mels=80,
            patch_frames=4,
            stride_frames=2,
            embed_dim=256,
        )
    
    @pytest.fixture
    def vit_frontend(self):
        return SpectrogramPatchFrontend(
            patch_type="vit",
            n_mels=80,
            vit_patch_size=(16, 16),
            embed_dim=256,
        )
    
    @pytest.fixture
    def sample_waveform(self):
        return torch.randn(4, 32000)
    
    def test_mel_transform(self, mel_transform, sample_waveform):
        """Test mel spectrogram computation."""
        mel = mel_transform(sample_waveform)
        
        assert mel.dim() == 3
        assert mel.shape[0] == 4  # Batch
        assert mel.shape[1] == 80  # n_mels
        assert mel.shape[2] > 0  # Time frames
    
    def test_tall_narrow_output_shape(self, tall_narrow_frontend, sample_waveform):
        """Test tall-narrow patcher output."""
        patches, lengths = tall_narrow_frontend(sample_waveform)
        
        assert patches.dim() == 3
        assert patches.shape[0] == 4
        assert patches.shape[2] == 256  # embed_dim
    
    def test_vit_output_shape(self, vit_frontend, sample_waveform):
        """Test ViT-style patcher output."""
        patches, lengths = vit_frontend(sample_waveform)
        
        assert patches.dim() == 3
        assert patches.shape[0] == 4
        assert patches.shape[2] == 256
    
    def test_with_lengths(self, tall_narrow_frontend, sample_waveform):
        """Test length handling."""
        lengths = torch.tensor([32000, 24000, 16000, 8000])
        patches, out_lengths = tall_narrow_frontend(sample_waveform, lengths)
        
        assert out_lengths is not None
        assert len(out_lengths) == 4
    
    def test_gradient_flow(self, tall_narrow_frontend, sample_waveform):
        """Test gradients flow through patcher."""
        patches, _ = tall_narrow_frontend(sample_waveform)
        loss = patches.sum()
        loss.backward()
        
        has_grads = any(p.grad is not None for p in tall_narrow_frontend.parameters())
        assert has_grads


class TestTallNarrowPatcher:
    """Tests specifically for TallNarrowPatcher."""
    
    @pytest.fixture
    def patcher(self):
        return TallNarrowPatcher(n_mels=80, patch_frames=4, stride_frames=2, embed_dim=256)
    
    @pytest.fixture
    def sample_mel(self):
        """Sample mel spectrogram: batch of 4, 80 mels, 100 frames."""
        return torch.randn(4, 80, 100)
    
    def test_patch_count(self, patcher, sample_mel):
        """Test number of patches is correct."""
        patches, _ = patcher(sample_mel)
        
        expected_patches = (100 - 4) // 2 + 1  # (T - patch_size) // stride + 1
        assert patches.shape[1] == expected_patches
    
    def test_full_frequency_coverage(self, patcher, sample_mel):
        """Verify patches span full frequency range."""
        # Each patch should have n_mels * patch_frames elements before projection
        # We can verify through the projection input dimension
        expected_input = 80 * 4  # n_mels * patch_frames
        assert patcher.projection[0].in_features == expected_input


class TestViTStylePatcher:
    """Tests specifically for ViTStylePatcher."""
    
    @pytest.fixture
    def patcher(self):
        return ViTStylePatcher(n_mels=80, patch_size=(16, 16), embed_dim=256)
    
    @pytest.fixture
    def sample_mel(self):
        return torch.randn(4, 80, 100)
    
    def test_2d_patching(self, patcher, sample_mel):
        """Test 2D patch extraction."""
        patches, _ = patcher(sample_mel)
        
        # Should have grid of patches
        assert patches.dim() == 3
        assert patches.shape[0] == 4
        assert patches.shape[2] == 256
    
    def test_position_embeddings(self, patcher, sample_mel):
        """Test position embeddings are added."""
        # Just verify it runs without error
        patches1, _ = patcher(sample_mel)
        patches2, _ = patcher(sample_mel)
        
        # With same input, output should be deterministic
        assert torch.allclose(patches1, patches2)
