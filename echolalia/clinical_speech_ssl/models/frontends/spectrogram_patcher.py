"""
Spectrogram patch frontend for processing mel spectrograms.

Supports two patching strategies:
1. Tall-narrow patches: Full frequency range, few time steps
   - Respects frequency structure (not translation invariant)
   - Better for capturing formant patterns and spectral envelopes

2. ViT-style square patches: Standard grid of square patches
   - Treats spectrogram more like an image
   - May miss frequency-specific structure but more general
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from einops import rearrange, repeat


class MelSpectrogramTransform(nn.Module):
    """Compute mel spectrogram from waveform."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        normalized: bool = True,
        log_scale: bool = True,
        log_offset: float = 1e-6,
    ):
        super().__init__()
        self.log_scale = log_scale
        self.log_offset = log_offset
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate // 2,
            power=power,
            normalized=normalized,
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram.
        
        Args:
            waveform: [B, T] or [B, 1, T]
            
        Returns:
            mel_spec: [B, n_mels, T'] where T' = T // hop_length
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        
        mel = self.mel_spec(waveform)
        
        if self.log_scale:
            mel = torch.log(mel + self.log_offset)
        
        return mel


class TallNarrowPatcher(nn.Module):
    """
    Patches spectrogram using tall-narrow patches.
    
    Each patch spans the full frequency range but only a few time frames.
    This respects the fact that frequency bins have specific meaning
    (unlike spatial dimensions in images).
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        patch_frames: int = 4,
        stride_frames: int = 2,
        embed_dim: int = 768,
        dropout: float = 0.0,
    ):
        """
        Args:
            n_mels: Number of mel frequency bins
            patch_frames: Number of time frames per patch
            stride_frames: Stride between patches in time
            embed_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.patch_frames = patch_frames
        self.stride_frames = stride_frames
        self.embed_dim = embed_dim
        
        # Patch size is [n_mels, patch_frames]
        patch_size = n_mels * patch_frames
        
        # Linear projection of flattened patch
        self.projection = nn.Sequential(
            nn.Linear(patch_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        
        # Learnable position embeddings
        # Max sequence length - will be interpolated if needed
        self.max_patches = 512
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.max_patches, embed_dim) * 0.02
        )
    
    def get_num_patches(self, spec_length: int) -> int:
        """Compute number of patches given spectrogram time length."""
        return (spec_length - self.patch_frames) // self.stride_frames + 1
    
    def forward(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract tall-narrow patches from mel spectrogram.
        
        Args:
            mel_spec: [B, n_mels, T]
            lengths: Optional time lengths [B]
            
        Returns:
            patches: [B, num_patches, embed_dim]
            patch_lengths: Adjusted lengths [B]
        """
        B, F, T = mel_spec.shape
        assert F == self.n_mels, f"Expected {self.n_mels} mel bins, got {F}"
        
        # Extract patches using unfold
        # [B, n_mels, T] -> [B, n_mels, num_patches, patch_frames]
        patches = mel_spec.unfold(
            dimension=2,
            size=self.patch_frames,
            step=self.stride_frames,
        )
        
        num_patches = patches.shape[2]
        
        # Reshape to [B, num_patches, n_mels * patch_frames]
        patches = rearrange(patches, 'b f p t -> b p (f t)')
        
        # Project to embedding dimension
        patches = self.projection(patches)
        
        # Add position embeddings
        if num_patches <= self.max_patches:
            pos_emb = self.position_embeddings[:, :num_patches, :]
        else:
            # Interpolate position embeddings for longer sequences
            pos_emb = F.interpolate(
                self.position_embeddings.transpose(1, 2),
                size=num_patches,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
        
        patches = patches + pos_emb
        
        # Compute output lengths
        patch_lengths = None
        if lengths is not None:
            patch_lengths = torch.tensor([
                self.get_num_patches(l.item()) for l in lengths
            ], device=lengths.device)
        
        return patches, patch_lengths


class ViTStylePatcher(nn.Module):
    """
    Patches spectrogram using ViT-style square/rectangular patches.
    
    Treats spectrogram as a 2D image and extracts patches in a grid.
    More general but may miss frequency-specific structure.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        patch_size: Tuple[int, int] = (16, 16),
        stride: Optional[Tuple[int, int]] = None,
        embed_dim: int = 768,
        dropout: float = 0.0,
    ):
        """
        Args:
            n_mels: Number of mel frequency bins
            patch_size: (freq_bins, time_frames) per patch
            stride: (freq_stride, time_stride). If None, equals patch_size
            embed_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.embed_dim = embed_dim
        
        # Use Conv2d for efficient patch extraction + projection
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=self.stride,
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Position embeddings
        self.max_patches_freq = n_mels // self.stride[0] + 1
        self.max_patches_time = 256
        
        # 2D position embeddings (separable)
        self.freq_pos = nn.Parameter(
            torch.randn(1, self.max_patches_freq, embed_dim // 2) * 0.02
        )
        self.time_pos = nn.Parameter(
            torch.randn(1, self.max_patches_time, embed_dim // 2) * 0.02
        )
    
    def get_num_patches(self, spec_length: int) -> Tuple[int, int]:
        """Compute (freq_patches, time_patches) given spectrogram shape."""
        freq_patches = (self.n_mels - self.patch_size[0]) // self.stride[0] + 1
        time_patches = (spec_length - self.patch_size[1]) // self.stride[1] + 1
        return freq_patches, time_patches
    
    def forward(
        self,
        mel_spec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract ViT-style patches from mel spectrogram.
        
        Args:
            mel_spec: [B, n_mels, T]
            lengths: Optional time lengths [B]
            
        Returns:
            patches: [B, num_patches, embed_dim]
            patch_lengths: Adjusted lengths [B]
        """
        B, F, T = mel_spec.shape
        
        # Add channel dimension [B, 1, F, T]
        mel_spec = mel_spec.unsqueeze(1)
        
        # Extract and project patches
        # [B, embed_dim, F', T']
        patches = self.patch_embed(mel_spec)
        
        _, _, Fp, Tp = patches.shape
        
        # Build 2D position embeddings
        freq_emb = self.freq_pos[:, :Fp, :]  # [1, Fp, D/2]
        time_emb = self.time_pos[:, :Tp, :]  # [1, Tp, D/2]
        
        # Expand to grid
        freq_emb = repeat(freq_emb, '1 f d -> 1 f t d', t=Tp)
        time_emb = repeat(time_emb, '1 t d -> 1 f t d', f=Fp)
        pos_emb = torch.cat([freq_emb, time_emb], dim=-1)  # [1, Fp, Tp, D]
        
        # Reshape patches to [B, Fp, Tp, D]
        patches = rearrange(patches, 'b d f t -> b f t d')
        
        # Add position embeddings
        patches = patches + pos_emb
        
        # Flatten spatial dimensions
        patches = rearrange(patches, 'b f t d -> b (f t) d')
        
        patches = self.norm(patches)
        patches = self.dropout(patches)
        
        # Compute output lengths (number of time patches)
        patch_lengths = None
        if lengths is not None:
            patch_lengths = torch.tensor([
                Fp * self.get_num_patches(l.item())[1] for l in lengths
            ], device=lengths.device)
        
        return patches, patch_lengths


class SpectrogramPatchFrontend(nn.Module):
    """
    Complete frontend: mel spectrogram + patching.
    
    Combines mel spectrogram computation with patch extraction
    for easy use.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        patch_type: Literal["tall_narrow", "vit"] = "tall_narrow",
        patch_frames: int = 4,
        stride_frames: int = 2,
        vit_patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 768,
        dropout: float = 0.0,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bins
            patch_type: "tall_narrow" or "vit"
            patch_frames: Time frames per patch (tall_narrow only)
            stride_frames: Stride in time (tall_narrow only)
            vit_patch_size: Patch size for ViT style (freq, time)
            embed_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.mel_transform = MelSpectrogramTransform(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        
        self.patch_type = patch_type
        
        if patch_type == "tall_narrow":
            self.patcher = TallNarrowPatcher(
                n_mels=n_mels,
                patch_frames=patch_frames,
                stride_frames=stride_frames,
                embed_dim=embed_dim,
                dropout=dropout,
            )
        elif patch_type == "vit":
            self.patcher = ViTStylePatcher(
                n_mels=n_mels,
                patch_size=vit_patch_size,
                embed_dim=embed_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown patch type: {patch_type}")
        
        self.output_dim = embed_dim
        self.hop_length = hop_length
    
    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process waveform through mel spectrogram and patching.
        
        Args:
            waveform: [B, T] raw audio
            lengths: Optional lengths [B]
            
        Returns:
            patches: [B, num_patches, embed_dim]
            patch_lengths: Adjusted lengths [B]
        """
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Adjust lengths from samples to spectrogram frames
        spec_lengths = None
        if lengths is not None:
            spec_lengths = lengths // self.hop_length
        
        # Extract patches
        patches, patch_lengths = self.patcher(mel_spec, spec_lengths)
        
        return patches, patch_lengths
