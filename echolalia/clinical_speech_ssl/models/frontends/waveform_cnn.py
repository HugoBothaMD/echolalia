"""
Waveform CNN frontend for processing raw audio.

Based on wav2vec 2.0 / HuBERT style convolutional feature extractor.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Single convolutional block with optional normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        groups: int = 1,
        norm_type: str = "group",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        
        if norm_type == "group":
            self.norm = nn.GroupNorm(1, out_channels)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(out_channels)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class WaveformCNNFrontend(nn.Module):
    """
    Convolutional frontend for raw waveform input.
    
    Converts raw audio [B, T] to frame-level features [B, T', D]
    where T' << T due to striding.
    
    Default architecture matches wav2vec 2.0:
    - 7 conv layers with kernel sizes [10, 3, 3, 3, 3, 2, 2]
    - Strides [5, 2, 2, 2, 2, 2, 2]
    - Total downsampling: 320x (20ms frames at 16kHz)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: List[int] = [512, 512, 512, 512, 512, 512, 512],
        kernel_sizes: List[int] = [10, 3, 3, 3, 3, 2, 2],
        strides: List[int] = [5, 2, 2, 2, 2, 2, 2],
        norm_type: str = "group",
        dropout: float = 0.0,
        output_dim: Optional[int] = None,
    ):
        """
        Args:
            in_channels: Number of input channels (1 for mono audio)
            hidden_dims: Hidden dimensions for each conv layer
            kernel_sizes: Kernel sizes for each layer
            strides: Stride for each layer
            norm_type: Type of normalization ("group", "layer", "batch", "none")
            dropout: Dropout probability
            output_dim: If provided, project to this dimension
        """
        super().__init__()
        
        assert len(hidden_dims) == len(kernel_sizes) == len(strides)
        
        self.in_channels = in_channels
        self.output_dim = output_dim or hidden_dims[-1]
        
        # Build conv layers
        layers = []
        current_channels = in_channels
        
        for i, (dim, k, s) in enumerate(zip(hidden_dims, kernel_sizes, strides)):
            # Compute padding to minimize information loss
            padding = k // 2
            
            layers.append(ConvBlock(
                current_channels,
                dim,
                kernel_size=k,
                stride=s,
                padding=padding,
                norm_type=norm_type,
                dropout=dropout,
            ))
            current_channels = dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Optional projection
        if output_dim is not None and output_dim != hidden_dims[-1]:
            self.projection = nn.Linear(hidden_dims[-1], output_dim)
        else:
            self.projection = nn.Identity()
        
        # Compute total stride for reference
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s
    
    def get_output_length(self, input_length: int) -> int:
        """Compute output sequence length given input length."""
        length = input_length
        for layer in self.conv_layers:
            k = layer.conv.kernel_size[0]
            s = layer.conv.stride[0]
            p = layer.conv.padding[0]
            length = (length + 2 * p - k) // s + 1
        return length
    
    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process raw waveform.
        
        Args:
            waveform: Raw audio [B, T] or [B, 1, T]
            lengths: Optional original lengths [B]
            
        Returns:
            features: Frame-level features [B, T', D]
            output_lengths: Adjusted lengths [B] if input lengths provided
        """
        # Ensure 3D input [B, C, T]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        # Apply conv layers
        features = self.conv_layers(waveform)  # [B, D, T']
        
        # Transpose to [B, T', D]
        features = features.transpose(1, 2)
        
        # Project if needed
        features = self.projection(features)
        
        # Compute output lengths
        output_lengths = None
        if lengths is not None:
            output_lengths = torch.tensor([
                self.get_output_length(l.item()) for l in lengths
            ], device=lengths.device)
        
        return features, output_lengths


class WaveformCNNFrontendSmall(WaveformCNNFrontend):
    """Smaller variant with 4 layers for faster experimentation."""
    
    def __init__(
        self,
        output_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__(
            in_channels=1,
            hidden_dims=[256, 256, 256, 256],
            kernel_sizes=[10, 3, 3, 3],
            strides=[5, 2, 2, 2],
            output_dim=output_dim,
            dropout=dropout,
        )


class WaveformCNNFrontendLarge(WaveformCNNFrontend):
    """Larger variant matching wav2vec 2.0 large."""
    
    def __init__(
        self,
        output_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__(
            in_channels=1,
            hidden_dims=[512, 512, 512, 512, 512, 512, 512],
            kernel_sizes=[10, 3, 3, 3, 3, 2, 2],
            strides=[5, 2, 2, 2, 2, 2, 2],
            output_dim=output_dim,
            dropout=dropout,
        )
