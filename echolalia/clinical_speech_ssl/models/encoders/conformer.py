"""
Conformer encoder for clinical speech SSL.

Conformer combines convolutions (for local patterns) with
self-attention (for global context), making it well-suited
for speech processing.

Reference: Gulati et al., "Conformer: Convolution-augmented 
Transformer for Speech Recognition", 2020.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module.
    
    Captures local patterns with depthwise separable convolutions.
    """
    
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 31,
        dropout: float = 0.0,
        expansion_factor: int = 2,
    ):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        inner_dim = embed_dim * expansion_factor
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Pointwise conv -> GLU -> Depthwise conv -> BatchNorm -> Swish -> Pointwise
        self.pointwise1 = nn.Linear(embed_dim, inner_dim * 2)
        
        self.depthwise = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner_dim,
        )
        
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.activation = nn.SiLU()  # Swish
        
        self.pointwise2 = nn.Linear(inner_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            
        Returns:
            output: [B, T, D]
        """
        x = self.layer_norm(x)
        
        # Pointwise + GLU
        x = self.pointwise1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        
        # Depthwise conv (need to transpose for Conv1d)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # [B, T, D]
        
        # Final pointwise
        x = self.pointwise2(x)
        x = self.dropout(x)
        
        return x


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative position encoding for Conformer."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Relative position embedding
        self.pos_bias_u = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        self.pos_bias_v = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        
        self.dropout = nn.Dropout(dropout)
    
    def _relative_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative position shift for efficient computation."""
        B, H, T, _ = x.shape
        
        # Pad and reshape
        x = F.pad(x, (1, 0))
        x = x.view(B, H, -1, T)
        x = x[:, :, 1:, :]
        x = x.view(B, H, T, T)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            pos_emb: Positional embeddings [1, 2T-1, D]
            attention_mask: [B, T]
            
        Returns:
            output: [B, T, D]
        """
        B, T, D = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project position embeddings
        pos_emb = pos_emb.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention with relative position
        q_with_bias_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        
        # Content attention
        content_attn = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        
        # Position attention
        pos_attn = torch.matmul(q_with_bias_v, pos_emb.transpose(-2, -1))
        pos_attn = self._relative_shift(pos_attn)
        
        # Combine
        attn = (content_attn + pos_attn) * self.scale
        
        # Apply mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_proj(output)
        
        return output


class FeedForwardModule(nn.Module):
    """Conformer feed-forward module with pre-norm."""
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.SiLU()  # Swish
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConformerBlock(nn.Module):
    """
    Single Conformer block.
    
    Structure: FFN -> MHA -> Conv -> FFN
    With half-step residual connections for FFN modules.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        conv_dropout: float = 0.1,
    ):
        super().__init__()
        
        # First feed-forward module (half-step residual)
        self.ffn1 = FeedForwardModule(embed_dim, ffn_dim, dropout)
        
        # Multi-head self-attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = RelativeMultiHeadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
        )
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_module = ConvolutionModule(
            embed_dim,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        
        # Second feed-forward module (half-step residual)
        self.ffn2 = FeedForwardModule(embed_dim, ffn_dim, dropout)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            pos_emb: Positional embeddings [1, 2T-1, D]
            attention_mask: [B, T]
            
        Returns:
            output: [B, T, D]
        """
        # First FFN (half-step)
        x = x + 0.5 * self.ffn1(x)
        
        # Self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, pos_emb, attention_mask)
        x = residual + self.attn_dropout(x)
        
        # Convolution module
        x = x + self.conv_module(x)
        
        # Second FFN (half-step)
        x = x + 0.5 * self.ffn2(x)
        
        # Final layer norm
        x = self.final_norm(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Relative sinusoidal positional encoding."""
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Create position encoding matrix
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-math.log(10000.0) / embed_dim)
        )
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, length: int) -> torch.Tensor:
        """
        Generate positional embeddings for relative attention.
        
        Args:
            length: Sequence length
            
        Returns:
            pos_emb: [1, 2*length-1, D]
        """
        # For relative position, we need positions from -(length-1) to (length-1)
        center = length - 1
        pos_emb = self.pe[:2 * length - 1].unsqueeze(0)
        return pos_emb


class ConformerEncoder(nn.Module):
    """
    Conformer encoder for speech processing.
    
    Combines convolutions for local patterns with attention
    for global context.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 12,
        ffn_dim: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        conv_dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of conformer blocks
            ffn_dim: Feed-forward hidden dimension
            conv_kernel_size: Kernel size for conv module
            dropout: General dropout
            attention_dropout: Attention dropout
            conv_dropout: Convolution dropout
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
                conv_dropout=conv_dropout,
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            x: Input embeddings [B, T, D]
            attention_mask: Attention mask [B, T]
            return_all_layers: Whether to return all layer outputs
            
        Returns:
            output: Final layer output [B, T, D]
            all_layers: List of all layer outputs (if requested)
        """
        T = x.shape[1]
        pos_emb = self.pos_encoding(T)
        
        all_layer_outputs = [] if return_all_layers else None
        
        for layer in self.layers:
            x = layer(x, pos_emb, attention_mask)
            
            if return_all_layers:
                all_layer_outputs.append(x)
        
        return x, all_layer_outputs


class ConformerEncoderSmall(ConformerEncoder):
    """Small conformer for faster experimentation."""
    
    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=4,
            ffn_dim=embed_dim * 4,
            conv_kernel_size=15,
            dropout=dropout,
        )


class ConformerEncoderMedium(ConformerEncoder):
    """Medium conformer."""
    
    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=12,
            ffn_dim=embed_dim * 4,
            conv_kernel_size=31,
            dropout=dropout,
        )


class ConformerEncoderLarge(ConformerEncoder):
    """Large conformer."""
    
    def __init__(self, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=17,
            ffn_dim=embed_dim * 4,
            conv_kernel_size=31,
            dropout=dropout,
        )
