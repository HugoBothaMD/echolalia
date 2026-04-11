"""
Transformer encoder for clinical speech SSL.

Standard transformer architecture with optional modifications
for speech processing.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional relative position encoding."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_relative_pos: bool = False,
        max_relative_pos: int = 128,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_relative_pos = use_relative_pos
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_relative_pos:
            self.max_relative_pos = max_relative_pos
            self.relative_pos_embed = nn.Embedding(
                2 * max_relative_pos + 1,
                self.head_dim,
            )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            attention_mask: [B, T] boolean mask (True = valid)
            
        Returns:
            output: [B, T, D]
        """
        B, T, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias if enabled
        if self.use_relative_pos:
            positions = torch.arange(T, device=x.device)
            relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
            relative_pos = relative_pos.clamp(-self.max_relative_pos, self.max_relative_pos)
            relative_pos = relative_pos + self.max_relative_pos
            
            pos_embed = self.relative_pos_embed(relative_pos)  # [T, T, head_dim]
            pos_bias = torch.einsum('bhid,ijd->bhij', q, pos_embed)
            attn = attn + pos_bias
        
        # Apply attention mask
        if attention_mask is not None:
            # [B, T] -> [B, 1, 1, T]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_proj(output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        use_relative_pos: bool = False,
        pre_norm: bool = True,
    ):
        super().__init__()
        
        self.pre_norm = pre_norm
        
        self.self_attn = MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            use_relative_pos=use_relative_pos,
        )
        
        self.ffn = FeedForward(
            embed_dim,
            ffn_dim,
            dropout=dropout,
            activation=activation,
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            # Pre-normalization (more stable training)
            x = x + self.dropout(self.self_attn(self.norm1(x), attention_mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            # Post-normalization (original transformer)
            x = self.norm1(x + self.dropout(self.self_attn(x, attention_mask)))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for sequence modeling.
    
    Processes a sequence of frame embeddings and outputs
    contextualized representations.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        use_relative_pos: bool = True,
        pre_norm: bool = True,
        layer_drop: float = 0.0,
    ):
        """
        Args:
            embed_dim: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ffn_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            activation: Activation function
            use_relative_pos: Whether to use relative position encodings
            pre_norm: Whether to use pre-normalization
            layer_drop: Probability of dropping a layer during training
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.layer_drop = layer_drop
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
                use_relative_pos=use_relative_pos,
                pre_norm=pre_norm,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm (for pre-norm architecture)
        self.final_norm = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
    
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
            all_layers: List of all layer outputs (if return_all_layers=True)
        """
        all_layer_outputs = [] if return_all_layers else None
        
        for i, layer in enumerate(self.layers):
            # Layer drop during training
            if self.training and self.layer_drop > 0:
                drop_prob = self.layer_drop * (i + 1) / self.num_layers
                if torch.rand(1).item() < drop_prob:
                    continue
            
            x = layer(x, attention_mask)
            
            if return_all_layers:
                all_layer_outputs.append(x)
        
        x = self.final_norm(x)
        
        return x, all_layer_outputs


class TransformerEncoderSmall(TransformerEncoder):
    """Small transformer for faster experimentation."""
    
    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=4,
            ffn_dim=embed_dim * 4,
            dropout=dropout,
        )


class TransformerEncoderBase(TransformerEncoder):
    """Base transformer matching wav2vec 2.0 base."""
    
    def __init__(self, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=12,
            num_layers=12,
            ffn_dim=3072,
            dropout=dropout,
        )


class TransformerEncoderLarge(TransformerEncoder):
    """Large transformer matching wav2vec 2.0 large."""
    
    def __init__(self, embed_dim: int = 1024, dropout: float = 0.1):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=16,
            num_layers=24,
            ffn_dim=4096,
            dropout=dropout,
        )
