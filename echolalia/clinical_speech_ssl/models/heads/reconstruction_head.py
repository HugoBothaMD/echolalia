"""
Masked reconstruction head for SSL.

Implements masked prediction objective where the model must
reconstruct masked portions of the input, with configurable
bias toward masking transitions.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskGenerator(nn.Module):
    """
    Generates masks for masked prediction with transition bias.
    
    Can bias masking toward phoneme transitions (high entropy in gamma)
    to encourage learning of coarticulation patterns.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_span_length: int = 10,
        min_masks: int = 1,
        transition_bias: float = 2.0,
        entropy_threshold: float = 0.3,
    ):
        """
        Args:
            mask_prob: Base probability of masking each frame
            mask_span_length: Length of masked spans
            min_masks: Minimum number of masks per sequence
            transition_bias: Factor to increase masking prob at transitions
            entropy_threshold: Entropy threshold for transition detection
        """
        super().__init__()
        
        self.mask_prob = mask_prob
        self.mask_span_length = mask_span_length
        self.min_masks = min_masks
        self.transition_bias = transition_bias
        self.entropy_threshold = entropy_threshold
    
    def identify_transitions(self, gamma: torch.Tensor) -> torch.Tensor:
        """
        Identify transition frames from CTC gamma matrix.
        
        Args:
            gamma: CTC posteriors [T, num_phonemes] or [B, T, num_phonemes]
            
        Returns:
            is_transition: Boolean tensor indicating transitions
        """
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(0)
        
        # Compute entropy at each frame
        gamma_safe = gamma.clamp(min=1e-10)
        entropy = -(gamma_safe * gamma_safe.log()).sum(dim=-1)
        max_entropy = np.log(gamma.shape[-1])
        normalized_entropy = entropy / max_entropy
        
        # High entropy = transition
        is_transition = normalized_entropy > self.entropy_threshold
        
        return is_transition.squeeze(0) if gamma.shape[0] == 1 else is_transition
    
    def generate_mask(
        self,
        seq_length: int,
        gamma: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate mask for a single sequence.
        
        Args:
            seq_length: Length of sequence
            gamma: Optional CTC posteriors for transition bias
            device: Device for output tensor
            
        Returns:
            mask: Boolean tensor [seq_length] where True = masked
        """
        device = device or (gamma.device if gamma is not None else torch.device('cpu'))
        
        # Compute per-frame masking probability
        if gamma is not None and self.transition_bias > 1.0:
            is_transition = self.identify_transitions(gamma)
            mask_probs = torch.where(
                is_transition,
                torch.tensor(self.mask_prob * self.transition_bias, device=device),
                torch.tensor(self.mask_prob, device=device),
            )
        else:
            mask_probs = torch.full((seq_length,), self.mask_prob, device=device)
        
        # Sample span starting points
        num_spans = max(
            self.min_masks,
            int(seq_length * self.mask_prob / self.mask_span_length)
        )
        
        # Weight span starts by mask probability
        span_starts = torch.multinomial(
            mask_probs / mask_probs.sum(),
            min(num_spans, seq_length),
            replacement=False,
        )
        
        # Create mask by expanding spans
        mask = torch.zeros(seq_length, dtype=torch.bool, device=device)
        
        for start in span_starts:
            end = min(start + self.mask_span_length, seq_length)
            mask[start:end] = True
        
        return mask
    
    def forward(
        self,
        batch_size: int,
        seq_length: int,
        gammas: Optional[List[torch.Tensor]] = None,
        lengths: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate masks for a batch.
        
        Args:
            batch_size: Number of sequences
            seq_length: Padded sequence length
            gammas: List of CTC posteriors per sample
            lengths: Actual lengths [B]
            device: Output device
            
        Returns:
            mask: Boolean tensor [B, seq_length]
        """
        masks = []
        
        for i in range(batch_size):
            actual_length = lengths[i].item() if lengths is not None else seq_length
            gamma = gammas[i] if gammas is not None else None
            
            mask_i = self.generate_mask(actual_length, gamma, device)
            
            # Pad to seq_length
            if actual_length < seq_length:
                padding = torch.zeros(
                    seq_length - actual_length,
                    dtype=torch.bool,
                    device=device,
                )
                mask_i = torch.cat([mask_i, padding])
            
            masks.append(mask_i)
        
        return torch.stack(masks)


class ReconstructionHead(nn.Module):
    """
    Head for reconstructing masked frames.
    
    Can reconstruct either:
    1. Input features (like wav2vec 2.0)
    2. Quantized codes (like HuBERT)
    3. Spectrogram frames
    """
    
    def __init__(
        self,
        embed_dim: int,
        target_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            target_dim: Target reconstruction dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.target_dim = target_dim

        hidden_dim = hidden_dim or embed_dim

        # Build projection network
        layers = []
        current_dim = embed_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, target_dim))
        self.projection = nn.Sequential(*layers)
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project features to reconstruction targets.
        
        Args:
            features: Encoder output [B, T, D]
            mask: Optional mask to select frames [B, T]
            
        Returns:
            predictions: [B, T, target_dim] or [B, T, num_codes] if quantizer
        """
        if mask is not None:
            # Only process masked frames
            # This is more efficient but requires careful handling
            pass
        
        predictions = self.projection(features)
        
        return predictions
    
    def get_targets(
        self,
        original_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get reconstruction targets.

        Args:
            original_features: Original input features [B, T, D]
            mask: Mask indicating which frames to reconstruct [B, T]

        Returns:
            targets: Target values for masked frames
        """
        return original_features


class MaskedReconstructionLoss(nn.Module):
    """
    Loss function for masked reconstruction objective.
    """

    def __init__(
        self,
        use_cosine: bool = True,
        temperature: float = 0.1,
    ):
        """
        Args:
            use_cosine: If True, use cosine similarity loss
            temperature: Temperature for cosine similarity
        """
        super().__init__()

        self.use_cosine = use_cosine
        self.temperature = temperature

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.

        Args:
            predictions: Model predictions [B, T, D]
            targets: Target values [B, T, D]
            mask: Boolean mask [B, T] where True = masked

        Returns:
            Dictionary with 'loss'
        """
        predictions_masked = predictions[mask]  # [N, D]
        targets_masked = targets[mask]  # [N, D]

        if self.use_cosine:
            predictions_norm = F.normalize(predictions_masked, dim=-1)
            targets_norm = F.normalize(targets_masked, dim=-1)
            similarity = (predictions_norm * targets_norm).sum(dim=-1)
            loss = (1 - similarity).mean()
        else:
            loss = F.mse_loss(predictions_masked, targets_masked)

        return {'loss': loss}


class MaskedPredictionModule(nn.Module):
    """
    Complete module for masked prediction SSL objective.
    
    Combines mask generation, reconstruction head, and loss computation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        target_dim: int,
        mask_prob: float = 0.15,
        mask_span_length: int = 10,
        transition_bias: float = 2.0,
        use_cosine_loss: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.mask_generator = MaskGenerator(
            mask_prob=mask_prob,
            mask_span_length=mask_span_length,
            transition_bias=transition_bias,
        )
        
        self.reconstruction_head = ReconstructionHead(
            embed_dim=embed_dim,
            target_dim=target_dim,
            dropout=dropout,
        )
        
        self.loss_fn = MaskedReconstructionLoss(use_cosine=use_cosine_loss)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
    
    def apply_mask(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace masked positions with mask token.
        
        Args:
            features: Input features [B, T, D]
            mask: Boolean mask [B, T]
            
        Returns:
            masked_features: Features with mask token at masked positions
        """
        masked_features = features.clone()
        masked_features[mask] = self.mask_token
        return masked_features
    
    def forward(
        self,
        input_features: torch.Tensor,
        encoder_features: torch.Tensor,
        gammas: Optional[List[torch.Tensor]] = None,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute masked prediction loss.
        
        Args:
            input_features: Original input to encoder [B, T, D]
            encoder_features: Encoder output [B, T, D]
            gammas: CTC alignments for transition bias
            lengths: Sequence lengths
            mask: Pre-computed mask (if None, generates new)
            
        Returns:
            Dictionary with loss and optional metrics
        """
        B, T, _ = encoder_features.shape
        
        # Generate mask if not provided
        if mask is None:
            mask = self.mask_generator(
                batch_size=B,
                seq_length=T,
                gammas=gammas,
                lengths=lengths,
                device=encoder_features.device,
            )
        
        # Get predictions for masked positions
        predictions = self.reconstruction_head(encoder_features, mask)
        
        # Get targets
        targets = self.reconstruction_head.get_targets(input_features, mask)
        
        # Compute loss
        result = self.loss_fn(predictions, targets, mask)
        result['mask'] = mask
        
        return result
