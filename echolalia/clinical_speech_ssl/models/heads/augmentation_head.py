"""
Augmentation prediction head for SSL.

This head predicts what augmentations were applied to each region
of the input audio, forcing the model to learn representations
sensitive to acoustic perturbations.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AugmentationPredictionHead(nn.Module):
    """
    Predicts augmentation parameters for each phoneme region.
    
    Can operate in two modes:
    1. Per-region prediction: Predicts augmentations for each phoneme region
    2. Global prediction: Predicts augmentations applied to the whole utterance
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_augmentation_types: int = 4,
        hidden_dim: Optional[int] = None,
        predict_magnitude: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            num_augmentation_types: Number of augmentation types to predict
            hidden_dim: Hidden layer dimension
            predict_magnitude: If True, predict magnitude (regression)
                               If False, predict presence only (classification)
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_augmentation_types = num_augmentation_types
        self.predict_magnitude = predict_magnitude
        
        hidden_dim = hidden_dim or embed_dim // 2

        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Shared prediction head (works for both global and per-region modes)
        output_dim = num_augmentation_types
        if predict_magnitude:
            output_dim *= 2  # presence + magnitude for each
        self.global_head = nn.Linear(hidden_dim, output_dim)
    
    def pool_to_regions(
        self,
        features: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool frame features to phoneme regions using gamma matrix.
        
        Args:
            features: Frame features [B, T, D]
            gamma: CTC posteriors [B, T, num_phonemes]
            
        Returns:
            region_features: [B, num_phonemes, D]
        """
        # Normalize gamma to sum to 1 per phoneme
        gamma_sum = gamma.sum(dim=1, keepdim=True).clamp(min=1e-10)
        gamma_norm = gamma / gamma_sum
        
        # Weighted sum of features for each phoneme
        # [B, T, P] x [B, T, D] -> [B, P, D]
        region_features = torch.einsum('btp,btd->bpd', gamma_norm, features)
        
        return region_features
    
    def forward_with_regions(
        self,
        features: torch.Tensor,
        region_masks: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict augmentation parameters per region using explicit region masks.

        Args:
            features: Encoder output [B, T, D]
            region_masks: Per-region weight masks [B, K, T] (K = max regions)
            valid_mask: Boolean [B, K] indicating which regions are real vs padding

        Returns:
            Dictionary containing:
                - 'presence': [B, K, num_aug_types] logits
                - 'magnitude': [B, K, num_aug_types] (if predict_magnitude=True)
        """
        # Normalize region masks to sum-to-1 per region
        mask_sum = region_masks.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        region_weights = region_masks / mask_sum  # [B, K, T]

        # Pool features per region: [B, K, T] x [B, T, D] -> [B, K, D]
        region_features = torch.einsum('bkt,btd->bkd', region_weights, features)

        # Apply shared feature net + head
        region_features = self.feature_net(region_features)
        predictions = self.global_head(region_features)  # [B, K, output_dim]

        return self._split_predictions(predictions)

    def forward(
        self,
        features: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict augmentation parameters (global pooling fallback).

        Args:
            features: Encoder output [B, T, D]
            gamma: CTC alignment posteriors [B, T, P] (for gamma-pooled mode)
            lengths: Sequence lengths [B]

        Returns:
            Dictionary containing:
                - 'presence': [B, num_aug_types]
                - 'magnitude': [B, num_aug_types] (if predict_magnitude=True)
        """
        if gamma is not None:
            # Gamma-based phoneme pooling
            region_features = self.pool_to_regions(features, gamma)
            region_features = self.feature_net(region_features)
            predictions = self.global_head(region_features)  # [B, P, output_dim]
        else:
            # Global prediction with mean pooling
            if lengths is not None:
                mask = torch.arange(features.shape[1], device=features.device)
                mask = mask.unsqueeze(0) < lengths.unsqueeze(1)
                features = features * mask.unsqueeze(-1)
                pooled = features.sum(dim=1) / lengths.unsqueeze(-1).float()
            else:
                pooled = features.mean(dim=1)

            pooled = self.feature_net(pooled)
            predictions = self.global_head(pooled)

        return self._split_predictions(predictions)

    def _split_predictions(
        self,
        predictions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Split raw predictions into presence and magnitude dicts."""
        result = {}
        if self.predict_magnitude:
            presence = predictions[..., :self.num_augmentation_types]
            magnitude = predictions[..., self.num_augmentation_types:]
            result['presence'] = presence
            result['magnitude'] = magnitude
        else:
            result['presence'] = predictions
        return result


class AugmentationPredictionLoss(nn.Module):
    """
    Loss function for augmentation prediction objective.
    
    Combines binary cross-entropy for presence detection
    and MSE for magnitude regression.
    """
    
    def __init__(
        self,
        magnitude_weight: float = 1.0,
        presence_weight: float = 1.0,
        transition_weight: float = 2.0,
        per_region: bool = False,
    ):
        """
        Args:
            magnitude_weight: Weight for magnitude regression loss
            presence_weight: Weight for presence classification loss
            transition_weight: Additional weight for transition regions
            per_region: Whether predictions are per-region
        """
        super().__init__()
        
        self.magnitude_weight = magnitude_weight
        self.presence_weight = presence_weight
        self.transition_weight = transition_weight
        self.per_region = per_region
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        is_transition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute augmentation prediction loss.
        
        Args:
            predictions: Output from AugmentationPredictionHead
            targets: Dictionary with 'presence' (binary) and 'magnitude' (float)
            is_transition: Boolean mask indicating transition regions [B, num_regions]
            
        Returns:
            Dictionary with 'total_loss', 'presence_loss', 'magnitude_loss'
        """
        presence_pred = predictions['presence']
        presence_target = targets['presence'].float()
        
        # Presence loss (BCE)
        presence_loss = self.bce(presence_pred, presence_target)
        
        # Apply transition weighting
        if is_transition is not None and self.per_region:
            # [B, num_regions] -> [B, num_regions, 1]
            weight = torch.where(
                is_transition.unsqueeze(-1),
                torch.tensor(self.transition_weight, device=presence_loss.device),
                torch.tensor(1.0, device=presence_loss.device),
            )
            presence_loss = presence_loss * weight
        
        presence_loss = presence_loss.mean()
        
        # Magnitude loss (MSE, only where augmentation was applied)
        magnitude_loss = torch.tensor(0.0, device=presence_loss.device)
        
        if 'magnitude' in predictions and 'magnitude' in targets:
            magnitude_pred = predictions['magnitude']
            magnitude_target = targets['magnitude']
            
            # Only compute loss where augmentation was applied
            mask = presence_target > 0.5
            
            if mask.any():
                mag_loss = self.mse(magnitude_pred, magnitude_target)
                mag_loss = (mag_loss * mask).sum() / mask.sum().clamp(min=1)
                magnitude_loss = mag_loss
        
        total_loss = (
            self.presence_weight * presence_loss +
            self.magnitude_weight * magnitude_loss
        )
        
        return {
            'total_loss': total_loss,
            'presence_loss': presence_loss,
            'magnitude_loss': magnitude_loss,
        }
