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
        num_phoneme_regions: Optional[int] = None,
        predict_magnitude: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            num_augmentation_types: Number of augmentation types to predict
            hidden_dim: Hidden layer dimension
            num_phoneme_regions: If provided, predict per-region
            predict_magnitude: If True, predict magnitude (regression)
                               If False, predict presence only (classification)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_augmentation_types = num_augmentation_types
        self.num_phoneme_regions = num_phoneme_regions
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
        
        if num_phoneme_regions is not None:
            # Per-region prediction
            # Output: [num_regions, num_aug_types] or [num_regions, num_aug_types * 2] 
            # (classification + magnitude)
            output_dim = num_augmentation_types
            if predict_magnitude:
                output_dim *= 2  # presence + magnitude for each
            
            self.region_heads = nn.ModuleList([
                nn.Linear(hidden_dim, output_dim)
                for _ in range(num_phoneme_regions)
            ])
        else:
            # Global prediction (pooled)
            output_dim = num_augmentation_types
            if predict_magnitude:
                output_dim *= 2
            
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
    
    def forward(
        self,
        features: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict augmentation parameters.
        
        Args:
            features: Encoder output [B, T, D]
            gamma: CTC alignment posteriors [B, T, P] (for per-region mode)
            lengths: Sequence lengths [B]
            
        Returns:
            Dictionary containing:
                - 'presence': [B, num_aug_types] or [B, num_regions, num_aug_types]
                - 'magnitude': [B, num_aug_types] or [B, num_regions, num_aug_types]
                  (if predict_magnitude=True)
        """
        B = features.shape[0]
        
        if self.num_phoneme_regions is not None and gamma is not None:
            # Per-region prediction
            region_features = self.pool_to_regions(features, gamma)
            region_features = self.feature_net(region_features)
            
            outputs = []
            for i, head in enumerate(self.region_heads):
                if i < region_features.shape[1]:
                    outputs.append(head(region_features[:, i, :]))
                else:
                    # Pad with zeros if fewer regions than expected
                    outputs.append(torch.zeros(
                        B, head.out_features, device=features.device
                    ))
            
            # Stack: [B, num_regions, output_dim]
            predictions = torch.stack(outputs, dim=1)
            
        else:
            # Global prediction with mean pooling
            if lengths is not None:
                # Masked mean pooling
                mask = torch.arange(features.shape[1], device=features.device)
                mask = mask.unsqueeze(0) < lengths.unsqueeze(1)
                features = features * mask.unsqueeze(-1)
                pooled = features.sum(dim=1) / lengths.unsqueeze(-1).float()
            else:
                pooled = features.mean(dim=1)
            
            pooled = self.feature_net(pooled)
            predictions = self.global_head(pooled)
        
        # Split into presence and magnitude
        result = {}
        
        if self.predict_magnitude:
            if self.num_phoneme_regions is not None:
                presence = predictions[..., :self.num_augmentation_types]
                magnitude = predictions[..., self.num_augmentation_types:]
            else:
                presence = predictions[:, :self.num_augmentation_types]
                magnitude = predictions[:, self.num_augmentation_types:]
            
            result['presence'] = presence  # Logits for BCE
            result['magnitude'] = magnitude  # Regression targets
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
