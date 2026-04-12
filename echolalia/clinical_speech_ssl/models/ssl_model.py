"""
Main SSL model combining frontend, encoder, and objective heads.

This is the primary model class for clinical speech SSL with
configurable components and objectives.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn

from clinical_speech_ssl.models.frontends import (
    WaveformCNNFrontend,
    WaveformCNNFrontendSmall,
    WaveformCNNFrontendLarge,
    SpectrogramPatchFrontend,
)
from clinical_speech_ssl.models.encoders import (
    TransformerEncoder,
    TransformerEncoderSmall,
    TransformerEncoderBase,
    ConformerEncoder,
    ConformerEncoderSmall,
    ConformerEncoderMedium,
)
from clinical_speech_ssl.models.heads import (
    AugmentationPredictionHead,
    AugmentationPredictionLoss,
    MaskedPredictionModule,
    ContrastiveModule,
)
from clinical_speech_ssl.models.heads.gop_head import (
    GOPPredictionHead,
    GOPPredictionLoss,
)
from clinical_speech_ssl.data.augmentations import (
    RegionAugmentor,
    SafeAugmentor,
    AugmentationConfig,
    CLINICAL_AUGMENTATIONS,
    CLINICAL_AUGMENTATION_INDEX,
)


@dataclass
class ClinicalSpeechSSLConfig:
    """Configuration for the SSL model."""
    
    # Input configuration
    input_type: Literal["waveform", "spectrogram"] = "waveform"
    sample_rate: int = 16000
    
    # Frontend configuration
    frontend_type: Literal["cnn_small", "cnn_base", "cnn_large", 
                          "patch_tall_narrow", "patch_vit"] = "cnn_base"
    frontend_dropout: float = 0.0
    
    # Encoder configuration
    encoder_type: Literal["transformer_small", "transformer_base", "transformer_large",
                         "conformer_small", "conformer_medium", "conformer_large"] = "conformer_medium"
    embed_dim: int = 256
    encoder_dropout: float = 0.1
    
    # SSL objectives (enable/disable)
    use_augmentation_prediction: bool = True
    use_masked_reconstruction: bool = True
    use_contrastive: bool = True
    use_gop_prediction: bool = False
    
    # Augmentation prediction config
    num_augmentation_types: int = 4
    augmentation_per_region: bool = True
    predict_magnitude: bool = True
    
    # Masked reconstruction config
    mask_prob: float = 0.15
    mask_span_length: int = 10
    transition_bias: float = 2.0
    
    # Contrastive config
    contrastive_projection_dim: int = 256
    contrastive_temperature: float = 0.07
    
    # Loss weights
    aug_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0
    contrastive_loss_weight: float = 1.0
    gop_loss_weight: float = 0.5
    
    # Spectrogram-specific config
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    patch_frames: int = 4
    patch_stride: int = 2
    vit_patch_size: Tuple[int, int] = (16, 16)


class ClinicalSpeechSSL(nn.Module):
    """
    Clinical Speech SSL Model.
    
    A modular self-supervised learning model for clinical speech that combines:
    1. Configurable input frontend (waveform CNN or spectrogram patches)
    2. Configurable encoder (Transformer or Conformer)
    3. Multiple SSL objectives:
       - Augmentation prediction (detect/measure acoustic perturbations)
       - Masked reconstruction (predict masked frames, biased toward transitions)
       - Contrastive learning (invariance to recording conditions)
    
    Example usage:
        config = ClinicalSpeechSSLConfig(
            input_type="waveform",
            encoder_type="conformer_medium",
            use_augmentation_prediction=True,
            use_masked_reconstruction=True,
            use_contrastive=True,
        )
        model = ClinicalSpeechSSL(config)
        
        # Training
        losses = model(waveforms, gammas, lengths)
        total_loss = losses['total_loss']
        
        # Extract representations
        features = model.encode(waveforms, lengths)
    """
    
    def __init__(self, config: ClinicalSpeechSSLConfig):
        super().__init__()
        
        self.config = config
        
        # Build frontend
        self.frontend = self._build_frontend()
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build SSL heads
        self.aug_head = None
        self.mask_module = None
        self.contrastive_module = None
        
        if config.use_augmentation_prediction:
            self.aug_head = AugmentationPredictionHead(
                embed_dim=config.embed_dim,
                num_augmentation_types=config.num_augmentation_types,
                predict_magnitude=config.predict_magnitude,
            )
            self.aug_loss = AugmentationPredictionLoss(
                per_region=config.augmentation_per_region,
            )
        
        if config.use_masked_reconstruction:
            self.mask_module = MaskedPredictionModule(
                embed_dim=config.embed_dim,
                target_dim=config.embed_dim,
                mask_prob=config.mask_prob,
                mask_span_length=config.mask_span_length,
                transition_bias=config.transition_bias,
            )
        
        if config.use_contrastive:
            self.contrastive_module = ContrastiveModule(
                embed_dim=config.embed_dim,
                projection_dim=config.contrastive_projection_dim,
                temperature=config.contrastive_temperature,
            )
        
        if config.use_gop_prediction:
            self.gop_head = GOPPredictionHead(
                embed_dim=config.embed_dim,
            )
            self.gop_loss = GOPPredictionLoss()
        else:
            self.gop_head = None
            self.gop_loss = None

        # Augmentors
        self.region_augmentor = RegionAugmentor(
            sample_rate=config.sample_rate,
            transition_bias=config.transition_bias,
            clinical_only=True,
        )
        self.safe_augmentor = SafeAugmentor(sample_rate=config.sample_rate)
    
    def _build_frontend(self) -> nn.Module:
        """Build input frontend based on config."""
        config = self.config
        
        if config.input_type == "waveform":
            if config.frontend_type == "cnn_small":
                return WaveformCNNFrontendSmall(
                    output_dim=config.embed_dim,
                    dropout=config.frontend_dropout,
                )
            elif config.frontend_type == "cnn_base":
                return WaveformCNNFrontend(
                    output_dim=config.embed_dim,
                    dropout=config.frontend_dropout,
                )
            elif config.frontend_type == "cnn_large":
                return WaveformCNNFrontendLarge(
                    output_dim=config.embed_dim,
                    dropout=config.frontend_dropout,
                )
            else:
                raise ValueError(f"Unknown frontend type for waveform: {config.frontend_type}")
        
        elif config.input_type == "spectrogram":
            patch_type = "tall_narrow" if "tall_narrow" in config.frontend_type else "vit"
            return SpectrogramPatchFrontend(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                patch_type=patch_type,
                patch_frames=config.patch_frames,
                stride_frames=config.patch_stride,
                vit_patch_size=config.vit_patch_size,
                embed_dim=config.embed_dim,
                dropout=config.frontend_dropout,
            )
        
        else:
            raise ValueError(f"Unknown input type: {config.input_type}")
    
    def _build_encoder(self) -> nn.Module:
        """Build encoder based on config."""
        config = self.config
        
        if config.encoder_type == "transformer_small":
            return TransformerEncoderSmall(
                embed_dim=config.embed_dim,
                dropout=config.encoder_dropout,
            )
        elif config.encoder_type == "transformer_base":
            return TransformerEncoderBase(
                embed_dim=config.embed_dim,
                dropout=config.encoder_dropout,
            )
        elif config.encoder_type == "conformer_small":
            return ConformerEncoderSmall(
                embed_dim=config.embed_dim,
                dropout=config.encoder_dropout,
            )
        elif config.encoder_type == "conformer_medium":
            return ConformerEncoderMedium(
                embed_dim=config.embed_dim,
                dropout=config.encoder_dropout,
            )
        elif config.encoder_type == "conformer_large":
            return ConformerEncoder(
                embed_dim=config.embed_dim,
                num_heads=8,
                num_layers=17,
                ffn_dim=config.embed_dim * 4,
                dropout=config.encoder_dropout,
            )
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")
    
    def encode(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Encode audio to representations.
        
        Args:
            waveform: Input audio [B, T]
            lengths: Audio lengths [B]
            return_all_layers: Whether to return all encoder layers
            
        Returns:
            features: Encoder output [B, T', D]
            all_layers: List of all layer outputs (if requested)
        """
        # Frontend
        features, feature_lengths = self.frontend(waveform, lengths)
        
        # Create attention mask
        attention_mask = None
        if feature_lengths is not None:
            max_len = features.shape[1]
            attention_mask = torch.arange(max_len, device=features.device)
            attention_mask = attention_mask.unsqueeze(0) < feature_lengths.unsqueeze(1)
        
        # Encoder
        encoded, all_layers = self.encoder(
            features,
            attention_mask=attention_mask,
            return_all_layers=return_all_layers,
        )
        
        if return_all_layers:
            return encoded, all_layers
        return encoded
    
    def forward(
        self,
        waveform: torch.Tensor,
        gammas: Optional[List[torch.Tensor]] = None,
        lengths: Optional[torch.Tensor] = None,
        gop_targets: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing all SSL losses.

        Args:
            waveform: Input audio [B, T]
            gammas: List of CTC gamma matrices (one per sample)
            lengths: Audio lengths [B]
            gop_targets: Optional list of per-phoneme GOP score tensors [P_i]

        Returns:
            Dictionary containing:
                - 'total_loss': Combined weighted loss
                - 'aug_*': Augmentation prediction losses
                - 'mask_*': Masked reconstruction losses
                - 'contrastive_*': Contrastive losses
                - 'gop_*': GOP prediction losses (if enabled)
        """
        B = waveform.shape[0]
        device = waveform.device
        losses = {}

        # --- Augmentation Prediction Objective ---
        if self.aug_head is not None:
            aug_losses = self._compute_augmentation_loss(waveform, gammas, lengths)
            losses.update({f'aug_{k}': v for k, v in aug_losses.items()})

        # --- Masked Reconstruction Objective ---
        if self.mask_module is not None:
            mask_losses = self._compute_mask_loss(waveform, gammas, lengths)
            losses.update({f'mask_{k}': v for k, v in mask_losses.items()})

        # --- Contrastive Objective ---
        if self.contrastive_module is not None:
            contrastive_losses = self._compute_contrastive_loss(waveform, lengths)
            losses.update({f'contrastive_{k}': v for k, v in contrastive_losses.items()})

        # --- GOP Prediction Objective ---
        if self.gop_head is not None and gammas is not None and gop_targets is not None:
            gop_losses = self._compute_gop_loss(waveform, gammas, lengths, gop_targets)
            losses.update({f'gop_{k}': v for k, v in gop_losses.items()})

        # Combine losses
        total_loss = torch.tensor(0.0, device=device)

        if 'aug_total_loss' in losses:
            total_loss = total_loss + self.config.aug_loss_weight * losses['aug_total_loss']

        if 'mask_loss' in losses:
            total_loss = total_loss + self.config.mask_loss_weight * losses['mask_loss']
        
        if 'contrastive_loss' in losses:
            total_loss = total_loss + self.config.contrastive_loss_weight * losses['contrastive_loss']

        if 'gop_loss' in losses:
            total_loss = total_loss + self.config.gop_loss_weight * losses['gop_loss']

        losses['total_loss'] = total_loss

        return losses
    
    def _compute_augmentation_loss(
        self,
        waveform: torch.Tensor,
        gammas: Optional[List[torch.Tensor]],
        lengths: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute augmentation prediction loss."""
        B = waveform.shape[0]
        device = waveform.device
        num_aug_types = self.config.num_augmentation_types

        # Apply augmentations and collect per-region labels
        augmented_waveforms = []
        all_region_labels = []  # per-sample list of RegionAugmentationLabel

        for i in range(B):
            wav_i = waveform[i]
            gamma_i = gammas[i] if gammas is not None else None
            result = self.region_augmentor(wav_i, gamma_i)
            augmented_waveforms.append(result.waveform)
            all_region_labels.append(result.region_labels)

        augmented_waveforms = torch.stack(augmented_waveforms)
        # Augmentation is region-level; total waveform length must be preserved
        assert augmented_waveforms.shape == waveform.shape, (
            f"Augmented shape {augmented_waveforms.shape} != original {waveform.shape}"
        )

        # Encode augmented audio
        features = self.encode(augmented_waveforms, lengths)
        T_feat = features.shape[1]
        # Approximate samples-per-frame from frontend downsampling
        samples_per_frame = waveform.shape[1] // T_feat if T_feat > 0 else 1

        # Build per-region masks and targets
        # Each sample may have a different number of region labels K_i.
        # We pad to the max K across the batch.
        max_K = max(len(rl) for rl in all_region_labels) if all_region_labels else 1
        max_K = max(max_K, 1)

        region_masks = torch.zeros(B, max_K, T_feat, device=device)
        presence_targets = torch.zeros(B, max_K, num_aug_types, device=device)
        magnitude_targets = torch.zeros(B, max_K, num_aug_types, device=device)
        is_transition = torch.zeros(B, max_K, dtype=torch.bool, device=device)
        valid_mask = torch.zeros(B, max_K, dtype=torch.bool, device=device)

        for i, rlabels in enumerate(all_region_labels):
            for k, rl in enumerate(rlabels):
                start_frame = rl.start_sample // samples_per_frame
                end_frame = min(rl.end_sample // samples_per_frame + 1, T_feat)
                region_masks[i, k, start_frame:end_frame] = 1.0

                j = CLINICAL_AUGMENTATION_INDEX.get(rl.aug_type)
                if j is not None:
                    presence_targets[i, k, j] = 1.0
                    magnitude_targets[i, k, j] = rl.param_normalized
                is_transition[i, k] = rl.is_transition
                valid_mask[i, k] = True

        # Predict augmentations per region
        predictions = self.aug_head.forward_with_regions(
            features, region_masks, valid_mask,
        )

        aug_targets = {
            'presence': presence_targets,
            'magnitude': magnitude_targets,
        }

        loss_dict = self.aug_loss(predictions, aug_targets, is_transition=is_transition)

        return loss_dict
    
    def _compute_mask_loss(
        self,
        waveform: torch.Tensor,
        gammas: Optional[List[torch.Tensor]],
        lengths: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute masked reconstruction loss."""
        # Get input features (before masking)
        input_features, feature_lengths = self.frontend(waveform, lengths)
        
        # Generate mask
        mask = self.mask_module.mask_generator(
            batch_size=input_features.shape[0],
            seq_length=input_features.shape[1],
            gammas=gammas,
            lengths=feature_lengths,
            device=input_features.device,
        )
        
        # Apply mask to input
        masked_features = self.mask_module.apply_mask(input_features, mask)
        
        # Create attention mask
        attention_mask = None
        if feature_lengths is not None:
            max_len = masked_features.shape[1]
            attention_mask = torch.arange(max_len, device=masked_features.device)
            attention_mask = attention_mask.unsqueeze(0) < feature_lengths.unsqueeze(1)
        
        # Encode masked input
        encoded, _ = self.encoder(masked_features, attention_mask=attention_mask)
        
        # Compute reconstruction loss
        predictions = self.mask_module.reconstruction_head(encoded)
        targets = input_features
        
        loss_dict = self.mask_module.loss_fn(predictions, targets, mask)
        
        return loss_dict
    
    def _compute_contrastive_loss(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss with safe augmentations."""
        # Create two views with different safe augmentations
        view_a = waveform  # Original
        view_b = torch.stack([
            self.safe_augmentor(waveform[i])
            for i in range(waveform.shape[0])
        ])
        
        # Encode both views
        features_a = self.encode(view_a, lengths)
        features_b = self.encode(view_b, lengths)
        
        # Compute contrastive loss
        loss_dict = self.contrastive_module(features_a, features_b)
        
        return loss_dict

    def _compute_gop_loss(
        self,
        waveform: torch.Tensor,
        gammas: List[torch.Tensor],
        lengths: Optional[torch.Tensor],
        gop_targets: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute GOP prediction loss."""
        features = self.encode(waveform, lengths)
        predictions = self.gop_head(features, gammas)
        return self.gop_loss(predictions, gop_targets)

    def get_representations(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        layer: int = -1,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """
        Get representations for downstream tasks.
        
        Args:
            waveform: Input audio [B, T]
            lengths: Audio lengths [B]
            layer: Which layer to extract (-1 = last, -2 = second-to-last, etc.)
            pooling: How to pool frames ("mean", "first", "last", "none")
            
        Returns:
            representations: [B, D] if pooling, [B, T', D] if pooling="none"
        """
        # Encode
        if layer == -1:
            features = self.encode(waveform, lengths)
        else:
            _, all_layers = self.encode(waveform, lengths, return_all_layers=True)
            features = all_layers[layer]
        
        # Pool if requested
        if pooling == "none":
            return features
        
        # Get feature lengths
        _, feature_lengths = self.frontend(waveform, lengths)
        
        if pooling == "mean":
            if feature_lengths is not None:
                mask = torch.arange(features.shape[1], device=features.device)
                mask = mask.unsqueeze(0) < feature_lengths.unsqueeze(1)
                features = features * mask.unsqueeze(-1)
                pooled = features.sum(dim=1) / feature_lengths.unsqueeze(-1).float()
            else:
                pooled = features.mean(dim=1)
        
        elif pooling == "first":
            pooled = features[:, 0, :]
        
        elif pooling == "last":
            if feature_lengths is not None:
                pooled = features[torch.arange(features.shape[0]), feature_lengths - 1]
            else:
                pooled = features[:, -1, :]
        
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        return pooled


def create_ssl_model(
    preset: str = "base",
    **kwargs,
) -> ClinicalSpeechSSL:
    """
    Create SSL model from preset configuration.
    
    Presets:
        - "tiny": Minimal model for testing
        - "small": Small model for experimentation
        - "base": Standard model
        - "large": Large model for best performance
    
    Args:
        preset: Configuration preset name
        **kwargs: Override any config parameter
        
    Returns:
        Configured ClinicalSpeechSSL model
    """
    presets = {
        "tiny": {
            "frontend_type": "cnn_small",
            "encoder_type": "transformer_small",
            "embed_dim": 128,
        },
        "small": {
            "frontend_type": "cnn_small",
            "encoder_type": "conformer_small",
            "embed_dim": 256,
        },
        "base": {
            "frontend_type": "cnn_base",
            "encoder_type": "conformer_medium",
            "embed_dim": 256,
        },
        "large": {
            "frontend_type": "cnn_large",
            "encoder_type": "conformer_large",
            "embed_dim": 512,
        },
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config_dict = presets[preset].copy()
    config_dict.update(kwargs)
    
    config = ClinicalSpeechSSLConfig(**config_dict)
    return ClinicalSpeechSSL(config)
