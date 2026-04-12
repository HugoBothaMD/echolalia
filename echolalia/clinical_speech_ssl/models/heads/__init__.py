"""SSL objective heads."""

from clinical_speech_ssl.models.heads.augmentation_head import (
    AugmentationPredictionHead,
    AugmentationPredictionLoss,
)
from clinical_speech_ssl.models.heads.reconstruction_head import (
    MaskGenerator,
    ReconstructionHead,
    MaskedReconstructionLoss,
    MaskedPredictionModule,
)
from clinical_speech_ssl.models.heads.contrastive_head import (
    ProjectionHead,
    ContrastiveHead,
    ContrastiveLoss,
    SimCLRLoss,
    ContrastiveModule,
)
from clinical_speech_ssl.models.heads.gop_head import (
    GOPPredictionHead,
    GOPPredictionLoss,
)

__all__ = [
    "AugmentationPredictionHead",
    "AugmentationPredictionLoss",
    "MaskGenerator",
    "ReconstructionHead",
    "MaskedReconstructionLoss",
    "MaskedPredictionModule",
    "ProjectionHead",
    "ContrastiveHead",
    "ContrastiveLoss",
    "SimCLRLoss",
    "ContrastiveModule",
    "GOPPredictionHead",
    "GOPPredictionLoss",
]
