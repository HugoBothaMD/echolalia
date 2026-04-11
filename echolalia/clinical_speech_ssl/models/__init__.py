"""Model components for clinical speech SSL."""

from clinical_speech_ssl.models.ssl_model import (
    ClinicalSpeechSSL,
    ClinicalSpeechSSLConfig,
    create_ssl_model,
)
from clinical_speech_ssl.models.frontends import (
    WaveformCNNFrontend,
    WaveformCNNFrontendSmall,
    WaveformCNNFrontendLarge,
    SpectrogramPatchFrontend,
    TallNarrowPatcher,
    ViTStylePatcher,
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
    MaskedPredictionModule,
    ContrastiveModule,
)

__all__ = [
    "ClinicalSpeechSSL",
    "ClinicalSpeechSSLConfig",
    "create_ssl_model",
    "WaveformCNNFrontend",
    "WaveformCNNFrontendSmall",
    "WaveformCNNFrontendLarge",
    "SpectrogramPatchFrontend",
    "TallNarrowPatcher",
    "ViTStylePatcher",
    "TransformerEncoder",
    "TransformerEncoderSmall",
    "TransformerEncoderBase",
    "ConformerEncoder",
    "ConformerEncoderSmall",
    "ConformerEncoderMedium",
    "AugmentationPredictionHead",
    "MaskedPredictionModule",
    "ContrastiveModule",
]
