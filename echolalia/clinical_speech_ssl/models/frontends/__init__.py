"""Frontend modules for processing raw audio or spectrograms."""

from clinical_speech_ssl.models.frontends.waveform_cnn import (
    WaveformCNNFrontend,
    WaveformCNNFrontendSmall,
    WaveformCNNFrontendLarge,
    ConvBlock,
)
from clinical_speech_ssl.models.frontends.spectrogram_patcher import (
    MelSpectrogramTransform,
    TallNarrowPatcher,
    ViTStylePatcher,
    SpectrogramPatchFrontend,
)
from clinical_speech_ssl.models.frontends.wavlm_frontend import (
    WavLMFrontend,
)

__all__ = [
    "WaveformCNNFrontend",
    "WaveformCNNFrontendSmall",
    "WaveformCNNFrontendLarge",
    "ConvBlock",
    "MelSpectrogramTransform",
    "TallNarrowPatcher",
    "ViTStylePatcher",
    "SpectrogramPatchFrontend",
    "WavLMFrontend",
]
