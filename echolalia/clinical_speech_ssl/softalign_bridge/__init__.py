"""
Bridge module for soft_align integration.

Provides wrappers around soft_align's public API for:
- Gamma matrix computation (PhonemeClassifierBackbone → CTCAligner)
- GOP target computation
- SoftAlign measure extraction
- Quality tier assessment via UtteranceAnalysisPipeline
"""

from soft_align import (
    CTCAligner,
    AlignmentResult,
    PhonemeCTCBackbone,
    PhonemeClassifierBackbone,
    load_backbone,
    compute_gop,
    GOPMode,
    GOPResult,
    QualityTier,
)

from clinical_speech_ssl.softalign_bridge.alignment import (
    get_alignment_backbone,
    compute_gamma,
)
from clinical_speech_ssl.softalign_bridge.gop import compute_gop_targets
from clinical_speech_ssl.softalign_bridge.measures import compute_measure_vector
from clinical_speech_ssl.softalign_bridge.quality import compute_quality_tier

__all__ = [
    "get_alignment_backbone",
    "compute_gamma",
    "compute_gop_targets",
    "compute_measure_vector",
    "compute_quality_tier",
]
