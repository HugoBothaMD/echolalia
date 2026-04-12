"""Quality tier assessment using soft_align's UtteranceAnalysisPipeline."""

from typing import Optional

import torch

from soft_align import (
    AudioBackbone,
    UtteranceAnalysisPipeline,
    UtteranceAnalysis,
    QualityTier,
)


# Map QualityTier enum to integer for storage
QUALITY_TIER_TO_INT = {
    QualityTier.HIGH: 3,
    QualityTier.MEDIUM: 2,
    QualityTier.LOW: 1,
    QualityTier.FAILED: 0,
}

INT_TO_QUALITY_TIER = {v: k for k, v in QUALITY_TIER_TO_INT.items()}

# Minimum integer quality tier thresholds
MIN_QUALITY_THRESHOLDS = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "failed": 0,
}


def compute_quality_tier(
    audio: torch.Tensor,
    transcript: str,
    backbone: AudioBackbone,
    sample_rate: int = 16000,
    patient_id: str = "unknown",
) -> int:
    """Compute quality tier as an integer for a single utterance.

    Args:
        audio: Waveform tensor
        transcript: Target transcript
        backbone: soft_align AudioBackbone
        sample_rate: Audio sample rate
        patient_id: Optional patient ID

    Returns:
        Integer quality tier (3=HIGH, 2=MEDIUM, 1=LOW, 0=FAILED)
    """
    pipeline = UtteranceAnalysisPipeline(backbone)
    result: UtteranceAnalysis = pipeline.analyze(
        audio, transcript,
        sample_rate=sample_rate,
        patient_id=patient_id,
    )
    return QUALITY_TIER_TO_INT.get(result.quality_tier, 0)
