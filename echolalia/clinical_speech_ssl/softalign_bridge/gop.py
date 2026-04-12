"""GOP target computation using soft_align."""

from typing import List, Optional

import torch

from soft_align import (
    AudioBackbone,
    BackboneOutput,
    compute_gop,
    GOPMode,
    GOPResult,
)


def compute_gop_targets(
    audio: torch.Tensor,
    transcript: str,
    backbone: AudioBackbone,
    sample_rate: int = 16000,
    mode: GOPMode = GOPMode.SOFT,
) -> GOPResult:
    """Compute per-phoneme GOP scores as SSL supervision targets.

    Args:
        audio: Waveform tensor (samples,) or (1, samples)
        transcript: Target transcript
        backbone: soft_align AudioBackbone
        sample_rate: Audio sample rate
        mode: GOP computation mode (SOFT recommended)

    Returns:
        GOPResult with .scores [num_phonemes] and .phoneme_labels
    """
    output = backbone.process_audio(audio, sample_rate)
    target_ids = backbone.tokenize_target(transcript)

    return compute_gop(
        output.log_probs,
        target_ids,
        mode=mode,
        blank_idx=backbone.get_blank_index(),
        phoneme_labels=backbone.tokens_to_text(target_ids),
    )
