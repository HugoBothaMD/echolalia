"""
WavLM frontend for clinical speech SSL.

Wraps a pretrained WavLM model as a drop-in replacement for the
waveform CNN frontend, providing the same interface:
    forward(waveform, lengths) -> (features, output_lengths)

Supports freezing for efficient training and staged unfreezing
for domain adaptation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig


class WavLMFrontend(nn.Module):
    """
    Pretrained WavLM as a frontend feature extractor.

    Replaces the waveform CNN frontend with WavLM's convolutional
    feature extractor + transformer layers. Produces frame-level
    features at the same 320x downsampling rate (20ms at 16kHz).

    Args:
        model_name: HuggingFace model name or local path
        output_dim: Project WavLM hidden dim to this size (must match encoder embed_dim)
        freeze: If True, freeze all WavLM parameters on init
        output_layer: Which transformer layer's output to use (-1 = last)
        dropout: Dropout on the projection layer
    """

    # WavLM uses the same 320x downsampling as wav2vec 2.0
    total_stride = 320

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        output_dim: int = 256,
        freeze: bool = True,
        output_layer: int = -1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.output_layer = output_layer

        # Load pretrained WavLM
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.hidden_size = self.wavlm.config.hidden_size  # 768 (base) or 1024 (large)
        self.num_layers = self.wavlm.config.num_hidden_layers

        # Projection from WavLM hidden dim to embed_dim
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all WavLM parameters."""
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def unfreeze(self, num_layers: Optional[int] = None):
        """Unfreeze WavLM parameters.

        Args:
            num_layers: If provided, only unfreeze the top N transformer layers.
                If None, unfreeze everything.
        """
        if num_layers is None:
            for param in self.wavlm.parameters():
                param.requires_grad = True
        else:
            # Always unfreeze layer norm
            if hasattr(self.wavlm, 'encoder') and hasattr(self.wavlm.encoder, 'layers'):
                layers = self.wavlm.encoder.layers
                total = len(layers)
                for i, layer in enumerate(layers):
                    if i >= total - num_layers:
                        for param in layer.parameters():
                            param.requires_grad = True

    def get_output_length(self, input_length: int) -> int:
        """Compute output sequence length given input waveform length."""
        # WavLM's conv feature extractor uses the same math as wav2vec 2.0
        # 7 conv layers with specific kernel/stride configs, but the net
        # result is ~320x downsampling. Use the model's own method if available.
        if hasattr(self.wavlm, '_get_feat_extract_output_lengths'):
            return self.wavlm._get_feat_extract_output_lengths(input_length).item()
        return input_length // self.total_stride

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process raw waveform through WavLM.

        Args:
            waveform: Raw audio [B, T]
            lengths: Optional original lengths [B]

        Returns:
            features: Frame-level features [B, T', output_dim]
            output_lengths: Adjusted lengths [B] if input lengths provided
        """
        # Build attention mask from lengths
        attention_mask = None
        if lengths is not None:
            max_len = waveform.shape[1]
            attention_mask = torch.arange(max_len, device=waveform.device)
            attention_mask = (attention_mask.unsqueeze(0) < lengths.unsqueeze(1)).long()

        # Run WavLM
        outputs = self.wavlm(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Select the desired layer output
        if self.output_layer == -1:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[self.output_layer]

        # Project to embed_dim
        features = self.projection(hidden_states)

        # Compute output lengths
        output_lengths = None
        if lengths is not None:
            output_lengths = self._compute_output_lengths(lengths)

        return features, output_lengths

    def _compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output lengths after WavLM's conv feature extractor."""
        if hasattr(self.wavlm, '_get_feat_extract_output_lengths'):
            return self.wavlm._get_feat_extract_output_lengths(input_lengths)
        return input_lengths // self.total_stride
