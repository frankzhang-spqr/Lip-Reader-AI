"""Conformer visual-speech model: Conv3D ResNet frontend + Conformer encoder + CTC head.

The frontend/encoder reuse the vendored espnet stack from autoavsr/ so a future
fine-tune from the pretrained AutoAVSR checkpoint is possible.
"""

import os
import sys

import torch
import torch.nn as nn

_AUTOAVSR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autoavsr")
if _AUTOAVSR not in sys.path:
    sys.path.insert(0, _AUTOAVSR)

from espnet.nets.pytorch_backend.transformer.encoder import Encoder  # noqa: E402


class VideoConformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        blank: int = 0,
        adim: int = 256,
        aheads: int = 4,
        linear_units: int = 2048,
        enc_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blank = blank
        self.vocab_size = vocab_size
        self.encoder = Encoder(
            idim=1,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=linear_units,
            num_blocks=enc_layers,
            dropout_rate=dropout,
            positional_dropout_rate=dropout,
            attention_dropout_rate=0.1,
            input_layer="conv3d",
            encoder_attn_layer_type="rel_mha",
            positionwise_layer_type="conv1d",
            positionwise_conv_kernel_size=3,
            macaron_style=True,
            use_cnn_module=True,
            cnn_module_kernel=31,
            normalize_before=True,
            relu_type="swish",
        )
        self.ctc_lin = nn.Linear(adim, vocab_size)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video clips to CTC logits.

        Args:
            video: (B, 1, T, 96, 96) float tensor.

        Returns:
            (B, T, vocab) logits.
        """
        xs, _ = self.encoder(video, None)
        return self.ctc_lin(xs)


def greedy_decode(logits: torch.Tensor, token_list: list[str]) -> list[str]:
    """CTC greedy decoding with blank collapsing.

    Args:
        logits: (B, T, vocab).
        token_list: index -> subword string (index 0 is blank, last is <eos>).

    Returns:
        One decoded string per batch item.
    """
    blank = 0
    eos = len(token_list) - 1
    preds = logits.argmax(-1)
    out = []
    for b in range(logits.size(0)):
        seq = []
        prev = None
        for t in range(logits.size(1)):
            idx = int(preds[b, t].item())
            if idx == blank or idx == eos:
                prev = None
            elif idx != prev:
                seq.append(idx)
                prev = idx
        out.append("".join(token_list[i] for i in seq))
    return out
