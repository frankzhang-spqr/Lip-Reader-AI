"""LipNet model (Assael et al. 2016) adapted for 96x96 grayscale mouth crops.

Input : (B, 1, T, 96, 96)
Output: (B, T, vocab) logits, vocab = 26 letters + space + CTC blank = 28.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class LipNet(nn.Module):
    def __init__(self, in_channels: int = 1, vocab_size: int = 28, dropout_p: float = 0.5):
        super().__init__()
        # 96x96 -> 6x6 after the three stride-2/pooling blocks
        self.conv1 = nn.Conv3d(in_channels, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        gru_in = 96 * 6 * 6
        self.gru1 = nn.GRU(gru_in, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)
        self.fc = nn.Linear(512, vocab_size)

        self.dropout = nn.Dropout(dropout_p)
        self.dropout3d = nn.Dropout3d(dropout_p)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for conv in (self.conv1, self.conv2, self.conv3):
            init.kaiming_normal_(conv.weight, nonlinearity="relu")
            init.constant_(conv.bias, 0)
        init.kaiming_normal_(self.fc.weight, nonlinearity="sigmoid")
        init.constant_(self.fc.bias, 0)
        for gru in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(gru.weight_ih_l0[i : i + 256], -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(gru.weight_hh_l0[i : i + 256])
                init.constant_(gru.bias_ih_l0[i : i + 256], 0)
                init.uniform_(gru.weight_ih_l0_reverse[i : i + 256], -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(gru.weight_hh_l0_reverse[i : i + 256])
                init.constant_(gru.bias_ih_l0_reverse[i : i + 256], 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.dropout3d(x)
        x = self.pool3(x)

        x = x.permute(2, 0, 1, 3, 4).contiguous()  # (T, B, C, H, W)
        x = x.view(x.size(0), x.size(1), -1)  # (T, B, C*H*W)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x)

        x = self.fc(x)  # (T, B, vocab)
        return x.permute(1, 0, 2).contiguous()  # (B, T, vocab)


def greedy_decode(logits: torch.Tensor, idx2char: list[str]) -> list[str]:
    """Greedy CTC decode: take argmax, collapse repeats, drop blank."""
    probs = logits.softmax(-1)
    pred = probs.argmax(-1).cpu().numpy()
    out = []
    for p in pred:
        prev = None
        chars = []
        for t in p:
            if t != prev:
                chars.append(idx2char[t])
                prev = t
            else:
                prev = t
        out.append("".join(c for c in chars if c != "<blank>"))
    return out
