"""CNN epoch encoder + transformer over a sequence of neighbouring epochs.

The 1D ResNet scores each 30 s epoch on its own, which throws away the fact
that sleep has structure in time: N1 is a transition stage, N3 arrives in long
blocks, REM follows N2. This keeps a convolutional encoder for the waveform
features and adds self-attention across a window of consecutive epochs, so the
label for one epoch can draw on its neighbours.

Sequence-to-sequence: a window of L epochs in, L predictions out.
"""
import math

import torch
import torch.nn as nn

from models.sleep_resnet import ResidualBlock1D


class EpochEncoder(nn.Module):
    """Single 30 s epoch -> one feature vector."""

    def __init__(self, in_channels=1, base_width=24, blocks_per_stage=(2, 2, 2, 2),
                 dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=49, stride=4,
                      padding=24, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        stages, ch = [], base_width
        for si, n in enumerate(blocks_per_stage):
            out = base_width * (2 ** si)
            for bi in range(n):
                stride = 2 if (bi == 0 and si > 0) else 1
                stages.append(ResidualBlock1D(ch, out, stride=stride, dropout=dropout))
                ch = out
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = ch

    def forward(self, x):                      # (B, 1, T) -> (B, C)
        x = self.stages(self.stem(x))
        return self.pool(x).flatten(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                      # (B, L, D)
        return x + self.pe[:, : x.size(1)]


class SleepTransformer(nn.Module):
    def __init__(self, num_classes=5, base_width=24, d_model=192, nhead=6,
                 num_layers=4, dim_feedforward=768, dropout=0.1, seq_len=21):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = EpochEncoder(base_width=base_width, dropout=dropout)

        self.proj = (nn.Identity() if self.encoder.out_dim == d_model
                     else nn.Linear(self.encoder.out_dim, d_model))
        self.pos = PositionalEncoding(d_model, max_len=max(seq_len * 4, 128))
        self.in_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 norm=nn.LayerNorm(d_model))
        self.head = nn.Sequential(nn.Dropout(dropout * 2), nn.Linear(d_model, num_classes))

    def forward(self, x):
        # (B, L, T) or (B, L, 1, T) -> encode every epoch, then attend across L
        if x.dim() == 4:
            x = x.squeeze(2)
        b, l, t = x.shape
        feats = self.encoder(x.reshape(b * l, 1, t)).reshape(b, l, -1)
        h = self.in_norm(self.proj(feats))
        h = self.transformer(self.pos(h))
        return self.head(h)                    # (B, L, num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = SleepTransformer()
    x = torch.randn(2, 21, 3000)
    out = m(x)
    print("output:", out.shape)
    print(f"total params:   {count_parameters(m):,}")
    print(f"  encoder:      {count_parameters(m.encoder):,}")
    print(f"  transformer:  {count_parameters(m.transformer):,}")
