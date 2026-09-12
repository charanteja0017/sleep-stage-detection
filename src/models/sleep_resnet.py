import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class SleepResNet1D(nn.Module):
    """1D ResNet for 30s EEG epoch -> 5-class sleep stage."""

    def __init__(self, in_channels=1, num_classes=5, base_width=34,
                 blocks_per_stage=(2, 2, 2, 2), dropout=0.1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=49, stride=4,
                      padding=24, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        stages = []
        channels = base_width
        for stage_idx, n_blocks in enumerate(blocks_per_stage):
            out_channels = base_width * (2 ** stage_idx)
            for block_idx in range(n_blocks):
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                stages.append(
                    ResidualBlock1D(channels, out_channels, stride=stride, dropout=dropout)
                )
                channels = out_channels
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout * 2),
            nn.Linear(channels, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SleepResNet1D(in_channels=1, num_classes=5)
    x = torch.randn(4, 1, 3000)
    print("output:", model(x).shape)
    print(f"params: {count_parameters(model):,}")
