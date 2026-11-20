import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(channels_out), nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
        )
        self.shortcut = nn.Identity() if stride == 1 and channels_in == channels_out else nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, stride, bias=False), nn.BatchNorm2d(channels_out)
        )

    def forward(self, x):
        """Combine the transform and shortcut of the same [B,C,H,W] input.
        The supplied shortcut changes resolution or channel count if needed.
        Return ReLU of their elementwise sum, after both paths have finished.
        """

        return torch.relu(self.main(x) + self.shortcut(x))



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU())
        self.stage = nn.Sequential(Residual(8, 8), Residual(8, 16, stride=2), Residual(16, 16))
        self.classifier = nn.Linear(16, 2)

    def forward(self, images):
        """Map [B,1,H,W] to [B,2] logits for arbitrary H,W >= 4.
        Apply stem, residual stage, global spatial mean, and classifier.
        """

        maps = self.stage(self.stem(images))
        return self.classifier(maps.mean(dim=(2, 3)))
