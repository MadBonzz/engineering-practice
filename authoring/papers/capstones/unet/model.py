import torch
from torch import nn


def convolutions(input_channels, out):
    return nn.Sequential(
        nn.Conv2d(input_channels, out, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out, out, 3, padding=1), nn.ReLU(),
    )


class Up(nn.Module):
    def __init__(self, deep, shallow):
        super().__init__()
        self.up = nn.ConvTranspose2d(deep, shallow, 2, stride=2)
        self.fuse = convolutions(2 * shallow, shallow)

    def forward(self, deep, skip):
        """Upsample deep [B,2C,H,W] to [B,C,2H,2W]. Concatenate skip first,
        upsampled map second along channels, then apply fuse to return [B,C,2H,2W].
        The encoder uses same-size padding and images divisible by four.
        """
        raise NotImplementedError("Complete this method using its contract.")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.high = convolutions(1, 8)
        self.mid = convolutions(8, 16)
        self.bridge = convolutions(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.decode_mid = Up(32, 16)
        self.decode_high = Up(16, 8)
        self.logits = nn.Conv2d(8, 1, 1)

    def forward(self, image):
        """Encode at full, half and quarter resolution, pooling between levels.
        Decode the bridge with the half-resolution skip, then decode that output
        with the full-resolution skip. Return one raw logit per input pixel.
        """
        raise NotImplementedError("Complete this method using its contract.")
