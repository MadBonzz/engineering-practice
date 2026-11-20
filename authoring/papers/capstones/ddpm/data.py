import torch


def batch(generator, size=32):
    cluster = torch.randint(2, (size, 1), generator=generator)
    center = (2 * cluster - 1) * torch.tensor([1.0, -0.75])
    return center + 0.18 * torch.randn(size, 2, generator=generator)
