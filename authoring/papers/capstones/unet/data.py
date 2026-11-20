import torch


def batch(generator, size=8, side=16):
    masks = torch.zeros(size, 1, side, side)
    offsets = torch.randint(1, side - 5, (size, 2), generator=generator)
    for i, (y, x) in enumerate(offsets.tolist()):
        masks[i, 0, y:y + 4, x:x + 4] = 1
    images = 0.7 * masks + 0.25 * torch.randn(masks.shape, generator=generator)
    return images, masks
