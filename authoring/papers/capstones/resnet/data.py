import torch


def batch(generator, size=12, side=16):
    images = torch.randn(size, 1, side, side, generator=generator) * 0.15
    labels = torch.randint(2, (size,), generator=generator)
    starts = torch.randint(1, side - 2, (size,), generator=generator)
    for i, (label, start) in enumerate(zip(labels.tolist(), starts.tolist())):
        if label == 0:
            images[i, 0, start:start + 2, :] += 1
        else:
            images[i, 0, :, start:start + 2] += 1
    return images, labels
