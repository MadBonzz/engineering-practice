import torch


def batch(generator, size=12, length=8):
    start = torch.randint(13, (size, 1), generator=generator)
    step = torch.randint(1, 4, (size, 1), generator=generator)
    sequence = (start + step * torch.arange(length + 1)) % 13
    return sequence[:, :-1], sequence[:, 1:]
