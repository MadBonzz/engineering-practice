import torch


def batch(generator, size=8, length=5):
    source = torch.randint(2, 12, (size, length), generator=generator)
    translated = 2 + (source.flip(1) - 2 + 1) % 10
    bos = torch.ones(size, 1, dtype=torch.long)
    eos = torch.zeros(size, 1, dtype=torch.long)
    return source, torch.cat([bos, translated], 1), torch.cat([translated, eos], 1)
