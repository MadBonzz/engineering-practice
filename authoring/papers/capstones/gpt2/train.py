import argparse
import torch
from torch.nn import functional as F
from data import batch
from model import Model


def objective(model, generator):
    ids, targets = batch(generator)
    logits = model(ids)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 1)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, generator)
        loss.backward()
        optimizer.step()
        if step <= 2 or step % 20 == 0 or step == args.steps:
            print(f"step {step:03d}  training loss {loss.item():.5f}")
    model.eval()
    with torch.no_grad():
        held_out = objective(model, torch.Generator().manual_seed(54))
        print(f"held-out loss {held_out.item():.5f}")
        ids = torch.tensor([[2, 4, 6]])
        for _ in range(4):
            next_id = model(ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        print("continuation from [2, 4, 6]", ids[0, 3:].tolist())


if __name__ == "__main__":
    main()
