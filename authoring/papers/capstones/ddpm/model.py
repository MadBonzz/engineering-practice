import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, steps=16):
        super().__init__()
        beta = torch.linspace(0.001, 0.36, steps)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0))
        self.time = nn.Embedding(steps, 12)
        self.denoiser = nn.Sequential(nn.Linear(14, 32), nn.SiLU(), nn.Linear(32, 32),
                                      nn.SiLU(), nn.Linear(32, 2))

    def forward(self, noisy, t):
        """Embed each timestep, concatenate [noisy, time_embedding] on features,
        and call the shared denoiser to predict a [B,2] noise vector.
        """
        raise NotImplementedError("Complete the time-conditioned predictor.")

    def add_noise(self, clean, t, epsilon):
        """Mix clean [B,2] and known unit Gaussian noise [B,2] at per-row integer t.
        Use sqrt(alpha_bar[t]) on clean and sqrt(1-alpha_bar[t]) on epsilon.
        Return [B,2]. Index zero already represents one forward noising step.
        """
        raise NotImplementedError("Complete this method using its contract.")

    def reverse(self, noisy, t, epsilon_hat, noise):
        """Compute one epsilon-parameterized reverse step. All point tensors are [B,2].
        Use mu=(x-beta_t*eps_hat/sqrt(1-abar_t))/sqrt(1-beta_t).
        Add noise scaled by sqrt(beta_t*(1-abar_prev)/(1-abar_t)).
        At t=0 use abar_prev=1, giving a deterministic final step.
        """
        raise NotImplementedError("Complete this method using its contract.")

    @torch.no_grad()
    def sample(self, count, generator):
        points = torch.randn(count, 2, generator=generator)
        for index in reversed(range(len(self.beta))):
            t = torch.full((count,), index, dtype=torch.long)
            noise = torch.randn(points.shape, generator=generator)
            points = self.reverse(points, t, self(points, t), noise)
        return points
