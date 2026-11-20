import unittest
import torch
from model import Model


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        self.net = Model(steps=4)

    def test_distinct_timesteps_use_distinct_coefficients(self):
        clean, epsilon = torch.eye(2), torch.eye(2).flip(1)
        t = torch.tensor([0, 3])
        noisy = self.net.add_noise(clean, t, epsilon)
        for i in range(2):
            keep = torch.prod(1 - self.net.beta[:t[i] + 1])
            torch.testing.assert_close(noisy[i], keep.sqrt() * clean[i] + (1 - keep).sqrt() * epsilon[i])

    def test_reverse_noise_is_gated_at_final_step(self):
        x = torch.ones(2, 2)
        t = torch.tensor([0, 2])
        mu = self.net.reverse(x, t, torch.zeros_like(x), torch.zeros_like(x))
        expected = x / (1 - self.net.beta[t, None]).sqrt()
        torch.testing.assert_close(mu, expected)
        with_noise = self.net.reverse(x, t, torch.zeros_like(x), torch.ones_like(x))
        torch.testing.assert_close(with_noise[0], mu[0])
        expected_sigma = (self.net.beta[2] * (1 - self.net.alpha_bar[1]) / (1 - self.net.alpha_bar[2])).sqrt()
        torch.testing.assert_close(with_noise[1] - mu[1], expected_sigma.expand(2))

    def test_sampling_is_seeded(self):
        a = self.net.sample(5, torch.Generator().manual_seed(8))
        b = self.net.sample(5, torch.Generator().manual_seed(8))
        self.assertEqual(a.shape, (5, 2))
        torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
