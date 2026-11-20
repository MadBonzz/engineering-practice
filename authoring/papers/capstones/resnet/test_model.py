import unittest
import torch
from model import Model, Residual


class ResidualTests(unittest.TestCase):
    def test_zero_transform_preserves_positive_input(self):
        block = Residual(3, 3).eval()
        for parameter in block.main.parameters():
            parameter.data.zero_()
        x = torch.linspace(-1, 1, 75).view(1, 3, 5, 5)
        torch.testing.assert_close(block(x), x.relu())

    def test_projection_and_readout(self):
        torch.manual_seed(12)
        block = Residual(2, 4, stride=2).eval()
        x = torch.randn(2, 2, 9, 11)
        torch.testing.assert_close(block(x), (block.main(x) + block.shortcut(x)).relu())
        model = Model().eval()
        for h, w in [(9, 11), (16, 16)]:
            image = torch.randn(2, 1, h, w)
            logits = model(image)
            self.assertEqual(logits.shape, (2, 2))
            logits.sum().backward()
            self.assertGreater(model.stage[1].shortcut[0].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
