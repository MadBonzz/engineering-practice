import unittest
import torch
from model import Model, Up


class UNetTests(unittest.TestCase):
    def test_fuse_receives_the_correct_skip(self):
        torch.manual_seed(13)
        block = Up(8, 4)
        deep = torch.randn(1, 8, 4, 6)
        skip = torch.randn(1, 4, 8, 12)
        torch.testing.assert_close(block(deep, skip), block.fuse(torch.cat([skip, block.up(deep)], 1)))

    def test_resolution_and_gradient_flow(self):
        torch.manual_seed(23)
        model = Model()
        image = torch.randn(2, 1, 16, 20)
        logits = model(image)
        self.assertEqual(logits.shape, image.shape)
        logits.square().mean().backward()
        for parameter in [model.high[0].weight, model.bridge[0].weight, model.decode_high.up.weight]:
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
