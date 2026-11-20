import math
import unittest
import torch
from model import Block, CausalAttention, Model


class GPTTests(unittest.TestCase):
    def test_masked_attention_with_identity_projections(self):
        layer = CausalAttention(2, 1)
        with torch.no_grad():
            layer.qkv.weight.copy_(torch.eye(2).repeat(3, 1))
            layer.out.weight.copy_(torch.eye(2))
            layer.out.bias.zero_()
            layer.qkv.bias.zero_()
        x = torch.eye(2).unsqueeze(0)
        weight = 1 / (1 + math.exp(1 / math.sqrt(2)))
        torch.testing.assert_close(layer(x), torch.tensor([[[1., 0.], [weight, 1 - weight]]]))

    def test_prefix_invariance_and_pre_norm(self):
        torch.manual_seed(14)
        model = Model().eval()
        a = model(torch.tensor([[1, 2, 3, 4]]))
        b = model(torch.tensor([[1, 2, 5, 6]]))
        self.assertEqual(a.shape, (1, 4, 13))
        torch.testing.assert_close(a[:, :2], b[:, :2])
        block = Block(12).eval()
        x = torch.randn(2, 3, 12)
        y = x + block.attn(block.norm1(x))
        torch.testing.assert_close(block(x), y + block.mlp(block.norm2(y)))


if __name__ == "__main__":
    unittest.main()
