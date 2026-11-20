import unittest
import torch
from data import batch
from model import Model


class TranslationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.net = Model().eval()

    def test_teacher_forcing(self):
        source, prefix, labels = batch(torch.Generator().manual_seed(8))
        self.assertTrue(torch.equal(prefix[:, 1:], labels[:, :-1]))
        self.assertTrue(torch.all(labels[:, -1] == 0))
        self.assertEqual(self.net(source, prefix).shape, (*labels.shape, 12))

    def test_causal_target_and_visible_source(self):
        source = torch.tensor([[2, 4, 6, 8]])
        a = self.net(source, torch.tensor([[1, 3, 5, 7]]))
        b = self.net(source, torch.tensor([[1, 3, 9, 2]]))
        c = self.net(torch.tensor([[2, 4, 6, 11]]), torch.tensor([[1, 3, 5, 7]]))
        torch.testing.assert_close(a[:, :2], b[:, :2])
        self.assertGreater((a[:, :1] - c[:, :1]).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
