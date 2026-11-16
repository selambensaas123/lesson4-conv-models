"""Verify custom layer derivatives, parameter matching and training output."""
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from homework.models.cnn_models import CNN, Residual
from homework.models.custom_layers import GatedConv, ChannelAttention, MixedPool, SmoothActivation
from homework.utils.training_utils import epoch


class CNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_kernel_budgets_and_shapes(self):
        for widths, kernels in [((11, 64), (3, 3)), ((11, 24), (5, 5)),
                                ((11, 12), (7, 7)), ((33, 23), (1, 3))]:
            model = CNN(1, widths, kernels)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 7160)
            self.assertEqual(model(torch.ones(2, 1, 28, 28)).shape, (2, 10))

    def test_custom_input_derivatives(self):
        for layer in (GatedConv(1, 2), ChannelAttention(1), SmoothActivation(), MixedPool()):
            x = torch.tensor([[[[.2, .5], [1.4, -.8]]]], dtype=torch.double, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(layer.double(), (x,)))

    def test_custom_outputs_and_parameter_gradients(self):
        x = torch.tensor([[[[0., 2.], [4., 6.]]]], requires_grad=True)
        gated = GatedConv(1, 2)
        torch.testing.assert_close(gated(x), gated.conv(x))
        torch.testing.assert_close(SmoothActivation()(x), nn.SiLU()(x))
        torch.testing.assert_close(MixedPool()(x), torch.tensor([[[[4.5]]]]))
        for layer in (gated, ChannelAttention(1), MixedPool()):
            layer(x).sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in layer.parameters()))

    def test_residual_preserves_shape_and_backpropagates(self):
        for kind in ('basic', 'bottleneck', 'wide'):
            x = torch.randn(2, 8, 7, 7, requires_grad=True)
            y = Residual(8, kind)(x)
            self.assertEqual(x.shape, y.shape)
            y.square().mean().backward()
            self.assertGreater(x.grad.abs().sum().item(), 0.)

    def test_confusion_counts_and_gradients(self):
        loader = DataLoader(TensorDataset(torch.rand(5, 1, 6, 6), torch.arange(5)),
                            batch_size=3)
        model = CNN(1)
        row = epoch(model, loader, torch.optim.SGD(model.parameters(), lr=.01))
        matrix = torch.tensor(row['confusion'])
        self.assertEqual(matrix.sum().item(), 5)
        self.assertEqual(matrix.trace().item() / 5, row['accuracy'])
        self.assertTrue(row['gradients'])


if __name__ == '__main__':
    unittest.main()
