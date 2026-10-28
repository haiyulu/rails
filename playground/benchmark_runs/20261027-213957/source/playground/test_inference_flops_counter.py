"""Closed-form checks independent of the four model integration cases."""

import unittest

import torch
import torch.nn.functional as F

from inference_flops_counter import ArithmeticLedger


class CounterTests(unittest.TestCase):
    def count(self, function):
        with torch.inference_mode(), ArithmeticLedger() as ledger:
            function()
        return ledger.totals()

    def test_linear_with_bias(self):
        x, w, b = torch.ones(4, 3), torch.ones(5, 3), torch.ones(5)
        result = self.count(lambda: F.linear(x, w, b))
        self.assertEqual(result["matrix_flops"], 120)
        self.assertEqual(result["other_arithmetic_flops"], 20)

    def test_pointwise_convolution(self):
        x, w, b = torch.ones(2, 3, 7), torch.ones(5, 3, 1), torch.ones(5)
        result = self.count(lambda: F.conv1d(x, w, b))
        self.assertEqual(result["matrix_flops"], 420)
        self.assertEqual(result["other_arithmetic_flops"], 70)

    def test_normalizations(self):
        x = torch.ones(2, 3)
        layer = self.count(lambda: F.layer_norm(x, (3,)))
        self.assertEqual(layer["arithmetic_flops"], 32)
        self.assertEqual(layer["special_functions"], {"rsqrt": 2})
        def normalized():
            return x / torch.clamp(torch.linalg.norm(x, ord=2, dim=-1, keepdim=True), min=1e-6)
        l2 = self.count(normalized)
        self.assertEqual(l2["arithmetic_flops"], 16)
        self.assertEqual(l2["special_functions"], {"sqrt": 2})

    def test_softmax_and_inference_renormalization(self):
        x = torch.ones(2, 3)
        def normalize():
            p = F.softmax(x, dim=-1)
            return p / torch.clamp(p.sum(-1, keepdim=True), min=1e-6)
        result = self.count(normalize)
        self.assertEqual(result["arithmetic_flops"], 26)
        self.assertEqual(result["special_functions"], {"exp": 6})

    def test_special_functions_not_priced_as_hardware_flops(self):
        x = torch.ones(2, 3)
        result = self.count(lambda: F.silu(x))
        self.assertEqual(result["arithmetic_flops"], 6)
        self.assertEqual(result["special_functions"], {"sigmoid": 6})

    def test_data_movement_and_inference_dropout_are_free(self):
        x = torch.ones(2, 3)
        result = self.count(lambda: F.dropout(x, .2, False).reshape(3, 2).T.clone())
        self.assertEqual(result["arithmetic_flops"], 0)
        self.assertEqual(result["special_functions"], {})

    def test_unknown_floating_op_fails(self):
        x = torch.ones(2, 3)
        with self.assertRaisesRegex(RuntimeError, "Unclassified floating operation"):
            self.count(lambda: torch.sin(x))


if __name__ == "__main__":
    unittest.main()
