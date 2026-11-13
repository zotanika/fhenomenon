"""Public softmax denominator intervals; no native ciphertext range probing."""

import math
import unittest

import numpy as np
import torch

from examples.openvla_training.interpreter import GraphInterpreter, _MaskedAttention
from fhenomenon.ml.tensor import ReferenceVectorBackend, TensorEngine


class IntervalEngine(TensorEngine):
    def __init__(self):
        super().__init__(ReferenceVectorBackend(), slots=32)
        self.intervals = []

    def nonlinear(self, name, args):
        if name in ("reciprocal", "log"):
            self.intervals.append((name, args[1:]))
        return super().nonlinear(name, args)


class SoftmaxBoundTests(unittest.TestCase):
    def test_private_scores_use_declared_bound_and_public_width(self):
        engine = IntervalEngine()
        interpreter = GraphInterpreter(engine)
        for source in (
            torch.zeros(2, 3),
            torch.tensor([[-0.5, 0.0, 0.5], [0.25, -0.25, 0.0]]),
        ):
            private = engine.encrypt(source, public_bound=0.5)
            result = interpreter._softmax(private, -1)
            self.assertEqual(engine.intervals[-1][0], "reciprocal")
            low, high = engine.intervals[-1][1]
            self.assertAlmostEqual(low, 3 * math.exp(-0.5) * (1 - 1e-7))
            self.assertAlmostEqual(high, 3 * math.exp(0.5) * (1 + 1e-7))
            np.testing.assert_allclose(
                engine.decrypt(result),
                source.double().softmax(-1).numpy(),
                atol=1e-8,
                rtol=1e-7,
            )
        self.assertEqual(engine.intervals[0], engine.intervals[1])

    def test_mask_counts_are_broadcast_public_metadata(self):
        engine = IntervalEngine()
        scores = engine.encrypt(torch.zeros(1, 2, 3, 3), public_bound=0.5)
        keep = torch.tril(torch.ones(3, 3)).reshape(1, 1, 3, 3)
        result = GraphInterpreter(engine)._softmax(_MaskedAttention(scores, keep), -1)
        low, high = engine.intervals[-1][1]
        self.assertAlmostEqual(low, math.exp(-0.5) * (1 - 1e-7))
        self.assertAlmostEqual(high, 3 * math.exp(0.5) * (1 + 1e-7))
        expected = torch.tensor(
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]]
        )
        np.testing.assert_allclose(
            engine.decrypt(result), expected.expand(1, 2, 3, 3).numpy(), atol=1e-7
        )

    def test_fully_masked_row_is_rejected_from_public_metadata(self):
        engine = IntervalEngine()
        scores = engine.encrypt(torch.zeros(2, 3), public_bound=1.0)
        keep = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "unmasked"):
            GraphInterpreter(engine)._softmax(_MaskedAttention(scores, keep), -1)

    def test_log_softmax_passes_the_positive_denominator_interval(self):
        engine = IntervalEngine()
        source = torch.tensor([[0.125, -0.125, 0.0, 0.0]])
        private = engine.encrypt(source, public_bound=0.125)
        result = GraphInterpreter(engine)._softmax(private, -1, logarithmic=True)
        self.assertEqual(engine.intervals[-1][0], "log")
        low, high = engine.intervals[-1][1]
        self.assertAlmostEqual(low, 4 * math.exp(-0.125) * (1 - 1e-7))
        self.assertAlmostEqual(high, 4 * math.exp(0.125) * (1 + 1e-7))
        np.testing.assert_allclose(
            engine.decrypt(result), source.double().log_softmax(-1).numpy(), atol=2e-7
        )

    def test_wide_interval_retains_the_declared_global_domain_requirement(self):
        engine = IntervalEngine()
        private = engine.encrypt(torch.zeros(1, 64), public_bound=8.0)
        GraphInterpreter(engine)._softmax(private, -1)
        low, high = engine.intervals[-1][1]
        self.assertAlmostEqual(low, 64 * math.exp(-8) * (1 - 1e-7))
        self.assertEqual(high, 131072.0)


if __name__ == "__main__":
    unittest.main()
