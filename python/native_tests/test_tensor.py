"""Whole-tensor lowering contracts independent of the upstream model."""

import unittest
import numpy as np


class TensorTests(unittest.TestCase):
    def setUp(self):
        from fhenomenon.ml.tensor import TensorEngine, ReferenceVectorBackend

        self.backend = ReferenceVectorBackend()
        self.engine = TensorEngine(self.backend, slots=8)

    def test_private_matmul_and_layout_cross_chunks(self):
        e = self.engine
        a = np.arange(12, dtype=float).reshape(3, 4) / 10
        b = np.arange(20, dtype=float).reshape(4, 5) / 10
        out = e.matmul(e.encrypt(a), e.encrypt(b))
        np.testing.assert_allclose(e.decrypt(out), a @ b, atol=1e-12)
        out = e.sum(e.permute(e.encrypt(a), (1, 0)), (1,), False)
        np.testing.assert_allclose(e.decrypt(out), a.sum(axis=0), atol=1e-12)
        self.assertEqual(self.backend.stats["intermediate_decryptions"], 0)

    def test_broadcast_slice_and_scatter_backward(self):
        e = self.engine
        x = e.encrypt(np.arange(24.0).reshape(2, 3, 4))
        y = e.mul(x, np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(
            e.decrypt(y), np.arange(24.0).reshape(2, 3, 4) * [1, 2, 3, 4]
        )
        part = e.slice(x, 1, 1, 3, 1)
        z = e.slice_backward(part, (2, 3, 4), 1, 1, 3, 1)
        expected = np.arange(24.0).reshape(2, 3, 4)
        expected[:, 0] = 0
        np.testing.assert_allclose(e.decrypt(z), expected, atol=1e-12)

    def test_convolution_and_activation_weight_gradients(self):
        import torch

        e = self.engine
        x = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4).requires_grad_()
        w = torch.tensor(
            [[[[0.1, 0.2], [0.3, 0.4]]]], requires_grad=True, dtype=torch.float64
        )
        y = torch.nn.functional.conv2d(x, w, stride=2)
        y.sum().backward()
        xx, ww = e.encrypt(x.detach().numpy()), e.encrypt(w.detach().numpy())
        got = e.conv2d(xx, ww, None, (2, 2), (0, 0), (1, 1), 1)
        np.testing.assert_allclose(e.decrypt(got), y.detach().numpy())
        dx, dw, db = e.conv2d_backward(
            np.ones((1, 1, 2, 2)), xx, ww, (2, 2), (0, 0), (1, 1), 1, (True, True, True)
        )
        np.testing.assert_allclose(e.decrypt(dx), x.grad.numpy())
        np.testing.assert_allclose(e.decrypt(dw), w.grad.numpy())
        np.testing.assert_allclose(e.decrypt(db), [4.0])

    def test_stack_negative_axis_matches_output_axis(self):
        e = self.engine
        a = e.encrypt(np.arange(6.0).reshape(2, 3))
        got = e.stack([a, e.add(a, 10)], -1)
        np.testing.assert_allclose(
            e.decrypt(got),
            np.stack(
                [np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3) + 10],
                axis=-1,
            ),
        )

    def test_reuses_refresh_of_same_immutable_source(self):
        e = self.engine
        x = e.encrypt([0.1, 0.2, 0.3])
        for c in x.chunks:
            c.level = 0
        e.mul(x, 2.0)
        e.mul(x, 3.0)
        self.assertEqual(e.backend.stats["mod_raise_calls"], 1)

    def test_no_implicit_array_conversion(self):
        x = self.engine.encrypt([1.0, 2.0])
        with self.assertRaises(TypeError):
            np.asarray(x)


if __name__ == "__main__":
    unittest.main()


class MagnitudeContractTests(unittest.TestCase):
    def test_public_bounds_reach_refresh_and_minimum_egress(self):
        from fhenomenon.ml.tensor import TensorEngine, ReferenceVectorBackend

        class Recording(ReferenceVectorBackend):
            def mod_raise(self, x, target_level, public_bound=1.0):
                self.refresh_bound = public_bound
                return super().mod_raise(x, target_level, public_bound=public_bound)

            def decrypt(self, x, public_bound=1.0):
                self.egress_bound = public_bound
                return super().decrypt(x, public_bound=public_bound)

        b = Recording()
        e = TensorEngine(b, slots=8)
        x = e.encrypt([4.0, -3.0], public_bound=8.0)
        for c in x.chunks:
            c.level = 0
        y = e.add(e.mul(x, 2.0), 1.0)
        np.testing.assert_allclose(e.decrypt(y), [9.0, -5.0])
        self.assertEqual(b.refresh_bound, 8.0)
        self.assertEqual(b.egress_bound, 17.0)
        self.assertEqual(e.slice(y, 0, 0, 1).public_bound, 17.0)

    def test_client_rejects_declared_input_range_violation(self):
        from fhenomenon.ml.tensor import TensorEngine, ReferenceVectorBackend

        e = TensorEngine(ReferenceVectorBackend(), slots=8)
        with self.assertRaisesRegex(ValueError, "public bound"):
            e.encrypt([2.0], public_bound=1.0)
        with self.assertRaises(ValueError):
            e.encrypt([float("nan")], public_bound=1.0)
