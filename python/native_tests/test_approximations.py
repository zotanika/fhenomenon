import unittest
import numpy as np
from fhenomenon.ml.tensor import TensorEngine, ReferenceVectorBackend


class ApproximationTests(unittest.TestCase):
    def test_exp_and_activation_polynomials_on_declared_domain(self):
        from fhenomenon.ml.approximations import evaluate

        e = TensorEngine(ReferenceVectorBackend(), slots=32)
        x = np.linspace(-4, 4, 17)
        xx = e.encrypt(x)
        np.testing.assert_allclose(
            e.decrypt(evaluate(e, "exp", (xx,))), np.exp(x), rtol=1e-7, atol=1e-7
        )
        import torch

        np.testing.assert_allclose(
            e.decrypt(evaluate(e, "gelu", (xx, "none"))),
            torch.nn.functional.gelu(torch.tensor(x)).numpy(),
            atol=3e-4,
        )

    def test_inverse_iterations_include_small_norm_variance(self):
        from fhenomenon.ml.approximations import evaluate

        e = TensorEngine(ReferenceVectorBackend(), slots=8)
        x = np.array([1e-6, 1e-4, 0.01, 1.0, 4.0, 16.0])
        np.testing.assert_allclose(
            e.decrypt(evaluate(e, "rsqrt", (e.encrypt(x),))), 1 / np.sqrt(x), rtol=1e-7
        )

    def test_log_range_reduction_and_reciprocal(self):
        from fhenomenon.ml.approximations import evaluate

        e = TensorEngine(ReferenceVectorBackend(), slots=8)
        x = np.array([1e-6, 0.01, 0.1, 1.0, 32.0, 1024.0, 131072.0])
        np.testing.assert_allclose(
            e.decrypt(evaluate(e, "log", (e.encrypt(x),))), np.log(x), atol=2e-7
        )
        np.testing.assert_allclose(
            e.decrypt(evaluate(e, "reciprocal", (e.encrypt(x),))), 1 / x, rtol=1e-7
        )

    def test_unused_inverse_lanes_remain_bounded(self):
        e = TensorEngine(ReferenceVectorBackend(), slots=8)
        y = e.nonlinear("reciprocal", (e.encrypt([0.5, 1.0]),))
        self.assertLess(float(np.max(np.abs(y.chunks[-1].value[2:]))), 2.0)

    def test_exact_host_policy_rejects_crypto_backend(self):
        class Crypto:
            kind = "ckks"

        with self.assertRaises(ValueError):
            TensorEngine(Crypto(), exact_reference=True)


if __name__ == "__main__":
    unittest.main()


class PositiveDomainTests(unittest.TestCase):
    def test_public_positive_interval_limits_inverse_range_and_work(self):
        e = TensorEngine(ReferenceVectorBackend(), slots=8)
        x = e.encrypt([2.0, 4.0, 8.0], public_bound=8.0)
        y = e.nonlinear("reciprocal", (x, 2.0, 8.0))
        np.testing.assert_allclose(e.decrypt(y), [0.5, 0.25, 0.125], rtol=1e-10)
        self.assertLessEqual(y.public_bound, 0.5)
        self.assertLess(e.stats["mod_raise_requests"], 10)

    def test_invalid_or_unsupported_positive_interval_rejected(self):
        e = TensorEngine(ReferenceVectorBackend(), slots=8)
        x = e.encrypt([1.0])
        for bounds in [(0.0, 8.0), (4.0, 2.0), (float("nan"), 8.0), (1e-9, 8.0)]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                e.nonlinear("reciprocal", (x, *bounds))
