"""Numeric lowering checks; the Torch engine here is never encryption evidence."""

import importlib.util
import operator
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten


def load_local(name):
    import importlib

    module = (
        "examples.openvla_training." + name
        if name in {"graph", "interpreter"}
        else "fhenomenon.ml." + name
    )
    return importlib.import_module(module)


class TorchNumericEngine:
    """Plain arithmetic reference for testing an encrypted-engine interface."""

    kind = "noncryptographic_reference"
    add = staticmethod(torch.add)
    sub = staticmethod(torch.sub)
    mul = staticmethod(torch.mul)
    div = staticmethod(torch.div)
    neg = staticmethod(torch.neg)
    pow = staticmethod(torch.pow)
    matmul = staticmethod(torch.matmul)
    cat = staticmethod(torch.cat)
    stack = staticmethod(torch.stack)

    def __init__(self):
        self.exp_maximum_magnitude = 0.0

    def reshape(self, x, shape):
        return x.reshape(shape)

    def permute(self, x, dims):
        return x.permute(dims)

    def expand(self, x, shape):
        return x.expand(shape)

    def sum(self, x, dims=None, keepdim=False):
        return x.sum(dim=dims, keepdim=keepdim)

    def mean(self, x, dims=None, keepdim=False):
        return x.mean(dim=dims, keepdim=keepdim)

    def zeros_like(self, x):
        return torch.zeros_like(x)

    def clone(self, x):
        return x

    def slice(self, x, dim, start, end, step=1):
        slices = [slice(None)] * x.ndim
        slices[dim] = slice(start, end, step)
        return x[tuple(slices)]

    def slice_backward(self, grad, input_sizes, dim, start, end, step):
        return torch.ops.aten.slice_backward.default(
            grad, input_sizes, dim, start, end, step
        )

    def nonlinear(self, name, args):
        x = args[0]
        if name == "exp":
            self.exp_maximum_magnitude = max(
                self.exp_maximum_magnitude, float(x.abs().max())
            )
        if name == "gelu":
            return F.gelu(x, approximate=args[1] if len(args) > 1 else "none")
        if name == "gelu_backward":
            return torch.ops.aten.gelu_backward.default(
                args[0], args[1], approximate=args[2] if len(args) > 2 else "none"
            )
        return {
            "exp": torch.exp,
            "log": torch.log,
            "reciprocal": torch.reciprocal,
            "rsqrt": torch.rsqrt,
            "sigmoid": torch.sigmoid,
            "erf": torch.erf,
        }[name](x)

    def conv2d(self, x, weight, bias, stride, padding, dilation, groups):
        return F.conv2d(x, weight, bias, stride, padding, dilation, groups)

    def conv2d_backward(
        self, grad, x, weight, stride, padding, dilation, groups, output_mask
    ):
        return torch.ops.aten.convolution_backward.default(
            grad,
            x,
            weight,
            [weight.shape[0]],
            stride,
            padding,
            dilation,
            False,
            [0, 0],
            groups,
            output_mask,
        )


class InterpreterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.module = load_local("interpreter")

    def interpreter(self):
        self.assertIsNotNone(self.module, "explicit private ATen lowering is missing")
        return self.module.GraphInterpreter(TorchNumericEngine())

    def test_full_model_forward_backward_and_updates_match_captured_graph(self):
        interpreter = self.interpreter()
        graph = load_local("graph").capture_training_graph(max_report_nodes=0)
        progress = []
        actual = interpreter.run(
            graph,
            progress=lambda index, total, node: progress.append(
                (index, total, node.name)
            ),
        )
        expected = graph.eager_result
        torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=1e-5)
        self.assertEqual(actual[1].keys(), expected[1].keys())
        for name in expected[1]:
            torch.testing.assert_close(
                actual[1][name], expected[1][name], atol=2e-5, rtol=1e-4
            )
            torch.testing.assert_close(
                actual[2][name], expected[2][name], atol=2e-7, rtol=1e-5
            )
        self.assertEqual(interpreter.stats["private_host_fallback_count"], 0)
        self.assertLess(
            interpreter.stats.get(
                "peak_live_graph_values", graph.metadata["node_count"]
            ),
            graph.metadata["node_count"] // 2,
            "consumed ciphertext values must be released during graph execution",
        )
        self.assertLess(
            interpreter.engine.exp_maximum_magnitude,
            100,
            "public causal-mask sentinels reached the nonlinear approximation",
        )
        self.assertEqual(len(progress), graph.metadata["node_count"])
        self.assertEqual(
            progress[-1][:2],
            (graph.metadata["node_count"], graph.metadata["node_count"]),
        )

    def test_unknown_secret_operation_fails_instead_of_calling_torch(self):
        interpreter = self.interpreter()
        with self.assertRaisesRegex(NotImplementedError, "atan"):
            interpreter.lower(torch.ops.aten.atan.default, (torch.ones(3),), {})

    def test_crypto_inputs_are_checked_before_any_graph_execution(self):
        # This checks the input type/ownership boundary, not FHE arithmetic.
        from fhenomenon.ml.tensor import Tensor
        import numpy as np

        engine = SimpleNamespace(backend=SimpleNamespace(kind="ckks"))
        other = SimpleNamespace(backend=SimpleNamespace(kind="ckks"))
        encrypted = Tensor(engine, (), np.empty(0, dtype=np.int64))
        foreign = Tensor(other, (), np.empty(0, dtype=np.int64))
        plaintext = torch.zeros(0)
        frozen = torch.ones(1)
        args = ({"adapter": plaintext}, frozen, [plaintext, plaintext])
        graph = torch.fx.Graph()
        placeholders = [
            graph.placeholder(name)
            for name in ("adapter", "weight", "pixels", "targets")
        ]
        for node, privacy in zip(
            placeholders, ("secret", "public", "secret", "secret")
        ):
            node.meta["training"] = {"privacy": privacy}
        output = graph.output(tuple(placeholders))
        output.meta["training"] = {"privacy": "secret"}
        gm = SimpleNamespace(
            graph=graph,
            _in_spec=tree_flatten(args)[1],
            _out_spec=tree_flatten(tuple(range(4)))[1],
        )
        captured = SimpleNamespace(gm=gm, args=args)
        interpreter = self.module.GraphInterpreter(engine)
        bad_inputs = [
            None,
            args,
            ({"adapter": encrypted}, frozen, [encrypted, plaintext]),
            ({"adapter": encrypted}, frozen, [encrypted, foreign]),
        ]
        for supplied in bad_inputs:
            progress = []
            with self.subTest(
                supplied="captured defaults" if supplied is None else "explicit inputs"
            ):
                with self.assertRaisesRegex(
                    ValueError, "secret placeholder.*same engine"
                ):
                    interpreter.run(
                        captured,
                        supplied,
                        progress=lambda *event: progress.append(event),
                    )
                self.assertEqual(
                    progress, [], "input rejection must precede all graph execution"
                )
        supplied = ({"adapter": encrypted}, frozen, [encrypted, encrypted])
        result = interpreter.run(captured, supplied)
        self.assertIs(result[0], encrypted)
        self.assertIs(result[1], frozen)
        self.assertIs(result[2], encrypted)
        self.assertIs(result[3], encrypted)

        # Plaintext is opt-in; an unmarked custom engine is not a reference policy.
        with self.assertRaisesRegex(ValueError, "secret placeholder"):
            self.module.GraphInterpreter(SimpleNamespace()).run(captured)
        result = self.module.GraphInterpreter(TorchNumericEngine()).run(captured)
        self.assertIs(result[0], plaintext)

    def test_layernorm_backward_respects_frozen_affine_parameters(self):
        interpreter = self.interpreter()
        x = torch.tensor([[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]])
        weight = torch.tensor([1.0, 0.5, 1.5])
        bias = torch.tensor([0.1, 0.2, 0.3])
        grad = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]])
        y, mean, rstd = interpreter.lower(
            torch.ops.aten.native_layer_norm.default, (x, [3], weight, bias, 1e-6), {}
        )
        got = interpreter.lower(
            torch.ops.aten.native_layer_norm_backward.default,
            (grad, x, [3], mean, rstd, weight, bias, [True, False, False]),
            {},
        )
        expected = torch.ops.aten.native_layer_norm_backward.default(
            grad, x, [3], mean, rstd, weight, bias, [True, False, False]
        )
        torch.testing.assert_close(got[0], expected[0], atol=1e-6, rtol=1e-5)
        self.assertIsNone(got[1])
        self.assertIsNone(got[2])

    def test_polynomial_normalization_uses_correlated_public_bounds(self):
        from fhenomenon.ml.tensor import ReferenceVectorBackend, TensorEngine

        backend = ReferenceVectorBackend()
        engine = TensorEngine(backend, slots=8, exact_reference=False)
        interpreter = self.module.GraphInterpreter(engine)
        source = torch.tensor([[-2.0, -1.0, 0.0, 1.0], [-1.0, 0.5, 0.0, 2.0]])
        x = engine.encrypt(source, public_bound=8.0)
        probabilities = interpreter.lower(
            torch.ops.aten._softmax.default, (x, -1, False), {}
        )
        self.assertLessEqual(probabilities.public_bound, 1.0)
        torch.testing.assert_close(
            torch.as_tensor(engine.decrypt(probabilities)),
            source.double().softmax(-1),
            atol=1e-7,
            rtol=1e-6,
        )
        g = engine.encrypt(torch.full((2, 4), 0.25), public_bound=2.0)
        backward = interpreter.lower(
            torch.ops.aten._softmax_backward_data.default,
            (g, probabilities, -1, torch.float32),
            {},
        )
        self.assertLessEqual(backward.public_bound, 4.0)
        weight, bias = torch.tensor([1.0, 0.5, 1.5, 1.0]), torch.full((4,), 0.125)
        y, mean, rstd = interpreter.lower(
            torch.ops.aten.native_layer_norm.default, (x, [4], weight, bias, 1e-6), {}
        )
        self.assertLessEqual(y.public_bound, 3.125)
        torch.testing.assert_close(
            torch.as_tensor(engine.decrypt(y)),
            F.layer_norm(source.double(), [4], weight.double(), bias.double(), 1e-6),
            atol=1e-7,
            rtol=1e-6,
        )
        backward = interpreter.lower(
            torch.ops.aten.native_layer_norm_backward.default,
            (g, x, [4], mean, rstd, weight, bias, [True, True, True]),
            {},
        )
        self.assertLessEqual(backward[1].public_bound, 8.0)
        self.assertEqual(backend.stats["intermediate_decryptions"], 0)

    def test_rms_bound_requires_same_source_and_nonnegative_epsilon(self):
        from fhenomenon.ml.tensor import ReferenceVectorBackend, TensorEngine

        engine = TensorEngine(ReferenceVectorBackend(), slots=8)
        source = torch.tensor([[-2.0, -1.0, 0.0, 1.0], [-1.0, 0.5, 0.0, 2.0]])
        x = engine.encrypt(source, public_bound=4.0)
        other = engine.encrypt(torch.full_like(source, 10.0), public_bound=10.0)
        for matched, epsilon in ((True, 1e-6), (False, 1e-6), (True, -1e-6)):
            graph = torch.fx.Graph()
            a, b = graph.placeholder("x"), graph.placeholder("other")
            squared = graph.call_function(torch.ops.aten.pow.Tensor_Scalar, (a, 2))
            mean = graph.call_function(torch.ops.aten.mean.dim, (squared, [-1], True))
            offset = graph.call_function(torch.ops.aten.add.Tensor, (mean, epsilon))
            inverse = graph.call_function(torch.ops.aten.rsqrt.default, (offset,))
            result = graph.call_function(
                torch.ops.aten.mul.Tensor, (a if matched else b, inverse)
            )
            graph.output((result,))
            for node in graph.nodes:
                node.meta["training"] = {"privacy": "secret"}
            gm = SimpleNamespace(
                graph=graph,
                _in_spec=tree_flatten((source, source))[1],
                _out_spec=tree_flatten(source)[1],
            )
            captured = SimpleNamespace(gm=gm, args=(source, source))
            interpreter = self.module.GraphInterpreter(engine)
            actual = interpreter.run(captured, (x, other))
            with self.subTest(matched=matched, epsilon=epsilon):
                if matched and epsilon >= 0:
                    self.assertLessEqual(actual.public_bound, 2.0)
                    expected = source.double() / torch.sqrt(
                        source.double().square().mean(-1, True) + epsilon
                    )
                    torch.testing.assert_close(
                        torch.as_tensor(engine.decrypt(actual)),
                        expected,
                        atol=1e-7,
                        rtol=1e-6,
                    )
                else:
                    self.assertGreater(
                        actual.public_bound,
                        2.0,
                        "unrelated multiplication must retain its general bound",
                    )

    def test_backward_bounds_require_matching_forward_values(self):
        from fhenomenon.ml.tensor import ReferenceVectorBackend, TensorEngine

        engine = TensorEngine(ReferenceVectorBackend(), slots=8)
        interpreter = self.module.GraphInterpreter(engine)
        grad = engine.encrypt(torch.full((2, 4), 0.25), public_bound=2.0)
        arbitrary_output = engine.encrypt(torch.ones(2, 4), public_bound=1.0)
        backward = interpreter.lower(
            torch.ops.aten._softmax_backward_data.default,
            (grad, arbitrary_output, -1, torch.float32),
            {},
        )
        self.assertGreater(backward.public_bound, 4.0)
        x = engine.encrypt(
            torch.tensor([[-2.0, -1.0, 0.0, 1.0], [-1.0, 0.5, 0.0, 2.0]]),
            public_bound=4.0,
        )
        arbitrary_mean = engine.encrypt(torch.zeros(2, 1), public_bound=1.0)
        arbitrary_rstd = engine.encrypt(torch.full((2, 1), 3.0), public_bound=3.0)
        backward = interpreter.lower(
            torch.ops.aten.native_layer_norm_backward.default,
            (
                grad,
                x,
                [4],
                arbitrary_mean,
                arbitrary_rstd,
                None,
                None,
                [False, True, False],
            ),
            {},
        )
        self.assertGreater(backward[1].public_bound, 8.0)

    def test_transposed_convolution_is_explicitly_rejected(self):
        interpreter = self.interpreter()
        with self.assertRaisesRegex(NotImplementedError, "transposed"):
            interpreter.lower(
                torch.ops.aten.convolution.default,
                (
                    torch.ones(1, 1, 2, 2),
                    torch.ones(1, 1, 1, 1),
                    None,
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    True,
                    [0, 0],
                    1,
                ),
                {},
            )

    def test_packed_reference_engine_runs_without_intermediate_decryptions(self):
        self.interpreter()
        from fhenomenon.ml.tensor import ReferenceVectorBackend, TensorEngine

        graph = load_local("graph").capture_training_graph(max_report_nodes=0)
        backend = ReferenceVectorBackend()
        engine = TensorEngine(backend, slots=256, exact_reference=True)
        adapters, frozen, buffers, instructions, pixels, targets = graph.args
        args = (
            {name: engine.encrypt(value) for name, value in adapters.items()},
            frozen,
            buffers,
            engine.encrypt(instructions),
            engine.encrypt(pixels),
            engine.encrypt(targets),
        )
        interpreter = self.module.GraphInterpreter(engine)
        result = interpreter.run(graph, args)
        self.assertEqual(backend.stats["terminal_decryptions"], 0)
        self.assertEqual(backend.stats["intermediate_decryptions"], 0)
        torch.testing.assert_close(
            torch.as_tensor(engine.decrypt(result[0])),
            graph.eager_result[0].double(),
            atol=2e-6,
            rtol=1e-5,
        )
        for group in (1, 2):
            for name, value in result[group].items():
                torch.testing.assert_close(
                    torch.as_tensor(engine.decrypt(value)).double(),
                    graph.eager_result[group][name].double(),
                    atol=2e-5,
                    rtol=1e-4,
                )


if __name__ == "__main__":
    unittest.main()
