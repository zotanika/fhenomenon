"""Explicit public calibration is an empirical condition, never a proof."""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch


class PublicCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from examples.openvla_training.graph import capture_training_graph
        from examples.openvla_training.calibration import calibrate_public_ranges

        cls.captured = capture_training_graph(
            profile="micro", data_seed=2029, max_report_nodes=0
        )
        cls.contract = calibrate_public_ranges(cls.captured)

    def test_pixel_fixture_seed_is_separate_and_default_stays_constant(self):
        from examples.openvla_training.graph import public_pixel_fixture

        before = torch.random.get_rng_state().clone()
        first = public_pixel_fixture((2, 3), data_seed=101)
        self.assertTrue(torch.equal(first, public_pixel_fixture((2, 3), data_seed=101)))
        self.assertFalse(
            torch.equal(first, public_pixel_fixture((2, 3), data_seed=2029))
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(bool(((first >= -0.5) & (first < 0.5)).all()))
        self.assertTrue(
            torch.equal(public_pixel_fixture((2, 3)), torch.full((2, 3), 0.25))
        )
        with self.assertRaises(ValueError):
            public_pixel_fixture((2, 3), data_seed=-1)

    def test_public_fixture_provenance_and_digest_are_explicit(self):
        from examples.openvla_training.calibration import load_range_contract

        p = self.contract["provenance"]
        self.assertEqual(p["calibration_seeds"], [101, 102, 103, 104])
        self.assertEqual(p["guarantee"], "empirical_not_proof")
        self.assertFalse(p["training_data_used"])
        self.assertEqual(p["model_seed"], 7)
        self.assertEqual(p["headroom"], 4.0)
        self.assertEqual(p["floor"], 0.01)
        self.assertGreater(len(self.contract["nodes"]), 100)
        nodes = {node.name: node for node in self.captured.gm.graph.nodes}
        self.assertTrue(
            all(nodes[name].op == "call_function" for name in self.contract["nodes"])
        )
        self.assertFalse(
            any(
                str(nodes[name].target)
                in {"aten.view.default", "aten.slice.Tensor", "aten.expand.default"}
                for name in self.contract["nodes"]
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public-ranges.json"
            path.write_text(json.dumps(self.contract, allow_nan=False))
            loaded = load_range_contract(path, self.captured)
            self.assertEqual(
                loaded["source"]["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_mismatched_graph_public_weights_and_invalid_bound_fail_closed(self):
        from examples.openvla_training.calibration import validate_range_contract

        changed = copy.deepcopy(self.captured.gm)
        next(
            n for n in changed.graph.nodes if str(n.target) == "aten.add.Tensor"
        ).kwargs = {"alpha": 2}
        with self.assertRaisesRegex(ValueError, "graph"):
            validate_range_contract(self.contract, replace(self.captured, gm=changed))
        changed = copy.deepcopy(self.captured.gm)

        def alternative(grad, x, reciprocal, dims):
            return grad * 0

        alternative.__name__ = "rms_norm_backward"
        next(
            n
            for n in changed.graph.nodes
            if getattr(n.target, "__name__", None) == "rms_norm_backward"
        ).target = alternative
        with self.assertRaisesRegex(ValueError, "graph"):
            validate_range_contract(self.contract, replace(self.captured, gm=changed))
        fixed = dict(self.captured.args[1])
        name = next(iter(fixed))
        fixed[name] = fixed[name] + 1
        args = (self.captured.args[0], fixed, *self.captured.args[2:])
        with self.assertRaisesRegex(ValueError, "public weights"):
            validate_range_contract(self.contract, replace(self.captured, args=args))
        from examples.openvla_training.interpreter import GraphInterpreter
        from python.native_tests.test_interpreter import TorchNumericEngine

        with self.assertRaisesRegex(ValueError, "public weights"):
            GraphInterpreter(TorchNumericEngine()).run(
                self.captured, args, range_contracts=self.contract
            )
        malformed = copy.deepcopy(self.contract)
        entry = next(iter(malformed["nodes"].values()))
        next(leaf for leaf in entry["leaves"] if leaf["kind"] == "tensor")["bound"] = (
            float("inf")
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_range_contract(malformed, self.captured)

    def test_calibration_does_not_observe_captured_training_payloads(self):
        from examples.openvla_training.calibration import calibrate_public_ranges

        args = tuple(self.captured.args)
        adapters = {k: torch.full_like(v, float("nan")) for k, v in args[0].items()}
        poisoned = (
            adapters,
            args[1],
            args[2],
            torch.full_like(args[3], float("nan")),
            torch.full_like(args[4], float("nan")),
            torch.full_like(args[5], float("nan")),
        )
        repeated = calibrate_public_ranges(
            replace(self.captured, args=poisoned, eager_result=None, validation=None)
        )
        self.assertEqual(repeated, self.contract)

    def test_heldout_reference_checks_every_applied_bound(self):
        from examples.openvla_training.interpreter import GraphInterpreter
        from examples.openvla_training.run_training import _adapter_public_bounds
        from fhenomenon.ml.tensor import TensorEngine, ReferenceVectorBackend

        engine = TensorEngine(ReferenceVectorBackend(), slots=256)
        a, fixed, buffers, tokens, pixels, targets = self.captured.args
        bounds = _adapter_public_bounds(a, "initialization", 1.0)
        args = (
            {k: engine.encrypt(v, public_bound=bounds[k]) for k, v in a.items()},
            fixed,
            buffers,
            engine.encrypt(tokens),
            engine.encrypt(pixels),
            engine.encrypt(targets),
        )
        interpreter = GraphInterpreter(engine)
        output = interpreter.run(self.captured, args, range_contracts=self.contract)
        self.assertGreater(interpreter.stats["empirical_range_applications"], 100)
        self.assertEqual(engine.backend.stats["terminal_decryptions"], 0)
        self.assertTrue(np.isfinite(engine.decrypt(output[0])).all())
        self.assertAlmostEqual(
            float(engine.decrypt(output[0])),
            float(self.captured.eager_result[0]),
            delta=1e-5,
        )

    def test_crypto_range_application_only_reads_public_metadata(self):
        from examples.openvla_training.calibration import apply_node_range
        from fhenomenon.ml.tensor import Tensor, TensorEngine

        class Opaque:
            count = 8  # Public packed extent, not ciphertext payload.

            @property
            def value(self):
                raise AssertionError("secret payload inspection")

        engine = TensorEngine(SimpleNamespace(kind="ckks"), slots=8)
        value = Tensor(engine, [Opaque()], np.arange(3), public_bound=9.0)
        value._packed = value.chunks
        from torch.utils._pytree import tree_flatten, treespec_dumps

        entry = {
            "tree_spec": treespec_dumps(tree_flatten(torch.ones(3))[1]),
            "leaves": [
                {"kind": "tensor", "shape": [3], "dtype": "torch.float32", "bound": 2.0}
            ],
        }
        result = apply_node_range(value, entry, engine)
        self.assertEqual(result.public_bound, 2.0)
        self.assertIs(result.chunks[0], value.chunks[0])
        entry["leaves"][0]["bound"] = 20.0
        self.assertIs(apply_node_range(value, entry, engine), value)


if __name__ == "__main__":
    unittest.main()
