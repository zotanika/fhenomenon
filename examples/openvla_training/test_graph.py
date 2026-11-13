"""Regression tests for the actual OpenVLA joint training graph."""

import importlib.util
import json
from pathlib import Path
import unittest


class RmsBackwardRewriteTests(unittest.TestCase):
    @staticmethod
    def graph(divisor=4, exponent=3, epsilon=1e-6, residual=False):
        import torch
        from torch.fx.experimental.proxy_tensor import make_fx

        def original(x, grad):
            r = torch.rsqrt(x.pow(2).mean([-1], True) + epsilon)
            correction = (grad * x).sum([-1], True) * -0.5 * r.pow(exponent)
            correction = correction.expand(x.shape) / divisor
            direct = grad * r
            if residual:
                direct = x * 0.3 + direct
            return direct + correction * (x.pow(1.0) * 2.0)

        x = torch.randn(2, 3, 4, dtype=torch.float64)
        grad = torch.randn_like(x)
        return make_fx(original)(x, grad)

    def test_rms_composite_preserves_arbitrary_saved_reciprocal_values(self):
        import torch
        from fhenomenon.ml.rms import rms_norm_backward

        generator = torch.Generator().manual_seed(12)
        for width in (1, 4, 8):
            for epsilon in (1e-6, 0.25):
                x = torch.randn(2, 3, width, generator=generator, dtype=torch.float64)
                grad = torch.randn(x.shape, generator=generator, dtype=torch.float64)
                reciprocal = torch.rsqrt(x.square().mean(-1, True) + epsilon)
                # Saved approximate r is deliberately different from exact r.
                reciprocal = reciprocal * 0.997
                expected = grad * reciprocal - x * reciprocal.pow(3) * (grad * x).mean(
                    -1, True
                )
                actual = rms_norm_backward(grad, x, reciprocal, [-1])
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_rewrite_removes_cubed_reciprocal_and_matches_random_oracle(self):
        import torch
        from examples.openvla_training.graph import rewrite_rms_backward
        from fhenomenon.ml.rms import rms_norm_backward

        generator = torch.Generator().manual_seed(13)
        for residual in (False, True):
            original = self.graph(residual=residual)
            rewritten = self.graph(residual=residual)
            self.assertEqual(rewrite_rms_backward(rewritten), 1)
            self.assertTrue(
                any(n.target is rms_norm_backward for n in rewritten.graph.nodes)
            )
            self.assertFalse(
                any(
                    str(n.target) == "aten.pow.Tensor_Scalar" and n.args[1] == 3
                    for n in rewritten.graph.nodes
                )
            )
            for _ in range(8):
                x = torch.randn(
                    2,
                    3,
                    4,
                    generator=generator,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                grad = torch.randn(x.shape, generator=generator, dtype=torch.float64)
                torch.testing.assert_close(
                    rewritten(x, grad), original(x, grad), atol=1e-12, rtol=1e-12
                )
                y = x * torch.rsqrt(x.square().mean(-1, True) + 1e-6)
                (oracle,) = torch.autograd.grad(y, x, grad)
                if residual:
                    oracle = oracle + x * 0.3
                torch.testing.assert_close(
                    rewritten(x, grad), oracle, atol=1e-12, rtol=1e-12
                )

    def test_similar_unmatched_graphs_are_unchanged(self):
        from examples.openvla_training.graph import rewrite_rms_backward

        for kwargs in ({"divisor": 5}, {"exponent": 4}, {"epsilon": -1e-6}):
            gm = self.graph(**kwargs)
            before = gm.code
            self.assertEqual(rewrite_rms_backward(gm), 0)
            self.assertEqual(gm.code, before)


class TrainingGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        cls.captured = None

    def graph(self):
        graph_path = Path(__file__).with_name("graph.py")
        self.assertTrue(graph_path.is_file(), "actual OpenVLA graph capture is missing")
        if self.__class__.module is None:
            from examples.openvla_training import graph

            self.__class__.module = graph
        if self.__class__.captured is None:
            self.__class__.captured = self.module.capture_training_graph(
                max_report_nodes=16
            )
        return self.captured

    def test_joint_graph_matches_every_eager_adapter_gradient_and_update(self):
        captured = self.graph()
        self.assertEqual(
            type(captured.model.base_model.model).__name__, "OpenVLAForActionPrediction"
        )
        self.assertEqual(captured.validation["loss_absolute_error"], 0.0)
        self.assertLessEqual(captured.validation["gradient_max_absolute_error"], 2e-5)
        self.assertLessEqual(captured.validation["update_max_absolute_error"], 2e-7)
        self.assertGreater(captured.metadata["rms_backward_rewrites"], 0)
        self.assertTrue(captured.validation["native_reference"]["passed"])
        self.assertEqual(
            captured.validation["compared_adapter_tensors"],
            len(captured.trainable_names),
        )
        self.assertGreater(captured.validation["nonzero_updated_tensors"], 0)

    def test_private_indices_are_lowered_and_input_taint_is_conservative(self):
        captured = self.graph()
        operators = captured.metadata["operators"]
        self.assertNotIn("aten.embedding.default", operators)
        self.assertNotIn("aten.nll_loss_forward.default", operators)
        self.assertNotIn("aten.nll_loss_backward.default", operators)
        self.assertIn("aten.convolution_backward.default", operators)
        secret_roles = {
            n.meta["training"]["role"]
            for n in captured.gm.graph.nodes
            if n.op == "placeholder" and n.meta["training"]["privacy"] == "secret"
        }
        self.assertEqual(
            secret_roles,
            {"adapter", "instruction_onehot", "pixels", "action_targets_onehot"},
        )
        out = list(captured.gm.graph.nodes)[-1]
        self.assertEqual(out.meta["training"]["privacy"], "secret")

    def test_report_bounds_details_without_losing_cost_totals_or_egress_policy(self):
        captured = self.graph()
        report = json.loads(json.dumps(captured.metadata, allow_nan=False))
        self.assertLessEqual(len(report["nodes"]), 16)
        self.assertEqual(report["node_count"], len(list(captured.gm.graph.nodes)))
        self.assertGreater(report["cost_estimates"]["dense_multiply_accumulates"], 0)
        self.assertEqual(
            report["runtime_policy"]["intermediate_decryptions_allowed"], 0
        )
        self.assertEqual(
            report["runtime_policy"]["final_egress_level"], "minimum_supported_level"
        )
        self.assertFalse(report["encrypted_execution_performed"])
        phases = {n.meta["training"]["phase"] for n in captured.gm.graph.nodes}
        self.assertTrue({"forward", "backward", "update"}.issubset(phases))

    def test_cli_emits_a_finite_report_with_no_tensor_payloads(self):
        import subprocess
        import sys
        import tempfile

        runner = Path(__file__).with_name("run_graph.py")
        self.assertTrue(runner.is_file(), "graph report CLI is missing")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "graph.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "examples.openvla_training.run_graph",
                    "--output",
                    str(output),
                    "--max-report-nodes",
                    "4",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(len(report["nodes"]), 4)
            self.assertTrue(report["validation"]["passed"])
            self.assertFalse(report["encrypted_execution_performed"])

    def test_meta_full7b_profile_is_shape_only_and_keeps_full_architecture(self):
        self.graph()
        captured = self.module.capture_training_graph(
            profile="full7b_meta", max_report_nodes=0
        )
        self.assertIsNone(captured.eager_result)
        self.assertIsNone(captured.validation)
        self.assertFalse(captured.metadata["checkpoint_weights_loaded"])
        self.assertGreater(captured.metadata["base_parameter_count"], 7_000_000_000)
        self.assertGreater(captured.metadata["node_count"], 20_000)
        self.assertEqual(captured.metadata["nodes"], [])

    def test_micro_keeps_all_model_stages_and_complete_adapter_backward(self):
        self.graph()
        captured = self.module.capture_training_graph(
            profile="micro", max_report_nodes=0
        )
        model = captured.model.base_model.model
        self.assertEqual(type(model).__name__, "OpenVLAForActionPrediction")
        self.assertEqual(len(model.language_model.model.layers), 2)
        self.assertEqual(len(model.vision_backbone.featurizer.blocks), 2)
        self.assertEqual(len(model.vision_backbone.fused_featurizer.blocks), 2)
        self.assertEqual(model.language_model.config.hidden_size, 4)
        self.assertEqual(captured.metadata["input_shapes"]["pixels"], [1, 6, 2, 2])
        self.assertTrue(captured.validation["passed"])
        self.assertLessEqual(captured.validation["gradient_max_absolute_error"], 2e-5)
        self.assertLessEqual(captured.validation["update_max_absolute_error"], 2e-7)
        self.assertEqual(captured.metadata["rms_backward_rewrites"], 5)
        self.assertIn(
            "aten.convolution_backward.default", captured.metadata["operators"]
        )


if __name__ == "__main__":
    unittest.main()
