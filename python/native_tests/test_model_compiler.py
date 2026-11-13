"""Public model API connected to the local encrypted runtime provider."""

import os
import unittest

import torch

from fhenomenon.ml import (
    TensorEngine,
    capture_training,
    compile_model,
    native_tensor_registry,
    runtime_environment,
    price_kernel_requests,
)
from fhenomenon.backends.runtime import create_runtime_context


class LowRankModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(3, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.a = torch.nn.Parameter(torch.tensor([[0.1, 0.2, 0.1]]))
        self.b = torch.nn.Parameter(torch.tensor([[0.2], [0.1]]))

    def forward(self, x):
        return self.base(x) + x @ self.a.T @ self.b.T


class ModelCompilerIntegrationTests(unittest.TestCase):
    def test_full_training_uses_coarse_kernel_and_preserves_scope_and_cost_identity(
        self,
    ):
        torch.manual_seed(14)
        client, backend = create_runtime_context(
            profile="NativeTest",
            device=os.environ.get("OPENVLA_SCOPE_TEST_DEVICE", "cpu"),
            mod_raise_levels=[8],
        )
        try:
            engine = TensorEngine(backend, client=client, slots=16)
            captured = capture_training(
                LowRankModel(),
                (torch.tensor([[0.2, 0.1, 0.3]]),),
                torch.tensor([[0.15, 0.2]]),
                lambda prediction, target: ((prediction - target) ** 2).mean(),
                input_visibility=("secret",),
                learning_rate=0.01,
            )
            program = compile_model(
                captured,
                registry=native_tensor_registry(),
                environment=runtime_environment(engine),
            )
            bound = program.encrypt_inputs(engine, public_bound=2.0)
            outputs = program.execute(engine, bound)
            self.assertEqual(client.stats["decrypt_calls"], 0)
            self.assertEqual(backend.stats["evaluator_decrypt_calls"], 0)
            report = program.execution_report
            self.assertEqual(report["kernel_calls"]["native.tensor.public_linear"], 1)
            scope = backend.runtime_stats["last_scope"]
            self.assertEqual(scope["kernel_spans"], 1)
            self.assertEqual(scope["commands"], scope["executed_commands"])
            self.assertEqual(report["private_host_fallback_count"], 0)
            evidence = program.measurement()
            self.assertIsNotNone(evidence)
            self.assertEqual(
                len(evidence["node_measurements"]),
                sum(n.visibility == "secret" for n in program.nodes),
            )
            request = next(
                item
                for item in program.kernel_requests()["requests"]
                if item["current_implementation"] == "native.tensor.public_linear@1"
            )
            estimate = price_kernel_requests(
                program.kernel_requests(),
                evidence,
                [
                    {
                        "signature": request["signature"],
                        "assumed_seconds_per_invocation": 0.0,
                        "setup_seconds": 0.0,
                        "transfer_seconds": 0.0,
                    }
                ],
            )
            self.assertIsNotNone(estimate["combined"])
            calls = dict(backend.runtime_stats["recipe_calls"])

            def check(actual, expected):
                if isinstance(actual, dict):
                    for name in actual:
                        check(actual[name], expected[name])
                else:
                    torch.testing.assert_close(
                        torch.tensor(engine.decrypt(actual)),
                        expected.double(),
                        atol=2e-5,
                        rtol=2e-5,
                    )

            for actual, expected in zip(outputs, captured.eager_result):
                check(actual, expected)
            self.assertEqual(calls, backend.runtime_stats["recipe_calls"])
            # A later preflight failure cannot reuse the previous successful timing.
            with self.assertRaises(ValueError):
                program.execute(engine, (*bound[:-1], torch.zeros(5)))
            self.assertIsNone(program.measurement())
        finally:
            client.close()
            backend.close()
