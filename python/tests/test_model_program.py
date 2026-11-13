import json
import unittest
import torch

from fhenomenon.ml import (
    capture_training,
    compile_model,
    KernelRegistry,
    KernelDefinition,
    KernelMatch,
    TensorEngine,
    ReferenceVectorBackend,
)


class LowRankModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(3, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.a = torch.nn.Parameter(torch.tensor([[0.1, 0.2, 0.1]]))
        self.b = torch.nn.Parameter(torch.tensor([[0.2], [0.1]]))

    def forward(self, x):
        return self.base(x) + x @ self.a.T @ self.b.T


def loss(prediction, targets):
    return ((prediction - targets) ** 2).mean()


class ModelProgramTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.model = LowRankModel()
        self.inputs = (torch.tensor([[0.2, 0.1, 0.3]]),)
        self.targets = torch.tensor([[0.15, 0.2]])

    def capture(self):
        return capture_training(
            self.model,
            self.inputs,
            self.targets,
            loss,
            learning_rate=0.01,
            input_visibility=("secret",),
        )

    def test_unmodified_module_captures_full_training_and_reference_executes(self):
        captured = self.capture()
        program = compile_model(captured)
        self.assertTrue(program.supported)
        engine = TensorEngine(ReferenceVectorBackend(), slots=16, exact_reference=True)
        bound = program.encrypt_inputs(engine, public_bound=2.0)
        actual = program.execute(engine, bound)

        def check(a, b):
            if isinstance(a, dict):
                for key in a:
                    check(a[key], b[key])
            else:
                torch.testing.assert_close(
                    torch.as_tensor(engine.decrypt(a)), b.double(), atol=1e-7, rtol=1e-7
                )

        for a, b in zip(actual, captured.eager_result):
            check(a, b)
        self.assertIn("aten.mm.default", {node.operation for node in program.nodes})
        self.assertEqual(set(actual[1]), {"a", "b"})
        self.assertEqual(program.execution_report["private_host_fallback_count"], 0)

    def test_registry_contribution_needs_no_interpreter_changes(self):
        called = []
        registry = KernelRegistry()
        registry.register(
            KernelDefinition(
                name="test.tensor_matmul",
                version=1,
                operations=("aten.mm.default",),
                requirements={"scheme": "ckks", "layout": "public tensor layout"},
                accepts=lambda node, environment: KernelMatch.accept(),
                execute=lambda context, args, kwargs: (
                    called.append(context.node.name) or context.engine.matmul(*args)
                ),
            )
        )
        program = compile_model(self.capture(), registry=registry)
        registry.register(
            KernelDefinition(
                name="later",
                version=1,
                operations=("aten.mm.default",),
                priority=999,
                requirements={},
                accepts=lambda *_: KernelMatch.accept(),
                execute=lambda *_: None,
            )
        )
        engine = TensorEngine(ReferenceVectorBackend(), slots=16, exact_reference=True)
        program.execute(engine, program.encrypt_inputs(engine, public_bound=2.0))
        self.assertTrue(called)
        self.assertNotIn("later", program.execution_report["kernel_calls"])
        self.assertEqual(
            sum(program.execution_report["kernel_calls"].values()), len(called)
        )

    def test_unknown_secret_operator_reported_before_input_encryption(self):
        class Unknown(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(3))

            def forward(self, x):
                return torch.sin(x * self.weight)

        captured = capture_training(
            Unknown(),
            self.inputs,
            torch.zeros(1, 3),
            loss,
            input_visibility=("secret",),
        )
        program = compile_model(captured)
        self.assertFalse(program.supported)
        self.assertIn(
            "aten.sin.default",
            {x["operation"] for x in program.kernel_requests()["requests"]},
        )
        with self.assertRaisesRegex(NotImplementedError, "unsupported"):
            program.encrypt_inputs(
                TensorEngine(ReferenceVectorBackend()), public_bound=2.0
            )

    def test_request_artifact_has_contracts_but_no_parameter_contents(self):
        program = compile_model(self.capture())
        request = program.kernel_requests()
        self.assertEqual(request["schema_version"], 1)
        self.assertEqual(
            request["latency_estimates"], "unpriced_without_matching_measurements"
        )
        text = json.dumps(request, allow_nan=False)
        self.assertNotIn("state_dict", text)
        self.assertNotIn("tensor(", text)
        self.assertTrue(request["requests"])
        for item in request["requests"]:
            self.assertIn("signature", item)
            self.assertIn("input_contracts", item)
            self.assertIn("acceptance", item)
            self.assertIn("output_contract", item)

    def test_selected_kernel_failure_is_not_silently_decomposed(self):
        registry = KernelRegistry()

        def fail(*_):
            raise RuntimeError("deliberate kernel failure")

        registry.register(
            KernelDefinition(
                name="broken",
                version=1,
                operations=("aten.mm.default",),
                requirements={},
                accepts=lambda *_: KernelMatch.accept(),
                execute=fail,
            )
        )
        program = compile_model(self.capture(), registry=registry)
        engine = TensorEngine(ReferenceVectorBackend(), slots=16, exact_reference=True)
        with self.assertRaisesRegex(RuntimeError, "deliberate kernel failure"):
            program.execute(engine, program.encrypt_inputs(engine, public_bound=2.0))

    def test_frozen_weight_privacy_is_explicit_and_supported(self):
        captured = capture_training(
            self.model,
            self.inputs,
            self.targets,
            loss,
            input_visibility=("secret",),
            frozen_visibility="secret",
        )
        program = compile_model(captured)
        specs = [s for s in program.input_contracts if s["role"] == "frozen_weight"]
        self.assertTrue(specs and all(s["visibility"] == "secret" for s in specs))

    def test_changed_graph_is_rejected_before_binding(self):
        captured = self.capture()
        program = compile_model(captured)
        node = next(
            n for n in captured.gm.graph.nodes if str(n.target) == "aten.mm.default"
        )
        node.meta["training"]["privacy"] = "public"
        with self.assertRaisesRegex(ValueError, "graph changed"):
            program.encrypt_inputs(
                TensorEngine(ReferenceVectorBackend()), public_bound=2.0
            )

    def test_wrong_bound_input_shape_is_rejected_before_execution(self):
        program = compile_model(self.capture())
        engine = TensorEngine(ReferenceVectorBackend(), slots=16)
        bound = program.encrypt_inputs(engine, public_bound=2.0)
        bound = (*bound[:-1], engine.encrypt(torch.ones(2, 2), public_bound=2.0))
        with self.assertRaisesRegex(ValueError, "input shape/dtype"):
            program.execute(engine, bound)

    def test_wrong_output_shape_from_contribution_is_rejected(self):
        registry = KernelRegistry()
        registry.register(
            KernelDefinition(
                name="wrong_shape",
                version=1,
                operations=("aten.mm.default",),
                requirements={},
                accepts=lambda *_: KernelMatch.accept(),
                execute=lambda ctx, args, kwargs: args[0],
            )
        )
        program = compile_model(self.capture(), registry=registry)
        engine = TensorEngine(ReferenceVectorBackend(), slots=16)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            program.execute(engine, program.encrypt_inputs(engine, public_bound=2.0))

    def test_keywords_participate_in_signature(self):
        from fhenomenon.ml import NodeContract

        a = NodeContract(
            "a", "aten.addmm.default", "training", (), None, "secret", {"alpha": 1}
        )
        b = NodeContract(
            "b", "aten.addmm.default", "training", (), None, "secret", {"alpha": 2}
        )
        self.assertNotEqual(a.signature, b.signature)

    def test_request_identity_binds_contributor_version(self):
        def requests(version):
            registry = KernelRegistry()
            registry.register(
                KernelDefinition(
                    name="same_name",
                    version=version,
                    operations=("aten.mm.default",),
                    requirements={},
                    accepts=lambda *_: KernelMatch.accept(),
                    execute=lambda ctx, args, kwargs: ctx.engine.matmul(*args),
                )
            )
            return compile_model(self.capture(), registry=registry).kernel_requests()

        self.assertNotEqual(requests(1), requests(2))


if __name__ == "__main__":
    unittest.main()
