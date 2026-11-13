import unittest
import torch

from fhenomenon.ml import (
    capture_training,
    compile_model,
    TensorEngine,
    ReferenceVectorBackend,
    Tensor,
)


def loss(prediction, target):
    return ((prediction - target) ** 2).mean()


class BufferedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("state", torch.tensor([0.1, 0.2]))

    def forward(self, x):
        return x * self.weight + self.state


class FrontendPolicyTests(unittest.TestCase):
    def capture(self, model, **options):
        return capture_training(
            model,
            (torch.tensor([[0.2, 0.5]]),),
            torch.zeros(1, 2),
            loss,
            input_visibility=("secret",),
            **options,
        )

    def test_registered_buffers_are_secret_unless_owner_declares_otherwise(self):
        program = compile_model(
            self.capture(BufferedModel(), frozen_visibility="secret")
        )
        engine = TensorEngine(ReferenceVectorBackend(), slots=8)
        bound = program.encrypt_inputs(engine, public_bound=2.0)
        self.assertIsInstance(bound[2]["state"], Tensor)
        public = compile_model(
            self.capture(BufferedModel(), buffer_visibility={"state": "public"})
        )
        self.assertEqual(
            next(x for x in public.input_contracts if x["role"] == "buffer")[
                "visibility"
            ],
            "public",
        )

    def test_mask_fusion_preserves_add_alpha_zero(self):
        class Masked(BufferedModel):
            def forward(self, x):
                return torch.softmax(
                    torch.add(x * self.weight, self.state, alpha=0), dim=-1
                )

        model = Masked()
        model.state[1] = -1e31
        captured = self.capture(model, buffer_visibility="public")
        program = compile_model(captured)
        engine = TensorEngine(ReferenceVectorBackend(), slots=8, exact_reference=True)
        actual = program.execute(
            engine, program.encrypt_inputs(engine, public_bound=2.0)
        )
        torch.testing.assert_close(
            torch.tensor(engine.decrypt(actual[0])), captured.eager_result[0].double()
        )
        torch.testing.assert_close(
            torch.tensor(engine.decrypt(actual[1]["weight"])),
            captured.eager_result[1]["weight"].double(),
        )
        self.assertEqual(program.execution_report["public_mask_fusions"], 0)

    def test_random_training_is_rejected_and_rng_state_preserved(self):
        class Random(BufferedModel):
            def forward(self, x):
                return x * self.weight + torch.rand_like(x)

        model = Random()
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, "random"):
            self.capture(model)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_mutating_buffers_are_rejected_without_mutating_the_model(self):
        class Mutating(BufferedModel):
            def forward(self, x):
                self.state.add_(1.0)
                return x * self.weight + self.state

        model = Mutating()
        before = model.state.clone()
        with self.assertRaisesRegex((ValueError, RuntimeError), "mutat"):
            self.capture(model)
        torch.testing.assert_close(model.state, before)

    def test_reduction_dtype_conversion_is_reported_before_encryption(self):
        class Converted(BufferedModel):
            def forward(self, x):
                return (x * self.weight).sum(dtype=torch.float64)

        captured = capture_training(
            Converted(),
            (torch.tensor([0.2, 0.5]),),
            torch.tensor(0.0, dtype=torch.float64),
            loss,
            input_visibility=("secret",),
        )
        program = compile_model(captured)
        self.assertFalse(program.supported)
        with self.assertRaisesRegex(NotImplementedError, "unsupported"):
            program.encrypt_inputs(
                TensorEngine(ReferenceVectorBackend()), public_bound=2.0
            )

    def test_grouped_convolution_is_reported_before_encryption(self):
        model = torch.nn.Conv2d(2, 2, 1, groups=2)
        program = compile_model(
            capture_training(
                model,
                (torch.ones(1, 2, 2, 2),),
                torch.zeros(1, 2, 2, 2),
                loss,
                input_visibility=("secret",),
            )
        )
        self.assertFalse(program.supported)
        self.assertTrue(
            any(
                "groups" in str(item["declined_candidates"])
                for item in program.kernel_requests()["requests"]
            )
        )

    def test_nonempty_tensor_requires_complete_ciphertext_coverage(self):
        engine = TensorEngine(ReferenceVectorBackend(), slots=8)
        with self.assertRaisesRegex(ValueError, "ciphertext"):
            engine._new([], (2,), torch.float32, 1.0)

    def test_mask_fusion_never_bypasses_selected_kernel_or_changes_its_input_type(self):
        from fhenomenon.ml import KernelRegistry, KernelDefinition, KernelMatch
        from fhenomenon.ml.interpreter import GraphInterpreter

        class Masked(BufferedModel):
            def forward(self, x):
                return torch.softmax(x * self.weight + self.state, dim=-1)

        for operation in ("aten.add.Tensor", "aten._softmax.default"):
            model = Masked()
            model.state[1] = -1e31
            captured = self.capture(model, buffer_visibility="public")
            called = []

            def execute(context, args, kwargs):
                self.assertIsInstance(args[0], Tensor)
                called.append(context.node.name)
                target = (
                    torch.ops.aten.add.Tensor
                    if operation == "aten.add.Tensor"
                    else torch.ops.aten._softmax.default
                )
                return GraphInterpreter(context.engine).lower(target, args, kwargs)

            registry = KernelRegistry()
            registry.register(
                KernelDefinition(
                    name="custom",
                    version=1,
                    operations=(operation,),
                    requirements={},
                    accepts=lambda *_: KernelMatch.accept(),
                    execute=execute,
                )
            )
            program = compile_model(captured, registry=registry)
            engine = TensorEngine(
                ReferenceVectorBackend(), slots=8, exact_reference=True
            )
            program.execute(engine, program.encrypt_inputs(engine, public_bound=2.0))
            self.assertTrue(called)
            self.assertEqual(program.execution_report["public_mask_fusions"], 0)
