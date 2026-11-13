"""Run an ordinary module through the semantic compiler's explicit reference backend.

This checks model adaptation and kernel requests. It is not encryption evidence.
An application can instead supply a TensorEngine backed by its runtime provider.
"""

import json

import torch

from fhenomenon.ml import (
    ReferenceVectorBackend,
    TensorEngine,
    capture_training,
    compile_model,
    native_tensor_registry,
    runtime_environment,
)


class AdaptedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(3, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.down = torch.nn.Parameter(torch.tensor([[0.1, 0.2, 0.1]]))
        self.up = torch.nn.Parameter(torch.tensor([[0.2], [0.1]]))

    def forward(self, observation):
        return self.base(observation) + observation @ self.down.T @ self.up.T


def build_program(engine):
    torch.manual_seed(14)
    captured = capture_training(
        AdaptedLinear(),
        (torch.tensor([[0.2, 0.1, 0.3]]),),
        torch.tensor([[0.15, 0.2]]),
        lambda prediction, target: ((prediction - target) ** 2).mean(),
        input_visibility=("secret",),
        frozen_visibility="public",  # Explicitly permit the executor to see the base weights.
        learning_rate=0.01,
    )
    return compile_model(
        captured,
        registry=native_tensor_registry(),
        environment=runtime_environment(engine),
    )


if __name__ == "__main__":
    engine = TensorEngine(ReferenceVectorBackend(), slots=16, exact_reference=True)
    program = build_program(engine)
    outputs = program.execute(engine, program.encrypt_inputs(engine, public_bound=2.0))
    requests = program.kernel_requests()
    print(
        json.dumps(
            {
                "mode": "noncryptographic_reference",
                "loss": float(engine.decrypt(outputs[0])),
                "updated_parameters": sorted(outputs[2]),
                "semantic_operations": len(program.nodes),
                "engineering_requests": len(requests["requests"]),
                "latency_estimates": requests["latency_estimates"],
            },
            indent=2,
        )
    )
