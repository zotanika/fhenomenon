"""Capture ordinary PyTorch modules without accessing a model repository."""

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch.func import functional_call, functionalize, grad_and_value
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_map


@dataclass
class CapturedModel:
    model: torch.nn.Module
    gm: torch.fx.GraphModule
    args: tuple[Any, ...]
    trainable_names: tuple[str, ...]
    eager_result: Any
    metadata: dict


def capture_training(
    model,
    example_inputs,
    targets,
    loss_fn,
    *,
    learning_rate=1e-3,
    input_visibility,
    frozen_visibility="public",
    target_visibility="secret",
    buffer_visibility="secret",
    constant_visibility="secret",
):
    """Capture forward, full activation/parameter backward, and trainable SGD.

    ``loss_fn(prediction, targets)`` returns a scalar. Existing requires_grad
    flags select the parameters to update. Training inputs are tensor pytrees;
    visibility has the identical pytree. This is a local capture operation: no
    model weights, source, training inputs or targets leave the caller.
    """
    if not isinstance(model, torch.nn.Module) or not callable(loss_fn):
        raise TypeError("expected a PyTorch module and loss callable")
    if not isinstance(example_inputs, tuple):
        raise TypeError("example_inputs must be positional arguments in a tuple")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    _, input_tree = tree_flatten(example_inputs)
    visibility, visibility_tree = tree_flatten(input_visibility)
    if input_tree != visibility_tree:
        raise ValueError("input_visibility must have the input pytree structure")
    if any(
        v not in {"public", "secret"}
        for v in (
            *visibility,
            frozen_visibility,
            target_visibility,
            constant_visibility,
        )
    ):
        raise ValueError("visibility must be public or secret")
    trainable = {
        name: parameter.detach().clone().requires_grad_(True)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    fixed = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    buffer_policy = (
        {name: buffer_visibility for name in buffers}
        if isinstance(buffer_visibility, str)
        else dict(buffer_visibility)
    )
    if set(buffer_policy) != set(buffers) or any(
        value not in {"secret", "public"} for value in buffer_policy.values()
    ):
        raise ValueError("buffer_visibility must declare every registered buffer")
    if not trainable:
        raise ValueError("training requires at least one requires_grad parameter")

    def loss(parameters, constants, state, inputs, labels):
        prediction = functional_call(
            model, ({**constants, **parameters}, state), inputs
        )
        return loss_fn(prediction, labels)

    def step(parameters, constants, state, inputs, labels):
        gradients, objective = grad_and_value(loss)(
            parameters, constants, state, inputs, labels
        )
        updated = {
            name: parameters[name] - learning_rate * gradient
            for name, gradient in gradients.items()
        }
        return objective, gradients, updated

    def copy_input(value):
        if not isinstance(value, torch.Tensor):
            raise TypeError("model inputs and targets must be tensor pytrees")
        return value.detach().clone()

    args = (
        trainable,
        fixed,
        buffers,
        tree_map(copy_input, example_inputs),
        tree_map(copy_input, targets),
    )
    leaves, _ = tree_flatten(args)
    versions = [value._version for value in leaves]
    cuda_devices = sorted(
        {value.device.index for value in leaves if value.device.type == "cuda"}
    )
    # Capture may execute the model's Python twice. It must not advance the
    # caller's RNG or silently train a different state on those two executions.
    with torch.random.fork_rng(devices=cuda_devices):
        eager = step(*args)
        if [value._version for value in leaves] != versions:
            raise ValueError(
                "state-mutating training requires an explicit state-transition contract"
            )
        gm = make_fx(functionalize(step), record_module_stack=True)(*args)
        if [value._version for value in leaves] != versions:
            raise ValueError(
                "state-mutating training requires an explicit state-transition contract"
            )
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ()):
            raise ValueError(
                "random training operations require a declared randomness contract"
            )
        if getattr(getattr(node.target, "_schema", None), "is_mutable", False):
            raise ValueError(
                "state-mutating training requires an explicit state-transition contract"
            )
    gm.graph.eliminate_dead_code()
    gm.recompile()
    roles = [("trainable", name, "secret") for name in trainable]
    roles += [("frozen_weight", name, frozen_visibility) for name in fixed]
    roles += [("buffer", name, buffer_policy[name]) for name in buffers]
    roles += [("input", str(i), value) for i, value in enumerate(visibility)]
    roles += [
        ("target", str(i), target_visibility)
        for i, _ in enumerate(tree_flatten(targets)[0])
    ]
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(roles):
        raise ValueError("captured placeholder structure differs from declared inputs")
    role_iter = iter(roles)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            role, name, visibility = next(role_iter)
            info = {
                "role": role,
                "parameter_or_input": name,
                "privacy": visibility,
                "phase": "input",
            }
        elif node.op == "get_attr":
            if constant_visibility != "public":
                raise ValueError(
                    "captured tensor constants require explicit constant_visibility='public' or promotion to model inputs"
                )
            info = {"role": "constant", "privacy": "public", "phase": "input"}
        else:
            secret = any(
                source.meta["training"]["privacy"] == "secret"
                for source in node.all_input_nodes
            )
            info = {
                "role": "operation",
                "privacy": "secret" if secret else "public",
                "phase": "training",
            }
        node.meta["training"] = info
    return CapturedModel(
        model,
        gm,
        args,
        tuple(trainable),
        eager,
        {
            "schema_version": 1,
            "model_class": type(model).__name__,
            "scope": "forward, full backward, trainable SGD",
            "input_visibility": roles,
            "encrypted_execution_performed": False,
        },
    )
