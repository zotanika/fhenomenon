"""Model adapters for public tensor-operation contracts.

The implementation of each native operation belongs to the runtime provider.
This module imports no arithmetic library and contains no native ciphertext ABI.
"""

import copy
import math

from .contracts import TensorContract
from .registry import KernelDefinition, KernelMatch, KernelRegistry
from .tensor import Tensor, array


def runtime_environment(engine):
    """Public capability snapshot used for selection and execution validation."""
    backend = engine.backend
    profile = copy.deepcopy(getattr(backend, "profile", {}))
    # Instance handles do not describe a device or cryptographic parameter set.
    # Input ownership is checked separately on every execution.
    profile.pop("context_id", None)
    return {
        "runtime": {
            "backend_kind": backend.kind,
            "slots": engine.slots,
            "refresh_target": engine.target_level,
            "profile": profile,
            "implementation_identity": copy.deepcopy(
                getattr(backend, "implementation_identity", {})
            ),
            "device": copy.deepcopy(getattr(backend, "device_info", {})),
            "kernel_catalog": copy.deepcopy(
                getattr(backend, "kernel_catalog", {"kernels": []})
            ),
        }
    }


def _linear_match(node, environment):
    runtime = environment.get("runtime", {})
    available = runtime.get("kernel_catalog", {}).get("kernels", [])
    if not any(entry.get("operation") == "tensor.public_linear" for entry in available):
        return KernelMatch.decline(
            "runtime catalog has no tensor.public_linear implementation"
        )
    index = 1 if node.operation == "aten.addmm.default" else 0
    left, right = node.inputs[index : index + 2]
    if not all(
        isinstance(value, TensorContract) and len(value.shape) == 2
        for value in (left, right)
    ):
        return KernelMatch.decline("requires two rank-2 tensors")
    if left.visibility != "secret" or right.visibility != "public":
        return KernelMatch.decline(
            "requires encrypted activations and explicitly public weights"
        )
    rows, inner = left.shape
    if (
        inner != right.shape[0]
        or min(rows, inner, right.shape[1]) < 1
        or max(math.prod(left.shape), rows * right.shape[1]) > runtime.get("slots", 0)
    ):
        return KernelMatch.decline(
            "input and output must fit one declared packed ciphertext"
        )
    if left.dtype != right.dtype or left.dtype not in {
        "torch.float32",
        "torch.float64",
    }:
        return KernelMatch.decline(
            "requires matching float32 or float64 semantic tensors"
        )
    return KernelMatch.accept()


def _linear_execute(context, args, kwargs):
    index = 1 if context.node.operation == "aten.addmm.default" else 0
    left, weights = args[index : index + 2]
    if isinstance(weights, Tensor):
        raise ValueError("public linear kernel cannot expose encrypted weights")
    coefficients = array(weights)
    rows, inner = left.shape
    columns = coefficients.shape[1]
    packed = context.packed_input(left, minimum_level=1)
    if len(packed) != 1:
        raise ValueError("public linear kernel requires one packed ciphertext")
    outputs = context.native(
        "tensor.public_linear",
        packed,
        {
            "rows": rows,
            "in_features": inner,
            "out_features": columns,
            "weights": coefficients.ravel().tolist(),
        },
    )
    if len(outputs) != 1:
        raise ValueError("public linear provider returned the wrong arity")
    bound = inner * left.public_bound * float(abs(coefficients).max())
    result = context.output(outputs, (rows, columns), left.dtype, bound)
    if index:
        alpha, beta = kwargs.get("alpha", 1), kwargs.get("beta", 1)
        if alpha != 1:
            result = context.engine.mul(result, alpha)
        if beta != 0:
            bias = args[0] if beta == 1 else context.engine.mul(args[0], beta)
            result = context.engine.add(bias, result)
    return result


def native_tensor_registry():
    """Register semantic adapters; unsupported providers retain decompositions."""
    registry = KernelRegistry()
    registry.register(
        KernelDefinition(
            name="native.tensor.public_linear",
            version=1,
            operations=("aten.mm.default", "aten.addmm.default"),
            requirements={
                "native_operation": "tensor.public_linear",
                "scheme": "ckks",
                "layout": "one row-major prefix, zero-padded output, slots representation",
                "input_level_minimum": 1,
                "native_level_consumption": 1,
                "keys": "complete power-of-two rotation basis for the ciphertext period",
                "packing": "materialization inside the same Scope; may add rotations and masks",
                "addmm": "alpha/beta scaling and bias addition remain explicit scoped operations",
                "memory": "provider must infer and reserve output and temporary storage before execution",
                "performance": "coarse runtime operation; no fused CUDA launch or speedup claim",
            },
            accepts=_linear_match,
            execute=_linear_execute,
        )
    )
    return registry
