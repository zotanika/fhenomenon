"""Replace semantic operations without editing the graph interpreter."""

from dataclasses import dataclass
from typing import Callable
import copy
import math
import torch
import numpy as np

from .contracts import NodeContract, TensorContract
from .tensor import Tensor


@dataclass(frozen=True)
class KernelMatch:
    accepted: bool
    reason: str = ""

    @classmethod
    def accept(cls):
        return cls(True)

    @classmethod
    def decline(cls, reason):
        return cls(False, str(reason))


@dataclass(frozen=True)
class KernelDefinition:
    name: str
    version: int
    operations: tuple[str, ...]
    requirements: dict
    accepts: Callable
    execute: Callable
    priority: int = 0


@dataclass(frozen=True)
class KernelContext:
    engine: object
    node: NodeContract
    environment: dict

    def packed_input(self, value, *, minimum_level=0):
        if not isinstance(value, Tensor) or value.engine is not self.engine:
            raise ValueError(
                "kernel input must be a tensor owned by the executing engine"
            )
        return tuple(
            self.engine._ready(
                chunk, depth=minimum_level, public_bound=value.public_bound
            )
            for chunk in self.engine._pack(value)
        )

    def output(self, chunks, shape, dtype, public_bound):
        return self.engine._new(chunks, shape, dtype, public_bound)

    def native(self, operation, inputs, attributes):
        return self.engine.backend.tensor_kernel(operation, inputs, attributes)


class KernelRegistry:
    def __init__(self):
        self._definitions = []

    def register(self, definition):
        if not isinstance(definition, KernelDefinition):
            raise TypeError("register requires a KernelDefinition")
        if (
            not definition.name
            or type(definition.version) is not int
            or definition.version <= 0
            or not definition.operations
            or not all(isinstance(op, str) and op for op in definition.operations)
            or not callable(definition.accepts)
            or not callable(definition.execute)
        ):
            raise ValueError("kernel requires an identity, operations and callbacks")
        if any(item.name == definition.name for item in self._definitions):
            raise ValueError("kernel name already registered")
        self._definitions.append(
            KernelDefinition(
                definition.name,
                definition.version,
                tuple(definition.operations),
                copy.deepcopy(definition.requirements),
                definition.accepts,
                definition.execute,
                definition.priority,
            )
        )
        self._definitions.sort(key=lambda item: -item.priority)

    def snapshot(self):
        return tuple(self._definitions)


def select_kernel(node, environment, definitions):
    declines = []
    for definition in definitions:
        if node.operation not in definition.operations:
            continue
        match = definition.accepts(node, environment)
        if not isinstance(match, KernelMatch):
            raise TypeError("kernel matcher must return KernelMatch")
        if match.accepted:
            return definition, declines
        declines.append({"kernel": definition.name, "reason": match.reason})
    return None, declines


def validate_output(value, contract, engine, *, allow_public=False):
    if isinstance(contract, TensorContract):
        # A trusted decomposition can construct constants from public shapes
        # (zeros_like, scratch initialization), without reading secret values.
        if allow_public and isinstance(value, torch.Tensor):
            if (
                tuple(value.shape) != contract.shape
                or str(value.dtype) != contract.dtype
            ):
                raise ValueError(
                    "decomposition output violates the semantic shape/dtype contract"
                )
            return
        if not isinstance(value, Tensor) or value.engine is not engine:
            raise ValueError(
                "kernel returned a plaintext or foreign tensor for a secret output"
            )
        if tuple(value.shape) != contract.shape or str(value.dtype) != contract.dtype:
            raise ValueError(
                f"kernel output violates the semantic shape/dtype contract: got {value.shape}/{value.dtype}, expected {contract.shape}/{contract.dtype}"
            )
        if not math.isfinite(value.public_bound) or value.public_bound < 0:
            raise ValueError("kernel output needs a finite public magnitude bound")
        if np.any(value.indices < -1) or np.any(
            value.indices >= len(value.chunks) * engine.slots
        ):
            raise ValueError("kernel output lacks ciphertext coverage for its layout")
        if any(chunk.count != engine.slots for chunk in value.chunks):
            raise ValueError(
                "kernel output ciphertext packing differs from the engine contract"
            )
    elif isinstance(contract, (tuple, list)):
        if not isinstance(value, (tuple, list)) or len(value) != len(contract):
            raise ValueError("kernel output structure differs from its contract")
        for item, spec in zip(value, contract):
            validate_output(item, spec, engine, allow_public=allow_public)
    elif contract is None and value is not None:
        raise ValueError("kernel output differs from the declared empty output")
