"""Public, serializable contracts. These contain metadata, never tensor values."""

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any


@dataclass(frozen=True)
class TensorContract:
    shape: tuple[int, ...]
    dtype: str
    visibility: str
    layout: str = "logical_row_major"

    def to_dict(self):
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "visibility": self.visibility,
            "layout": self.layout,
        }


@dataclass(frozen=True)
class NodeContract:
    name: str
    operation: str
    phase: str
    inputs: tuple[Any, ...]
    output: Any
    visibility: str
    keywords: dict = field(default_factory=dict)

    @property
    def signature(self):
        # Phase and node names do not change the arithmetic interface.
        specification = {
            "operation": self.operation,
            "inputs": describe(self.inputs),
            "output": describe(self.output),
            "keywords": describe(self.keywords),
        }
        return hashlib.sha256(
            json.dumps(
                specification, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()


def describe(value):
    if isinstance(value, TensorContract):
        return value.to_dict()
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if type(value).__module__ == "torch" and type(value).__name__ in {
        "dtype",
        "device",
        "layout",
        "memory_format",
    }:
        return {"type": type(value).__name__, "value": str(value)}
    # Scalar operation arguments are public specialization data. No tensors
    # or arbitrary object reprs are allowed to sneak into an exported request.
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"type": type(value).__name__}
