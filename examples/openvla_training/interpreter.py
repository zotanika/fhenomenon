"""Application range-policy adapter for the shared Fhenomenon interpreter."""

from types import SimpleNamespace
from fhenomenon.ml.interpreter import GraphInterpreter as _Interpreter, _MaskedAttention
from examples.openvla_training.calibration import (
    validate_range_contract,
    apply_node_range,
)

RANGE_POLICY = SimpleNamespace(validate=validate_range_contract, apply=apply_node_range)


class GraphInterpreter(_Interpreter):
    def __init__(self, engine, **options):
        options.setdefault("range_policy", RANGE_POLICY)
        super().__init__(engine, **options)
