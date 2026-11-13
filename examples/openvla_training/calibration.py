"""Optional empirical ranges observed only on reproducible PUBLIC fixtures.

These ranges are conditional assumptions, not cryptographic or mathematical
proofs. The producer regenerates public initialization and inputs independently
of captured private payloads. Runtime evaluation cannot update this profile.
"""

from __future__ import annotations

from .source import DEFAULT_SNAPSHOT

import argparse
import hashlib
import json
import math
import operator
from pathlib import Path
from types import CodeType

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, treespec_dumps

PUBLIC_SEEDS = (101, 102, 103, 104)
HELDOUT_SEED = 2029
_LAYOUT = {
    "aten.view.default",
    "aten._unsafe_view.default",
    "aten.reshape.default",
    "aten.permute.default",
    "aten.t.default",
    "aten.transpose.int",
    "aten.expand.default",
    "aten.unsqueeze.default",
    "aten.slice.Tensor",
    "aten.slice_backward.default",
    "aten.alias.default",
    "aten.detach.default",
    "aten.clone.default",
    "aten.cat.default",
    "aten.stack.default",
    "aten.split_with_sizes.default",
    "aten.unbind.int",
    "aten._to_copy.default",
    "aten.empty_like.default",
    "aten.zeros_like.default",
    "aten.ones_like.default",
    "aten.fill.Scalar",
}


def _eligible(node):
    if (
        str(node.target) == "aten.add.Tensor"
        and node.users
        and all(str(user.target) == "aten._softmax.default" for user in node.users)
        and any(
            isinstance(arg, torch.fx.Node)
            and arg.meta.get("training", {}).get("privacy") == "public"
            for arg in node.args
        )
    ):
        # The interpreter may fuse this public attention mask into an opaque
        # mask descriptor, so the eager sentinel tensor is not its output.
        return False
    return (
        node.op == "call_function"
        and node.meta.get("training", {}).get("privacy") == "secret"
        and node.target is not operator.getitem
        and str(node.target) not in _LAYOUT
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _tensor_digest(value):
    if value.device.type == "meta":
        raise ValueError("public calibration requires materialized public weights")
    data = (
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(data).hexdigest()


def _describe(value):
    from fhenomenon.ml.tensor import Tensor

    if isinstance(value, (torch.Tensor, Tensor)):
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    return {"kind": "literal", "type": type(value).__name__}


def _structure(value):
    flat, spec = tree_flatten(value)
    return {"tree_spec": treespec_dumps(spec), "leaves": [_describe(x) for x in flat]}


def _target(target):
    if (
        callable(target)
        and hasattr(target, "__qualname__")
        and not hasattr(target, "_schema")
    ):
        # The same helper is importable both by module and by run_graph.py.
        if target.__name__ == "rms_norm_backward":
            return "fhenomenon.ml.rms.rms_norm_backward"
        return target.__module__ + "." + target.__qualname__
    return str(target)


def _argument(value):
    if isinstance(value, torch.fx.Node):
        return {"node": value.name}
    if isinstance(value, (list, tuple)):
        return {
            "sequence": type(value).__name__,
            "items": [_argument(x) for x in value],
        }
    if isinstance(value, dict):
        return {
            str(k): _argument(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, torch.Tensor):
        return {**_describe(value), "public_constant_sha256": _tensor_digest(value)}
    if isinstance(value, float) and not math.isfinite(value):
        return {"public_float_constant": repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _implementation(code):
    def constant(value):
        if isinstance(value, CodeType):
            return _implementation(value)
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, tuple):
            return [constant(x) for x in value]
        return _argument(value)

    return {
        "bytecode": code.co_code.hex(),
        "constants": [constant(v) for v in code.co_consts],
        "names": code.co_names,
        "arguments": [
            code.co_argcount,
            code.co_kwonlyargcount,
            code.co_posonlyargcount,
        ],
    }


def binding_for(captured, args=None):
    """Read public graph/shape metadata and explicitly public weights only."""
    graph = []
    for node in captured.gm.graph.nodes:
        row = {
            "name": node.name,
            "op": node.op,
            "target": _target(node.target),
            "args": _argument(node.args),
            "kwargs": _argument(node.kwargs),
            "privacy": node.meta.get("training", {}).get("privacy"),
            "outputs": _structure(node.meta.get("val")),
        }
        if hasattr(node.target, "__code__"):
            row["implementation_sha256"] = _digest(
                {
                    "code": _implementation(node.target.__code__),
                    "defaults": _argument(node.target.__defaults__),
                    "keyword_defaults": _argument(node.target.__kwdefaults__),
                }
            )
        if node.op == "get_attr":
            value = captured.gm
            for part in node.target.split("."):
                value = getattr(value, part)
            if row["privacy"] != "public":
                raise ValueError(
                    "secret graph constants cannot enter a public calibration identity"
                )
            row["public_constant"] = _argument(value)
        graph.append(row)
    public_weights = []
    arguments = captured.args if args is None else args
    for role, values in (("frozen", arguments[1]), ("buffer", arguments[2])):
        for name, value in sorted(values.items()):
            public_weights.append(
                {
                    "role": role,
                    "name": name,
                    **_describe(value),
                    "sha256": _tensor_digest(value),
                }
            )
    return {
        "graph_sha256": _digest(graph),
        "public_weights_sha256": _digest(public_weights),
        "model_seed": captured.metadata.get("model_seed"),
        "profile": captured.metadata.get("profile"),
        "input_shapes": _structure(arguments),
    }


def calibrate_public_ranges(captured, seeds=PUBLIC_SEEDS, headroom=4.0, floor=0.01):
    """Produce a profile from named public fixtures; ignore private payloads."""
    from .graph import capture_training_graph, public_pixel_fixture

    if tuple(seeds) != PUBLIC_SEEDS:
        raise ValueError(
            "this calibration protocol uses only public seeds 101, 102, 103, 104"
        )
    if (
        not math.isfinite(headroom)
        or headroom < 1
        or not math.isfinite(floor)
        or floor <= 0
    ):
        raise ValueError("calibration headroom and floor must be finite and positive")
    settings = getattr(captured, "capture_settings", None)
    if not settings or settings.get("profile") not in {"micro", "tiny"}:
        raise ValueError(
            "public calibration requires reproducible materialized capture settings"
        )
    public = capture_training_graph(**settings, data_seed=PUBLIC_SEEDS[0])
    binding = binding_for(captured)
    if binding_for(public) != binding:
        raise ValueError(
            "graph or public weights differ from the independently regenerated public fixture"
        )
    nodes = {}

    class Observe(torch.fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            if _eligible(node):
                flat, spec = tree_flatten(value)
                entry = nodes.setdefault(
                    node.name,
                    {
                        "tree_spec": treespec_dumps(spec),
                        "leaves": [_describe(x) for x in flat],
                    },
                )
                if entry["tree_spec"] != treespec_dumps(spec):
                    raise ValueError("public calibration output structure changed")
                for item, leaf in zip(flat, entry["leaves"]):
                    if isinstance(item, torch.Tensor):
                        observed = (
                            float(item.detach().abs().max()) if item.numel() else 0.0
                        )
                        if not math.isfinite(observed):
                            raise ValueError(
                                "nonfinite value in a public calibration fixture"
                            )
                        leaf["observed_absmax"] = max(
                            leaf.get("observed_absmax", 0.0), observed
                        )
            return value

    with torch.no_grad():
        for seed in PUBLIC_SEEDS:
            args = list(public.args)
            pixels = public.args[4]
            args[4] = public_pixel_fixture(
                pixels.shape, dtype=pixels.dtype, data_seed=seed
            )
            Observe(public.gm).run(*args)
    for entry in nodes.values():
        for leaf in entry["leaves"]:
            if leaf["kind"] == "tensor":
                leaf["bound"] = max(floor, headroom * leaf["observed_absmax"])
    return {
        "schema_version": 1,
        "binding": binding,
        "nodes": nodes,
        "provenance": {
            "kind": "public_empirical_calibration",
            "guarantee": "empirical_not_proof",
            "training_data_used": False,
            "calibration_seeds": list(PUBLIC_SEEDS),
            "intended_validation_seed": HELDOUT_SEED,
            "headroom": headroom,
            "floor": floor,
            "model_seed": settings["seed"],
            "profile": settings["profile"],
            "inputs": "Fresh public seeded model/adapters, fixed public instruction/labels, independently seeded uniform pixels",
            "runtime_policy": "Read-only range assumptions; no calibration or secret range inspection during encrypted execution",
        },
    }


def validate_range_contract(contract, captured, args=None):
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError("unsupported public range contract schema")
    provenance = contract.get("provenance", {})
    if (
        provenance.get("kind") != "public_empirical_calibration"
        or provenance.get("guarantee") != "empirical_not_proof"
        or provenance.get("training_data_used") is not False
        or provenance.get("calibration_seeds") != list(PUBLIC_SEEDS)
    ):
        raise ValueError("invalid explicit public calibration provenance")
    binding = binding_for(captured, args=args)
    actual = contract.get("binding", {})
    for key, label in (
        ("graph_sha256", "graph"),
        ("public_weights_sha256", "public weights"),
        ("model_seed", "model seed"),
        ("profile", "model profile"),
        ("input_shapes", "input shapes"),
    ):
        if actual.get(key) != binding[key]:
            raise ValueError("public range contract does not match the " + label)
    if (
        provenance.get("model_seed") != binding["model_seed"]
        or provenance.get("profile") != binding["profile"]
    ):
        raise ValueError("public calibration provenance does not match the model")
    headroom, floor = provenance.get("headroom"), provenance.get("floor")
    if (
        not isinstance(headroom, (float, int))
        or not math.isfinite(headroom)
        or headroom < 1
        or not isinstance(floor, (float, int))
        or not math.isfinite(floor)
        or floor <= 0
    ):
        raise ValueError("calibration headroom and floor must be finite and positive")
    eligible = {node.name: node for node in captured.gm.graph.nodes if _eligible(node)}
    nodes = contract.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError(
            "public range contract requires explicit arithmetic node entries"
        )
    for name, entry in nodes.items():
        if name not in eligible:
            raise ValueError("range entry is not a secret arithmetic node: " + name)
        expected = _structure(eligible[name].meta.get("val"))
        if (
            not isinstance(entry, dict)
            or entry.get("tree_spec") != expected["tree_spec"]
            or len(entry.get("leaves", [])) != len(expected["leaves"])
        ):
            raise ValueError("range output structure does not match node " + name)
        for leaf, description in zip(entry["leaves"], expected["leaves"]):
            if any(leaf.get(k) != v for k, v in description.items()):
                raise ValueError(
                    "range output shape or dtype does not match node " + name
                )
            if leaf["kind"] == "tensor":
                bound, observed = leaf.get("bound"), leaf.get("observed_absmax")
                if (
                    not isinstance(bound, (int, float))
                    or not math.isfinite(bound)
                    or bound <= 0
                    or not isinstance(observed, (int, float))
                    or not math.isfinite(observed)
                    or observed < 0
                ):
                    raise ValueError("public node bounds must be finite and positive")
                if bound != max(floor, headroom * observed):
                    raise ValueError(
                        "public node bound does not match the declared calibration headroom"
                    )
    return contract


def load_range_contract(path, captured):
    path = Path(path).resolve()
    data = path.read_bytes()
    contract = json.loads(data)
    validate_range_contract(contract, captured)
    contract["source"] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    return contract


def apply_node_range(value, entry, engine):
    """Tighten public metadata; only explicit reference engines check values."""
    from fhenomenon.ml.tensor import Tensor

    flat, spec = tree_flatten(value)
    if treespec_dumps(spec) != entry["tree_spec"] or len(flat) != len(entry["leaves"]):
        raise ValueError("range contract output structure mismatch during execution")
    result = []
    for item, leaf in zip(flat, entry["leaves"]):
        if isinstance(item, Tensor):
            if (
                leaf["kind"] != "tensor"
                or list(item.shape) != leaf["shape"]
                or str(item.dtype) != leaf["dtype"]
            ):
                raise ValueError(
                    "range contract output tensor metadata mismatch during execution"
                )
            bound = min(item.public_bound, leaf["bound"])
            item = (
                engine.require_bound(item, bound) if bound < item.public_bound else item
            )
        elif isinstance(item, torch.Tensor):
            if (
                leaf["kind"] != "tensor"
                or list(item.shape) != leaf["shape"]
                or str(item.dtype) != leaf["dtype"]
            ):
                raise ValueError(
                    "range contract output tensor metadata mismatch during execution"
                )
            if (
                getattr(getattr(engine, "backend", engine), "kind", None)
                == "noncryptographic_reference"
            ):
                maximum = float(item.detach().abs().max()) if item.numel() else 0.0
                if (
                    not math.isfinite(maximum)
                    or maximum > leaf["bound"] * (1 + 1e-7) + 1e-10
                ):
                    raise ValueError("public bound violated in reference validation")
        elif leaf["kind"] == "tensor":
            raise ValueError("range contract expected a tensor output")
        result.append(item)
    return tree_unflatten(result, spec)


def main():
    from .graph import capture_training_graph

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("micro", "tiny"), default="micro")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    captured = capture_training_graph(
        args.snapshot, args.profile, data_seed=HELDOUT_SEED, max_report_nodes=0
    )
    contract = calibrate_public_ranges(captured)
    validate_range_contract(contract, captured)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "path": str(path.resolve()),
                "nodes": len(contract["nodes"]),
                "guarantee": "empirical_not_proof",
                "calibration_seeds": list(PUBLIC_SEEDS),
            }
        )
    )


if __name__ == "__main__":
    main()
