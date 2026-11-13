"""Semantic compilation before packing or primitive lowering."""

from collections import Counter, defaultdict
from contextlib import nullcontext
import copy
import hashlib
import json
import operator

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from .contracts import TensorContract, NodeContract, describe
from .interpreter import GraphInterpreter
from .registry import KernelRegistry, KernelContext, select_kernel, validate_output
from .tensor import Tensor


# The decomposition implementation remains separate from the contribution
# registry. A contribution may implement an otherwise unsupported operation.
DECOMPOSITIONS = frozenset(
    {
        "aten.add.Tensor",
        "aten.add.Scalar",
        "aten.sub.Tensor",
        "aten.sub.Scalar",
        "aten.mul.Tensor",
        "aten.mul.Scalar",
        "aten.div.Tensor",
        "aten.div.Scalar",
        "aten.neg.default",
        "aten.pow.Tensor_Scalar",
        "aten.mm.default",
        "aten.bmm.default",
        "aten.addmm.default",
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
        "aten.cat.default",
        "aten.stack.default",
        "aten.split_with_sizes.default",
        "aten.unbind.int",
        "aten.mean.dim",
        "aten.sum.dim_IntList",
        "aten.mean.default",
        "aten.sum.default",
        "aten.alias.default",
        "aten.detach.default",
        "aten.clone.default",
        "aten._to_copy.default",
        "aten.zeros_like.default",
        "aten.empty_like.default",
        "aten.ones_like.default",
        "aten.fill.Scalar",
        "aten.copy.default",
        "aten._softmax.default",
        "aten._log_softmax.default",
        "aten._softmax_backward_data.default",
        "aten._log_softmax_backward_data.default",
        "aten.native_layer_norm.default",
        "aten.native_layer_norm_backward.default",
        "aten.gelu.default",
        "aten.gelu_backward.default",
        "aten.silu.default",
        "aten.sigmoid.default",
        "aten.rsqrt.default",
        "aten.convolution.default",
        "aten.convolution_backward.default",
        "operator.getitem",
        "fhenomenon.ml.rms.rms_norm_backward",
    }
)


def operation_name(target):
    if target is operator.getitem:
        return "operator.getitem"
    if hasattr(target, "_schema"):
        return str(target)
    return f"{target.__module__}.{target.__qualname__}"


def tensor_contract(value, privacy):
    if isinstance(value, torch.Tensor):
        return TensorContract(tuple(value.shape), str(value.dtype), privacy)
    if isinstance(value, (tuple, list)):
        return tuple(tensor_contract(item, privacy) for item in value)
    return None


def argument_contract(value):
    if isinstance(value, torch.fx.Node):
        return tensor_contract(value.meta.get("val"), value.meta["training"]["privacy"])
    if isinstance(value, (tuple, list)):
        return tuple(argument_contract(item) for item in value)
    if isinstance(value, dict):
        return {key: argument_contract(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return tensor_contract(value, "public")
    return value


def decomposition_decline(node):
    """Known static limits; numeric domains remain separate runtime contracts."""
    if node.operation not in DECOMPOSITIONS:
        return "no registered decomposition"
    if node.operation.startswith("aten.convolution"):
        backward = "backward" in node.operation
        x = node.inputs[1 if backward else 0]
        group_index = 9 if backward else 8
        if node.inputs[group_index] != 1:
            return "convolution groups must equal one"
        if len(x.shape) != 4:
            return "only rank-4 two-dimensional convolution is supported"
        if node.inputs[group_index - 2] or any(node.inputs[group_index - 1]):
            return "transposed convolution requires a separate decomposition"
    if node.operation == "aten.pow.Tensor_Scalar":
        exponent = node.inputs[1]
        if exponent not in (-1, -0.5) and (
            not isinstance(exponent, (int, float))
            or exponent < 0
            or exponent != int(exponent)
        ):
            return "power requires a nonnegative integer or reciprocal/rsqrt exponent"
    if (
        node.operation in {"aten.gelu.default", "aten.gelu_backward.default"}
        and node.keywords.get("approximate", "none") != "none"
    ):
        return "GELU approximation requires a separate polynomial contract"

    def tensors(value):
        if isinstance(value, TensorContract):
            return [value]
        if isinstance(value, (tuple, list)):
            return [item for child in value for item in tensors(child)]
        return []

    private = [value for value in tensors(node.inputs) if value.visibility == "secret"]
    # Arithmetic keeps its declared semantic dtype. Precision conversion needs
    # its own numerical contract; CKKS approximate arithmetic is not a cast.
    if private and node.operation not in {
        "operator.getitem",
        "aten.zeros_like.default",
        "aten.empty_like.default",
        "aten.ones_like.default",
        "aten.fill.Scalar",
    }:
        if any(value.dtype != private[0].dtype for value in tensors(node.output)):
            return "secret dtype conversion requires an explicit engine policy"
    return None


def graph_identity(gm):
    """Bind contracts to topology and public metadata, without hashing values."""

    def edge(value):
        if isinstance(value, torch.fx.Node):
            return {"node": value.name}
        if isinstance(value, (tuple, list)):
            return [edge(item) for item in value]
        if isinstance(value, dict):
            return {key: edge(item) for key, item in value.items()}
        return describe(argument_contract(value))

    description = [
        {
            "name": node.name,
            "kind": node.op,
            "operation": operation_name(node.target)
            if node.op == "call_function"
            else node.target,
            "args": edge(node.args),
            "kwargs": edge(node.kwargs),
            "policy": node.meta.get("training"),
            "output": describe(tensor_contract(node.meta.get("val"), "unspecified")),
        }
        for node in gm.graph.nodes
    ]
    return hashlib.sha256(
        json.dumps(description, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class CompiledModel:
    def __init__(self, captured, registry, environment):
        self.captured = captured
        self.environment = copy.deepcopy(environment)
        self._definitions = registry.snapshot()
        self.nodes = []
        self._selected = {}
        self._declines = {}
        self.input_contracts = []
        self.execution_report = {}
        self.unsupported = []
        self._decomposition_reasons = {}
        self._graph_identity = graph_identity(captured.gm)
        for node in captured.gm.graph.nodes:
            info = node.meta.get("training")
            if not info or info.get("privacy") not in {"secret", "public"}:
                raise ValueError(f"node {node.name} lacks explicit privacy metadata")
            if node.op == "placeholder":
                specification = tensor_contract(node.meta.get("val"), info["privacy"])
                if not isinstance(specification, TensorContract):
                    raise ValueError("model placeholders must be tensors")
                if (
                    info["privacy"] == "secret"
                    and not node.meta["val"].is_floating_point()
                ):
                    raise ValueError(
                        "secret integer inputs need a declared oblivious encoding"
                    )
                self.input_contracts.append(
                    {
                        "name": node.name,
                        "role": info.get("role", "input"),
                        **specification.to_dict(),
                    }
                )
                continue
            if node.op != "call_function":
                if node.op not in {"output", "get_attr"}:
                    raise NotImplementedError(f"unsupported graph node kind {node.op}")
                continue
            contract = NodeContract(
                node.name,
                operation_name(node.target),
                info.get("phase", "unspecified"),
                tuple(argument_contract(arg) for arg in node.args),
                tensor_contract(node.meta.get("val"), info["privacy"]),
                info["privacy"],
                argument_contract(node.kwargs),
            )
            self.nodes.append(contract)
            if contract.visibility != "secret":
                continue
            selected, declined = select_kernel(
                contract, self.environment, self._definitions
            )
            self._selected[contract.name] = selected
            self._declines[contract.name] = declined
            reason = decomposition_decline(contract)
            self._decomposition_reasons[contract.name] = reason
            if selected is None and reason is not None:
                self.unsupported.append(contract)
                self._declines[contract.name].append(
                    {"kernel": "decomposition@1", "reason": reason}
                )
        self._by_name = {node.name: node for node in self.nodes}

    @property
    def supported(self):
        return not self.unsupported

    def _require_supported(self):
        if graph_identity(self.captured.gm) != self._graph_identity:
            raise ValueError(
                "captured graph changed after compilation; compile it again"
            )
        if self.unsupported:
            raise NotImplementedError(
                "unsupported secret operations: "
                + ", ".join(sorted({node.operation for node in self.unsupported}))
            )

    def encrypt_inputs(self, engine, *, public_bound, args=None, bounds=None):
        """Client-side input binding. Public/secret policy is fixed at compile time."""
        self._require_supported()
        leaves, spec = tree_flatten(self.captured.args if args is None else args)
        if spec != self.captured.gm._in_spec or len(leaves) != len(
            self.input_contracts
        ):
            raise ValueError("input tree differs from compiled contract")
        bound = []
        for value, contract in zip(leaves, self.input_contracts):
            if (
                tuple(value.shape) != tuple(contract["shape"])
                or str(value.dtype) != contract["dtype"]
            ):
                raise ValueError("input shape/dtype differs from compiled contract")
            limit = (
                public_bound
                if bounds is None
                else bounds.get(contract["name"], public_bound)
            )
            bound.append(
                engine.encrypt(value, public_bound=limit)
                if contract["visibility"] == "secret"
                else value
            )
        return tree_unflatten(bound, spec)

    def execute(
        self,
        engine,
        bound_args,
        *,
        scope_options=None,
        progress=None,
        range_contracts=None,
        range_policy=None,
    ):
        self.execution_report = {}
        self._require_supported()
        leaves, spec = tree_flatten(bound_args)
        if spec != self.captured.gm._in_spec or len(leaves) != len(
            self.input_contracts
        ):
            raise ValueError("input tree differs from compiled contract")
        for value, contract in zip(leaves, self.input_contracts):
            if (
                not hasattr(value, "shape")
                or tuple(value.shape) != tuple(contract["shape"])
                or str(value.dtype) != contract["dtype"]
            ):
                raise ValueError("input shape/dtype differs from compiled contract")
            if contract["visibility"] == "secret":
                if not isinstance(value, Tensor) or value.engine is not engine:
                    raise ValueError(
                        "secret input requires a Tensor from the same engine"
                    )
            elif not isinstance(value, torch.Tensor):
                raise ValueError("public input requires a plaintext torch.Tensor")
        if "runtime" in self.environment:
            from .kernels import runtime_environment

            if self.environment["runtime"] != runtime_environment(engine)["runtime"]:
                raise ValueError("runtime differs from compiled kernel environment")
        calls = Counter()

        def node_scope(node):
            origin = getattr(engine.backend, "semantic_node", None)
            contract = self._by_name[node.name]
            return (
                origin(node.name, contract.signature, self._implementation(node.name))
                if origin is not None
                else nullcontext()
            )

        def dispatch(node, args, kwargs, fallback):
            contract = self._by_name[node.name]
            kernel = self._selected.get(node.name)
            if kernel is None:
                result = fallback()
                validate_output(result, contract.output, engine, allow_public=True)
                return result
            result = kernel.execute(
                KernelContext(engine, contract, self.environment), args, kwargs
            )
            validate_output(result, contract.output, engine)
            calls[kernel.name] += 1
            return result

        interpreter = GraphInterpreter(
            engine,
            dispatcher=dispatch,
            range_policy=range_policy,
            node_scope=node_scope,
            mask_fusion_allowed=lambda node: (
                self._selected.get(node.name) is None
                and all(self._selected.get(user.name) is None for user in node.users)
            ),
        )

        def run():
            return interpreter.run(
                self.captured,
                bound_args,
                progress=progress,
                range_contracts=range_contracts,
            )

        reference = engine.backend.kind == "noncryptographic_reference"
        completed = False
        try:
            result = (
                run() if reference else engine.run_scope(run, **(scope_options or {}))
            )
            completed = True
            return result
        finally:
            self.execution_report = {
                **interpreter.stats,
                "kernel_calls": dict(calls),
                "semantic_graph_nodes": len(self.nodes),
                "encrypted_execution": not reference,
                "scope_timing": getattr(engine.backend, "last_scope_timing", None),
            }
            if not reference:
                statistics = getattr(engine.backend, "runtime_stats", {})
                self.execution_report["semantic_measurements"] = statistics.get(
                    "semantic_measurements", []
                )
                self.execution_report["scope_completed"] = (
                    completed
                    and statistics.get("last_scope", {}).get("status") == "completed"
                )

    def _implementation(self, name):
        kernel = self._selected[name]
        if kernel is not None:
            return f"{kernel.name}@{kernel.version}"
        return (
            "decomposition@1"
            if self._decomposition_reasons[name] is None
            else "unsupported"
        )

    def measurement(self):
        """Bind one successful profiled Scope to its exact engineering requests.

        This is synchronized provider wall time, not CUDA-event time. Packing
        at scope egress, graph validation and unmeasured work remain residual.
        Missing timing capability returns None rather than an invented estimate.
        """
        report = self.execution_report
        rows = report.get("semantic_measurements", [])
        timing = report.get("scope_timing")
        if not report.get("scope_completed") or not rows or not timing:
            return None
        from .costs import price_kernel_requests

        requests = self.kernel_requests()
        identity = price_kernel_requests(requests, None, [])
        evidence = {
            "schema_version": 1,
            "request_fingerprint": identity["request_fingerprint"],
            "environment": copy.deepcopy(self.environment),
            "timing_boundary": "synchronized_scope_execution",
            "attribution_kind": "exclusive_synchronized_wall",
            "sample_count": 1,
            "whole_scope_wall_seconds": timing["execution_and_finalize_seconds"],
            "node_measurements": [
                {
                    "node": row["name"],
                    "signature": row["signature"],
                    "current_implementation": row["current_implementation"],
                    "seconds": row["seconds"],
                }
                for row in rows
            ],
        }
        price_kernel_requests(requests, evidence, [])
        return evidence

    def kernel_requests(self):
        """Metadata-only engineering requests; architecture disclosure is explicit."""
        groups = defaultdict(list)
        for node in self.nodes:
            if node.visibility == "secret":
                groups[node.signature].append(node)
        requests = []
        for signature, nodes in sorted(groups.items()):
            node = nodes[0]
            kernel = self._selected[node.name]
            implementations = {
                item.name: self._implementation(item.name) for item in nodes
            }
            homogeneous = len(set(implementations.values())) == 1
            requests.append(
                {
                    "signature": signature,
                    "operation": node.operation,
                    "occurrences": len(nodes),
                    "nodes": [item.name for item in nodes],
                    "input_contracts": describe(node.inputs),
                    "keyword_contracts": describe(node.keywords),
                    "output_contract": describe(node.output),
                    "current_implementation": next(iter(implementations.values()))
                    if homogeneous
                    else "mixed",
                    "current_implementations": implementations,
                    "selected_requirements": copy.deepcopy(kernel.requirements)
                    if kernel
                    else None,
                    "declined_candidates": self._declines.get(node.name, []),
                    "acceptance": {
                        "private_host_fallbacks": 0,
                        "intermediate_decryptions": 0,
                        "preserve_output_shape_and_dtype": True,
                        "preserve_intermediate_liveouts": True,
                        "declare_layout_scale_level_and_key_requirements": True,
                        "declare_workspace_and_retained_bytes": True,
                        "numerical_tolerances": "must be supplied by the application; no default accuracy claim",
                    },
                }
            )
        return {
            "schema_version": 1,
            "environment": copy.deepcopy(self.environment),
            "graph_identity": self._graph_identity,
            "disclosure": "public-to-executor graph geometry and contracts; review before sharing outside the model owner",
            "contains_model_weights_or_training_values": False,
            "latency_estimates": "unpriced_without_matching_measurements",
            "requests": requests,
        }


def compile_model(captured, *, registry=None, environment=None):
    return CompiledModel(captured, registry or KernelRegistry(), environment or {})
