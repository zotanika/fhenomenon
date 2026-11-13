"""Explicit full-graph ATen lowering onto a runtime-backed tensor engine.

Only nodes classified public by graph capture may execute host PyTorch. Secret
operations have an explicit implementation below and fail closed otherwise.
This layer has no encryption, decryption, ciphertext refresh, or key API.
"""

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import math
import operator
import weakref

import torch
from torch.fx.node import map_arg
from torch.utils._pytree import tree_flatten, tree_unflatten
from fhenomenon.ml.rms import rms_norm_backward


@dataclass(frozen=True)
class _MaskedAttention:
    scores: object
    keep_mask: torch.Tensor


class GraphInterpreter:
    def __init__(
        self,
        engine,
        *,
        dispatcher=None,
        range_policy=None,
        node_scope=None,
        mask_fusion_allowed=None,
    ):
        self.engine = engine
        self.dispatcher = dispatcher
        self.range_policy = range_policy
        self.node_scope = node_scope or (lambda node: nullcontext())
        self.mask_fusion_allowed = mask_fusion_allowed or (lambda node: True)
        self.stats = {}
        self._softmax_proofs = {}
        self._norm_proofs = {}
        self._rms_proofs = {}

    def _require_bound(self, value, bound):
        # Exact-arithmetic invariants of the declared polynomial algorithms.
        # The tensor engine validates these on its explicit reference path;
        # CKKS numerical error is a separate requirement, never a secret probe.
        enforce = getattr(self.engine, "require_bound", None)
        return value if enforce is None else enforce(value, bound)

    @staticmethod
    def _bound(value):
        from fhenomenon.ml.tensor import Tensor

        if isinstance(value, Tensor):
            return value.public_bound
        public = torch.as_tensor(value)
        return float(public.abs().max()) if public.numel() else 0.0

    def run(self, captured, args=None, progress=None, range_contracts=None):
        """Run with engine-wrapped secret input leaves; return encrypted outputs.

        An explicitly marked plaintext reference engine is usable for validation, but
        this interpreter does not label that execution encrypted. The caller
        owns input encryption and the policy for authorized final output.
        """
        gm = captured.gm
        range_entries = {}
        self._softmax_proofs.clear()
        self._norm_proofs.clear()
        self._rms_proofs.clear()
        flat, spec = tree_flatten(captured.args if args is None else args)
        if spec != gm._in_spec:
            raise ValueError(
                "input pytree does not match the captured parameter and data layout"
            )
        nodes = list(gm.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        if len(placeholders) != len(flat):
            raise ValueError(
                "input leaf count does not match the captured placeholders"
            )
        reference = (
            getattr(getattr(self.engine, "backend", self.engine), "kind", None)
            == "noncryptographic_reference"
        )
        if not reference:
            from fhenomenon.ml.tensor import Tensor

            # Validate the entire input tree before executing even public nodes.
            # In particular, omitted args must not expose captured plaintext as
            # secret inputs to an otherwise cryptographic engine.
            for node, value in zip(placeholders, flat):
                if node.meta.get("training", {}).get("privacy") == "secret" and (
                    not isinstance(value, Tensor) or value.engine is not self.engine
                ):
                    raise ValueError(
                        f"secret placeholder {node.name} requires an encrypted Tensor from the same engine"
                    )
        if range_contracts is not None:
            if self.range_policy is None:
                raise ValueError(
                    "explicit range contracts require a validating range policy"
                )
            self.range_policy.validate(range_contracts, captured, args=args)
            range_entries = range_contracts["nodes"]
        inputs = iter(flat)
        env = {}
        remaining_users = {node: len(node.users) for node in nodes}
        private = Counter()
        public = Counter()
        self.stats = {
            "private_host_fallback_count": 0,
            "public_mask_fusions": 0,
            "peak_live_graph_values": 0,
            "rms_bound_fusions": 0,
            "empirical_range_applications": 0,
            "empirical_range_contract_enabled": range_contracts is not None,
        }
        for index, node in enumerate(nodes, start=1):
            info = node.meta.get("training")
            if info is None or info.get("privacy") not in {"public", "secret"}:
                raise ValueError(f"node {node.name} lacks reviewed privacy metadata")
            if node.op == "placeholder":
                env[node] = next(inputs)
            elif node.op == "get_attr":
                value = gm
                for part in node.target.split("."):
                    value = getattr(value, part)
                if info["privacy"] != "public":
                    raise ValueError(
                        "secret get_attr constants must be explicit encrypted inputs"
                    )
                env[node] = value
            elif node.op == "call_function":
                positional = map_arg(node.args, lambda n: env[n])
                keyword = map_arg(node.kwargs, lambda n: env[n])
                if info["privacy"] == "public":
                    public[str(node.target)] += 1
                    env[node] = node.target(*positional, **keyword)
                else:
                    private[str(node.target)] += 1
                    with self.node_scope(node):
                        fused = self._public_attention_mask(node, positional)
                        if fused is not None:
                            env[node] = fused
                        else:
                            fallback = lambda: self.lower(
                                node.target, positional, keyword
                            )
                            env[node] = (
                                self.dispatcher(node, positional, keyword, fallback)
                                if self.dispatcher is not None
                                else fallback()
                            )
                        env[node] = self._rms_norm_bound(node, positional, env[node])
                        if node.name in range_entries:
                            original = env[node]
                            env[node] = self.range_policy.apply(
                                original, range_entries[node.name], self.engine
                            )
                            self._transfer_provenance(original, env[node])
                            self.stats["empirical_range_applications"] += 1
            elif node.op == "output":
                output = map_arg(node.args[0], lambda n: env[n])
                self.stats.update(
                    {
                        "secret_node_count": sum(private.values()),
                        "public_node_count": sum(public.values()),
                        "secret_operators": dict(sorted(private.items())),
                        "public_operators": dict(sorted(public.items())),
                    }
                )
                if progress is not None:
                    progress(index, len(nodes), node)
                return tree_unflatten(output, gm._out_spec)
            else:
                raise NotImplementedError(f"unsupported graph node kind {node.op}")
            self.stats["peak_live_graph_values"] = max(
                self.stats["peak_live_graph_values"], len(env)
            )
            # Drop this interpreter's references at last use. Immutable engine
            # views retain any underlying chunks needed by their own users.
            for source in node.all_input_nodes:
                remaining_users[source] -= 1
                if remaining_users[source] == 0:
                    env.pop(source, None)
            if remaining_users[node] == 0:
                env.pop(node, None)
            if progress is not None:
                progress(index, len(nodes), node)
        raise ValueError("graph has no output")

    def _transfer_provenance(self, original, replacement):
        """A public bound override changes wrappers, not the underlying values."""
        old, _ = tree_flatten(original)
        new, _ = tree_flatten(replacement)
        changed = {id(a): (a, b) for a, b in zip(old, new) if a is not b}
        if not changed:
            return

        def update(reference):
            value = reference()
            pair = changed.get(id(value))
            return (
                weakref.ref(pair[1])
                if pair is not None and pair[0] is value
                else reference
            )

        for proof in list(self._softmax_proofs.values()):
            reference = update(proof[0])
            if reference() is not None:
                self._softmax_proofs[id(reference())] = (reference, proof[1])
        for proof in list(self._norm_proofs.values()):
            references = tuple(update(ref) for ref in proof[:3])
            if references[0]() is not None:
                self._norm_proofs[id(references[0]())] = (*references, proof[3])
        for proof in list(self._rms_proofs.values()):
            references = tuple(update(ref) for ref in proof[:2])
            if all(ref() is not None for ref in references):
                self._rms_proofs[(id(references[0]()), id(references[1]()))] = (
                    *references,
                    proof[2],
                )

    def _rms_norm_bound(self, node, args, value):
        """Recognize x * rsqrt(mean(x**2)+epsilon), using graph identity.

        General products involving rsqrt, including backward products with a
        different source, must keep their ordinary multiplication bounds.
        """

        def is_op(candidate, name):
            return (
                isinstance(candidate, torch.fx.Node) and str(candidate.target) == name
            )

        if str(node.target) != "aten.mul.Tensor" or len(node.args) != 2:
            return value
        for root_index, inverse_index in ((0, 1), (1, 0)):
            root, inverse = node.args[root_index], node.args[inverse_index]
            if not is_op(inverse, "aten.rsqrt.default"):
                continue
            offset = inverse.args[0]
            if (
                not is_op(offset, "aten.add.Tensor")
                or len(offset.args) != 2
                or offset.kwargs.get("alpha", 1) != 1
            ):
                continue
            mean, epsilon = offset.args
            if (
                not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon < 0
            ):
                continue
            if (
                not is_op(mean, "aten.mean.dim")
                or len(mean.args) < 3
                or mean.args[2] is not True
            ):
                continue
            square, dims = mean.args[:2]
            if not is_op(square, "aten.pow.Tensor_Scalar") or square.args != (root, 2):
                continue
            shape = args[root_index].shape
            if dims is None or len(dims) == 0:
                dims = tuple(range(len(shape)))
            if not dims or any(
                not isinstance(dim, int) or not -len(shape) <= dim < len(shape)
                for dim in dims
            ):
                continue
            width = math.prod(shape[dim] for dim in {dim % len(shape) for dim in dims})
            # sum(x_i**2) = N*mean(x_i**2), with nonnegative epsilon
            # and the declared Newton rsqrt undershoot invariant.
            self.stats["rms_bound_fusions"] += 1
            if hasattr(self.engine, "require_bound"):
                original, reciprocal = args[root_index], args[inverse_index]
                self._rms_proofs[(id(original), id(reciprocal))] = (
                    weakref.ref(original),
                    weakref.ref(reciprocal),
                    tuple(sorted({dim % len(shape) for dim in dims})),
                )
            return self._require_bound(value, math.sqrt(width))
        return value

    def _public_attention_mask(self, node, args):
        if not self.mask_fusion_allowed(node):
            return None
        if str(node.target) != "aten.add.Tensor" or len(args) < 2:
            return None
        if node.kwargs.get("alpha", 1) != 1:
            return None
        if not node.users or any(
            str(user.target) != "aten._softmax.default" for user in node.users
        ):
            return None
        for index in (0, 1):
            source = node.args[index]
            if (
                not isinstance(source, torch.fx.Node)
                or source.meta["training"]["privacy"] != "public"
            ):
                continue
            value = args[index]
            if not isinstance(value, torch.Tensor):
                continue
            is_blocked = value < -1e30
            if bool(is_blocked.any()) and bool(((value == 0) | is_blocked).all()):
                self.stats["public_mask_fusions"] += 1
                return _MaskedAttention(args[1 - index], (~is_blocked).to(value.dtype))
        return None

    def _softmax(self, x, dim, logarithmic=False):
        from fhenomenon.ml.approximations import CONTRACTS

        e = self.engine
        scores = x.scores if isinstance(x, _MaskedAttention) else x
        score_bound = min(self._bound(scores), 8.0)
        if isinstance(x, _MaskedAttention):
            if logarithmic:
                raise NotImplementedError(
                    "masked logarithmic softmax is not used by this training graph"
                )
            # The mask and tensor shape are public; encrypted score values are
            # never inspected to select this interval or its iteration count.
            keep = torch.broadcast_to(x.keep_mask, tuple(scores.shape))
            if not bool(((keep == 0) | (keep == 1)).all()):
                raise ValueError("softmax requires a public binary keep mask")
            counts = keep.sum(dim=dim)
            minimum_count = float(counts.min()) if counts.numel() else 0.0
            maximum_count = float(counts.max()) if counts.numel() else 0.0
            if minimum_count <= 0:
                raise ValueError(
                    "softmax requires at least one unmasked score in every row"
                )
            exponentials = e.mul(e.nonlinear("exp", (x.scores,)), x.keep_mask)
        else:
            minimum_count = maximum_count = int(scores.shape[dim])
            if minimum_count <= 0:
                raise ValueError(
                    "softmax requires at least one unmasked score in every row"
                )
            exponentials = e.nonlinear("exp", (x,))
        denominator = e.sum(exponentials, [dim], True)
        # This margin covers the ideal exp-polynomial relative error on [-8,8].
        # CKKS error remains a separate numerical requirement. Intersecting
        # with the predeclared global domain retains that existing requirement;
        # it does not prove that an arbitrary encrypted denominator meets it.
        positive_name = "log" if logarithmic else "reciprocal"
        global_low, global_high = CONTRACTS[positive_name]["interval"]
        low = max(global_low, minimum_count * math.exp(-score_bound) * (1 - 1e-7))
        high = min(global_high, maximum_count * math.exp(score_bound) * (1 + 1e-7))
        if low > high:
            raise ValueError(
                "softmax denominator has no supported public positive interval"
            )
        if logarithmic:
            return e.sub(x, e.nonlinear("log", (denominator, low, high)))
        # Positive polynomial exp and the reciprocal Newton undershoot imply
        # 0 <= p_i <= 1 and sum(p_i) <= 1 on their declared positive domains.
        inverse = e.nonlinear("reciprocal", (denominator, low, high))
        output = self._require_bound(e.mul(exponentials, inverse), 1.0)
        if hasattr(e, "require_bound"):
            self._softmax_proofs[id(output)] = (
                weakref.ref(output),
                dim % len(output.shape),
            )
        return output

    def _layer_norm(self, x, shape, weight, bias, epsilon):
        e = self.engine
        dims = list(range(len(x.shape) - len(shape), len(x.shape)))
        mean = e.mean(x, dims, True)
        centered = e.sub(x, mean)
        variance = e.mean(e.mul(centered, centered), dims, True)
        reciprocal_std = e.nonlinear("rsqrt", (e.add(variance, epsilon),))
        # sum(centered_i**2) = N*variance, and rsqrt Newton approaches
        # 1/sqrt(variance+epsilon) from below for nonnegative epsilon.
        output = e.mul(centered, reciprocal_std)
        if epsilon >= 0:
            output = self._require_bound(output, math.sqrt(math.prod(shape)))
            if hasattr(e, "require_bound"):
                self._norm_proofs[id(reciprocal_std)] = (
                    weakref.ref(reciprocal_std),
                    weakref.ref(x),
                    weakref.ref(mean),
                    tuple(shape),
                )
        if weight is not None:
            output = e.mul(output, weight)
        if bias is not None:
            output = e.add(output, bias)
        return output, mean, reciprocal_std

    def _layer_norm_backward(
        self, grad, x, shape, mean, reciprocal_std, weight, bias, mask
    ):
        e = self.engine
        dims = list(range(len(x.shape) - len(shape), len(x.shape)))
        outer = list(range(len(x.shape) - len(shape)))
        count = math.prod(shape)
        normalized = e.mul(e.sub(x, mean), reciprocal_std)
        proof = self._norm_proofs.get(id(reciprocal_std))
        if (
            proof is not None
            and proof[0]() is reciprocal_std
            and proof[1]() is x
            and proof[2]() is mean
            and proof[3] == tuple(shape)
        ):
            normalized = self._require_bound(normalized, math.sqrt(count))
        result = [None, None, None]
        if mask[0]:
            scaled = grad if weight is None else e.mul(grad, weight)
            centered = e.sub(e.mul(scaled, count), e.sum(scaled, dims, True))
            centered = e.sub(
                centered,
                e.mul(normalized, e.sum(e.mul(scaled, normalized), dims, True)),
            )
            result[0] = e.mul(e.mul(reciprocal_std, 1.0 / count), centered)
        if mask[1]:
            product = e.mul(grad, normalized)
            result[1] = e.sum(product, outer, False) if outer else product
        if mask[2]:
            result[2] = e.sum(grad, outer, False) if outer else grad
        return tuple(result)

    def lower(self, target, args, kwargs):
        """Lower one secret operation. There is deliberately no torch fallback."""
        e = self.engine
        op = str(target)
        if target is rms_norm_backward:
            grad, x, reciprocal, dims = args
            normalized = e.mul(x, reciprocal)
            proof = self._rms_proofs.get((id(x), id(reciprocal)))
            verified = (
                proof is not None
                and proof[0]() is x
                and proof[1]() is reciprocal
                and proof[2] == tuple(dims)
            )
            if verified:
                normalized = self._require_bound(
                    normalized, math.sqrt(math.prod(x.shape[d] for d in dims))
                )
            projection = e.mean(e.mul(grad, normalized), dims, True)
            if verified:
                # mean(z**2)<=1 implies mean(abs(z))<=1, so the
                # projection is bounded by the incoming gradient magnitude.
                projection = self._require_bound(projection, self._bound(grad))
            return e.mul(e.sub(grad, e.mul(normalized, projection)), reciprocal)
        if target is operator.getitem:
            if not isinstance(args[0], (tuple, list, dict)):
                raise NotImplementedError(
                    "secret tensor indexing must use an explicit slice"
                )
            return args[0][args[1]]
        if op in {
            "aten.add.Tensor",
            "aten.add.Scalar",
            "aten.sub.Tensor",
            "aten.sub.Scalar",
        }:
            right = args[1]
            alpha = kwargs.get("alpha", 1)
            if alpha != 1:
                right = e.mul(right, alpha)
            return (e.add if op.startswith("aten.add.") else e.sub)(args[0], right)
        if op in {"aten.mul.Tensor", "aten.mul.Scalar"}:
            return e.mul(args[0], args[1])
        if op in {"aten.div.Tensor", "aten.div.Scalar"}:
            return e.div(args[0], args[1])
        if op == "aten.neg.default":
            return e.neg(args[0])
        if op == "aten.pow.Tensor_Scalar":
            return e.pow(args[0], args[1])
        if op in {"aten.mm.default", "aten.bmm.default"}:
            return e.matmul(args[0], args[1])
        if op == "aten.addmm.default":
            product = e.matmul(args[1], args[2])
            alpha, beta = kwargs.get("alpha", 1), kwargs.get("beta", 1)
            if alpha != 1:
                product = e.mul(product, alpha)
            return (
                product
                if beta == 0
                else e.add(product, args[0] if beta == 1 else e.mul(args[0], beta))
            )
        if op in {
            "aten.view.default",
            "aten._unsafe_view.default",
            "aten.reshape.default",
        }:
            return e.reshape(args[0], args[1])
        if op == "aten.permute.default":
            return e.permute(args[0], args[1])
        if op in {"aten.t.default", "aten.transpose.int"}:
            rank = len(args[0].shape)
            dims = list(range(rank))
            first, second = (0, 1) if op == "aten.t.default" else (args[1], args[2])
            if rank < 2 and op == "aten.t.default":
                return args[0]
            dims[first], dims[second] = dims[second], dims[first]
            return e.permute(args[0], dims)
        if op == "aten.expand.default":
            return e.expand(args[0], args[1])
        if op == "aten.unsqueeze.default":
            shape = list(args[0].shape)
            dim = args[1] % (len(shape) + 1)
            shape.insert(dim, 1)
            return e.reshape(args[0], shape)
        if op == "aten.slice.Tensor":
            return e.slice(
                args[0],
                args[1] if len(args) > 1 else 0,
                args[2] if len(args) > 2 else None,
                args[3] if len(args) > 3 else None,
                args[4] if len(args) > 4 else 1,
            )
        if op == "aten.slice_backward.default":
            return e.slice_backward(*args)
        if op == "aten.cat.default":
            return e.cat(args[0], args[1] if len(args) > 1 else 0)
        if op == "aten.stack.default":
            return e.stack(args[0], args[1] if len(args) > 1 else 0)
        if op in {"aten.split_with_sizes.default", "aten.unbind.int"}:
            x = args[0]
            dim = (
                (args[2] if len(args) > 2 else 0)
                if "split" in op
                else (args[1] if len(args) > 1 else 0)
            )
            dim %= len(x.shape)
            sizes = args[1] if "split" in op else [1] * x.shape[dim]
            result, offset = [], 0
            for size in sizes:
                item = e.slice(x, dim, offset, offset + size, 1)
                if "unbind" in op:
                    shape = list(item.shape)
                    del shape[dim]
                    item = e.reshape(item, shape)
                result.append(item)
                offset += size
            return tuple(result)
        if op in {
            "aten.mean.dim",
            "aten.sum.dim_IntList",
            "aten.mean.default",
            "aten.sum.default",
        }:
            dims = args[1] if len(args) > 1 else None
            keepdim = args[2] if len(args) > 2 else False
            return (e.mean if "mean" in op else e.sum)(args[0], dims, keepdim)
        if op in {"aten.alias.default", "aten.detach.default", "aten.clone.default"}:
            return e.clone(args[0])
        if op == "aten._to_copy.default":
            dtype = kwargs.get("dtype", args[0].dtype)
            if dtype != args[0].dtype:
                raise NotImplementedError(
                    "secret dtype conversion requires an explicit engine policy"
                )
            return e.clone(args[0])
        if op in {
            "aten.zeros_like.default",
            "aten.empty_like.default",
            "aten.ones_like.default",
        }:
            # Values are independent of secret tensor contents; empty scratch
            # storage is zero-initialized until its explicit fill operation.
            value = torch.zeros(
                tuple(args[0].shape), dtype=kwargs.get("dtype", args[0].dtype)
            )
            return value + 1 if op == "aten.ones_like.default" else value
        if op == "aten.fill.Scalar":
            return torch.full(tuple(args[0].shape), args[1], dtype=args[0].dtype)
        if op == "aten.copy.default":
            return e.clone(args[1])
        if op == "aten._softmax.default":
            return self._softmax(args[0], args[1])
        if op == "aten._log_softmax.default":
            return self._softmax(args[0], args[1], logarithmic=True)
        if op == "aten._softmax_backward_data.default":
            grad, output, dim = args[:3]
            # The forward probability invariant also bounds this weighted
            # gradient sum independently of the reduction width.
            weighted = e.sum(e.mul(grad, output), [dim], True)
            proof = self._softmax_proofs.get(id(output))
            if (
                proof is not None
                and proof[0]() is output
                and proof[1] == dim % len(output.shape)
            ):
                weighted = self._require_bound(weighted, self._bound(grad))
            return e.mul(output, e.sub(grad, weighted))
        if op == "aten._log_softmax_backward_data.default":
            grad, output, dim = args[:3]
            return e.sub(
                grad, e.mul(e.nonlinear("exp", (output,)), e.sum(grad, [dim], True))
            )
        if op == "aten.native_layer_norm.default":
            return self._layer_norm(*args)
        if op == "aten.native_layer_norm_backward.default":
            return self._layer_norm_backward(*args)
        if op == "aten.gelu.default":
            return e.nonlinear("gelu", (args[0], kwargs.get("approximate", "none")))
        if op == "aten.gelu_backward.default":
            return e.nonlinear(
                "gelu_backward", (args[0], args[1], kwargs.get("approximate", "none"))
            )
        if op == "aten.silu.default":
            return e.mul(args[0], e.nonlinear("sigmoid", (args[0],)))
        if op in {"aten.sigmoid.default", "aten.rsqrt.default"}:
            return e.nonlinear(op.split(".")[1], (args[0],))
        if op == "aten.convolution.default":
            (
                x,
                weight,
                bias,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            ) = args
            if transposed or any(output_padding):
                raise NotImplementedError("transposed convolution is not supported")
            return e.conv2d(x, weight, bias, stride, padding, dilation, groups)
        if op == "aten.convolution_backward.default":
            (
                grad,
                x,
                weight,
                bias_sizes,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
                output_mask,
            ) = args
            if transposed or any(output_padding):
                raise NotImplementedError(
                    "transposed convolution backward is not supported"
                )
            return e.conv2d_backward(
                grad, x, weight, stride, padding, dilation, groups, output_mask
            )
        raise NotImplementedError(f"no encrypted lowering for {target}")
