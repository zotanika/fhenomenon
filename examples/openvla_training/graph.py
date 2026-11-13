"""Full OpenVLA LoRA graph capture; this module performs no encryption.

The public model architecture is retained. Only discrete embedding/label lookups
are represented by equivalent dense one-hot algebra, allowing those inputs to
remain secret in a subsequent encrypted interpreter. Shape-only profiles never
load checkpoint weights and are not numerical or encrypted execution evidence.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from .source import DEFAULT_SNAPSHOT

import hashlib
import math
import os
from pathlib import Path
import time
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "build/openvla-hf"))
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / "build/openvla-hf/modules"))

import torch
import torch.nn.functional as F
from torch.func import functional_call, functionalize, grad_and_value
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten
from fhenomenon.ml.rms import rms_norm_backward


@dataclass
class CapturedTrainingGraph:
    model: torch.nn.Module
    gm: torch.fx.GraphModule
    args: tuple[Any, ...]
    trainable_names: tuple[str, ...]
    eager_result: Any | None
    validation: dict[str, Any] | None
    metadata: dict[str, Any]
    capture_settings: dict[str, Any] | None = None


def public_pixel_fixture(shape, *, dtype=torch.float32, device="cpu", data_seed=None):
    """Generate synthetic pixels independently of the model initializer RNG."""
    if data_seed is None:
        return torch.full(shape, 0.25, dtype=dtype, device=device)
    if (
        isinstance(data_seed, bool)
        or not isinstance(data_seed, int)
        or not 0 <= data_seed < 2**63
    ):
        raise ValueError("data_seed must be a nonnegative int64")
    generator = torch.Generator(device="cpu").manual_seed(data_seed)
    return (
        torch.empty(shape, dtype=dtype)
        .uniform_(-0.5, 0.5, generator=generator)
        .to(device)
    )


def rewrite_rms_backward(gm):
    """Fuse only the reviewed RMS derivative, retaining the saved rsqrt value.

    Replace g*r + ((-.5*sum(g*x))*r**3/N)*(2*x) with the equivalent
    (g - (x*r)*mean(g*(x*r)))*r. Unmatched graphs remain untouched.
    This is an algebraic rewrite, not a nonlinear approximation or range clamp.
    """

    def call(node, names, arity):
        return (
            isinstance(node, torch.fx.Node)
            and node.op == "call_function"
            and str(node.target) in names
            and len(node.args) == arity
            and not node.kwargs
        )

    def dimensions(dims, shape):
        if dims is None or len(dims) == 0:
            return tuple(range(len(shape)))
        if any(
            not isinstance(d, int) or not -len(shape) <= d < len(shape) for d in dims
        ):
            return None
        return tuple(sorted({d % len(shape) for d in dims}))

    mul = {"aten.mul.Tensor", "aten.mul.Scalar"}
    count = 0
    for node in list(gm.graph.nodes):
        if not call(node, {"aten.add.Tensor"}, 2):
            continue
        direct, correction = node.args
        residual = None
        if call(direct, {"aten.add.Tensor"}, 2):
            residual, direct = direct.args
        if not call(direct, {"aten.mul.Tensor"}, 2) or not call(
            correction, {"aten.mul.Tensor"}, 2
        ):
            continue
        grad, reciprocal = direct.args
        divided, twice_x = correction.args
        if not call(reciprocal, {"aten.rsqrt.default"}, 1) or not call(
            divided, {"aten.div.Tensor", "aten.div.Scalar"}, 2
        ):
            continue
        if not call(twice_x, mul, 2) or twice_x.args[1] != 2:
            continue
        identity = twice_x.args[0]
        if not call(identity, {"aten.pow.Tensor_Scalar"}, 2) or identity.args[1] != 1:
            continue
        x = identity.args[0]
        if not isinstance(x, torch.fx.Node) or not isinstance(grad, torch.fx.Node):
            continue
        shape = tuple(getattr(x.meta.get("val"), "shape", ()))
        if not shape or tuple(getattr(grad.meta.get("val"), "shape", ())) != shape:
            continue
        expanded, divisor = divided.args
        if (
            not call(expanded, {"aten.expand.default"}, 2)
            or tuple(expanded.args[1]) != shape
        ):
            continue
        product = expanded.args[0]
        if not call(product, {"aten.mul.Tensor"}, 2):
            continue
        scaled_sum, cubed = product.args
        if not call(cubed, {"aten.pow.Tensor_Scalar"}, 2) or cubed.args != (
            reciprocal,
            3,
        ):
            continue
        if not call(scaled_sum, mul, 2) or scaled_sum.args[1] != -0.5:
            continue
        reduced = scaled_sum.args[0]
        if (
            not call(reduced, {"aten.sum.dim_IntList"}, 3)
            or reduced.args[2] is not True
        ):
            continue
        inner = reduced.args[0]
        if not call(inner, {"aten.mul.Tensor"}, 2) or inner.args != (grad, x):
            continue
        dims = dimensions(reduced.args[1], shape)
        if not dims or divisor != math.prod(shape[d] for d in dims):
            continue
        # Verify the saved reciprocal came from this x and these same axes.
        offset = reciprocal.args[0]
        if not call(offset, {"aten.add.Tensor"}, 2):
            continue
        mean, epsilon = offset.args
        if (
            not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon)
            or epsilon < 0
        ):
            continue
        if (
            not call(mean, {"aten.mean.dim"}, 3)
            or mean.args[2] is not True
            or dimensions(mean.args[1], shape) != dims
        ):
            continue
        square = mean.args[0]
        if not call(square, {"aten.pow.Tensor_Scalar"}, 2) or square.args != (x, 2):
            continue
        reduced_shape = tuple(1 if d in dims else size for d, size in enumerate(shape))
        if tuple(getattr(reciprocal.meta.get("val"), "shape", ())) != reduced_shape:
            continue
        with gm.graph.inserting_before(node):
            replacement = gm.graph.call_function(
                rms_norm_backward, (grad, x, reciprocal, list(dims))
            )
            replacement.meta = dict(node.meta)
            if residual is not None:
                replacement = gm.graph.call_function(
                    torch.ops.aten.add.Tensor, (residual, replacement)
                )
                replacement.meta = dict(node.meta)
        node.replace_all_uses_with(replacement)
        gm.graph.erase_node(node)
        count += 1
    if count:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return count


def _dense_embedding(module, onehot):
    return onehot @ module.weight


@contextmanager
def _meta_disabled_autocast():
    """Transformers 4.40 rotary uses unsupported, disabled meta autocast."""
    original = torch.autocast

    def wrapped(device_type, *args, **kwargs):
        if device_type == "meta" and kwargs.get("enabled") is False:
            return nullcontext()
        return original(device_type, *args, **kwargs)

    torch.autocast = wrapped
    try:
        yield
    finally:
        torch.autocast = original


def _build_model(snapshot, profile):
    import timm
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from timm.models import register_model
    from timm.models.vision_transformer import VisionTransformer
    from transformers import AutoConfig, AutoModelForVision2Seq, LlamaConfig

    timm.layers.set_fused_attn(False)
    config = AutoConfig.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True
    )
    if profile in {"micro", "tiny"}:
        micro = profile == "micro"
        timm_id = "fhn_training_micro_vit" if micro else "fhn_training_vit"
        if not timm.is_model(timm_id):

            def training_vit(pretrained=False, **kwargs):
                kwargs.pop("pretrained_cfg", None)
                kwargs.pop("pretrained_cfg_overlay", None)
                kwargs.pop("act_layer", None)
                return VisionTransformer(
                    patch_size=1 if micro else 8,
                    embed_dim=4 if micro else 8,
                    depth=2,
                    num_heads=1 if micro else 2,
                    mlp_ratio=2,
                    **kwargs,
                )

            training_vit.__name__ = timm_id
            register_model(training_vit)

        config.text_config = LlamaConfig(
            vocab_size=16 if micro else 32,
            hidden_size=4 if micro else 8,
            intermediate_size=8 if micro else 16,
            num_hidden_layers=2,
            num_attention_heads=1 if micro else 2,
            num_key_value_heads=1 if micro else 2,
            pad_token_id=0,
            max_position_embeddings=32,
        )
        config.pad_token_id = 0
        config.pad_to_multiple_of = 0
        config.n_action_bins = 8
        config.image_sizes = [2, 2] if micro else [16, 16]
        config.timm_model_ids = [timm_id, timm_id]
        config.timm_override_act_layers = [None, None]
    context = init_empty_weights() if profile == "full7b_meta" else nullcontext()
    with context:
        model = AutoModelForVision2Seq.from_config(
            config, trust_remote_code=True, attn_implementation="eager"
        )
        base_parameter_count = sum(p.numel() for p in model.parameters())
        model = get_peft_model(
            model,
            LoraConfig(
                r=2, lora_alpha=2, lora_dropout=0.0, target_modules="all-linear"
            ),
        )
    if profile == "full7b_meta":
        model.to("meta")
    model.train()
    return model, config, base_parameter_count


def _tensor_description(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "bytes": value.numel() * value.element_size(),
        }
    if isinstance(value, (tuple, list)):
        return [_tensor_description(item) for item in value]
    return None


def _numel(value):
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, (tuple, list)):
        return sum(_numel(v) for v in value)
    return 0


def _argument_description(value):
    """Bounded structural metadata; tensor contents never enter a report."""
    if isinstance(value, torch.fx.Node):
        return {"node": value.name}
    if isinstance(value, (tuple, list)):
        return [_argument_description(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _argument_description(v) for k, v in value.items()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _ancestors(nodes):
    found = set()
    pending = list(nodes)
    while pending:
        node = pending.pop()
        if not isinstance(node, torch.fx.Node) or node in found:
            continue
        found.add(node)
        pending.extend(node.all_input_nodes)
    return found


def _macs(node):
    target = str(node.target)
    if target in {"aten.mm.default", "aten.bmm.default", "aten.addmm.default"}:
        lhs = node.args[1] if target == "aten.addmm.default" else node.args[0]
        shape = lhs.meta["val"].shape
        return _numel(node.meta.get("val")) * shape[-1]
    if target in {"aten.convolution.default", "aten.convolution_backward.default"}:
        backward = "backward" in target
        weight = node.args[2] if backward else node.args[1]
        weight_shape = weight.meta["val"].shape
        output = node.args[0].meta["val"] if backward else node.meta.get("val")
        cost = _numel(output) * math.prod(weight_shape[1:])
        if backward:
            mask = node.args[-1]
            cost *= int(mask[0]) + int(mask[1])
        return cost
    return 0


def _annotate(gm, args, trainable_names, max_report_nodes):
    nodes = list(gm.graph.nodes)
    leaves, _ = tree_flatten(nodes[-1].args[0])
    forward = _ancestors(leaves[:1])
    backward = _ancestors(leaves[1 : 1 + len(trainable_names)]) - forward
    updates = _ancestors(leaves[1 + len(trainable_names) :]) - forward - backward
    roles = []
    for role, values, secret in (
        ("adapter", args[0], True),
        ("frozen_weight", args[1], False),
        ("buffer", args[2], False),
    ):
        roles.extend((role, name, secret) for name in values)
    roles.extend(
        (role, role, True)
        for role in ("instruction_onehot", "pixels", "action_targets_onehot")
    )
    role_iter = iter(roles)
    operators = Counter()
    phase_counts = Counter()
    privacy_counts = Counter()
    nonlinear = Counter()
    total_macs = 0
    tensor_elements = 0
    encrypted_macs = 0
    lower_depth = {}
    max_depth = 0
    rows = []
    for node in nodes:
        if node.op == "placeholder":
            role, name, secret = next(role_iter)
            info = {
                "role": role,
                "parameter_or_input": name,
                "phase": "input",
                "privacy": "secret" if secret else "public",
            }
        else:
            secret = any(
                n.meta["training"]["privacy"] == "secret" for n in node.all_input_nodes
            )
            phase = (
                "forward"
                if node in forward
                else "backward"
                if node in backward
                else "update"
                if node in updates
                else "output"
            )
            info = {
                "role": "constant" if node.op == "get_attr" else "operation",
                "phase": phase,
                "privacy": "secret" if secret else "public",
            }
        info["tensor"] = _tensor_description(node.meta.get("val"))
        info["module_stack"] = {
            str(k): [str(vv) for vv in v] if isinstance(v, tuple) else str(v)
            for k, v in node.meta.get("nn_module_stack", {}).items()
        }
        info["source_fn"] = str(node.meta.get("source_fn_stack", ""))
        info["encrypted_input_count"] = sum(
            n.meta["training"]["privacy"] == "secret" for n in node.all_input_nodes
        )
        target = str(node.target)
        macs = _macs(node) if node.op == "call_function" else 0
        depth = max((lower_depth[n] for n in node.all_input_nodes), default=0)
        if (
            secret
            and info["encrypted_input_count"] >= 2
            and (
                target.startswith("aten.mul.")
                or target
                in {
                    "aten.mm.default",
                    "aten.bmm.default",
                    "aten.addmm.default",
                    "aten.convolution.default",
                }
            )
        ):
            depth += 1
        lower_depth[node] = depth
        max_depth = max(max_depth, depth)
        info["dense_multiply_accumulates"] = macs
        info["multiplication_depth_lower_bound"] = depth
        node.meta["training"] = info
        phase_counts[info["phase"]] += 1
        privacy_counts[info["privacy"]] += 1
        if node.op == "call_function":
            operators[target] += 1
            total_macs += macs
            encrypted_macs += macs if secret else 0
            tensor_elements += _numel(node.meta.get("val"))
            if any(
                token in target
                for token in (
                    "softmax",
                    "gelu",
                    "silu",
                    "sigmoid",
                    "rsqrt",
                    "layer_norm",
                )
            ):
                nonlinear[target] += _numel(node.meta.get("val"))
        if len(rows) < max_report_nodes:
            row = {"name": node.name, "op": node.op, "target": target, **info}
            if node.op == "call_function":
                row["args"] = _argument_description(node.args)
                row["kwargs"] = _argument_description(node.kwargs)
            rows.append(row)
    return {
        "node_count": len(nodes),
        "operator_count": sum(operators.values()),
        "operators": dict(sorted(operators.items())),
        "phase_counts": dict(phase_counts),
        "privacy_counts": dict(privacy_counts),
        "nodes": rows,
        "node_details_truncated": len(rows) < len(nodes),
        "cost_estimates": {
            "dense_multiply_accumulates": total_macs,
            "secret_dense_multiply_accumulates": encrypted_macs,
            "operator_output_tensor_elements": tensor_elements,
            "multiplication_depth_lower_bound": max_depth,
            "nonlinear_output_elements_by_operator": dict(sorted(nonlinear.items())),
            "scope": "Shape-derived arithmetic counts, not ciphertext operations, memory peaks, or latency. Nonlinear polynomial depths, packing, bootstraps and slot/coefficient transforms are not priced.",
        },
    }


def _compare_result(actual, expected, adapters):
    grad_error = max(
        float((actual[1][k] - expected[1][k]).abs().max()) for k in adapters
    )
    update_error = max(
        float((actual[2][k] - expected[2][k]).abs().max()) for k in adapters
    )
    loss_error = float((actual[0] - expected[0]).abs())
    return {
        "loss_absolute_error": loss_error,
        "gradient_max_absolute_error": grad_error,
        "update_max_absolute_error": update_error,
        "compared_adapter_tensors": len(adapters),
        "nonzero_updated_tensors": sum(
            bool(torch.count_nonzero(actual[2][k] - adapters[k])) for k in adapters
        ),
        "passed": loss_error <= 2e-6 and grad_error <= 2e-5 and update_error <= 2e-7,
        "atol": {"loss": 2e-6, "gradient": 2e-5, "update": 2e-7},
    }


def capture_training_graph(
    snapshot=str(DEFAULT_SNAPSHOT),
    profile="tiny",
    *,
    learning_rate=1e-3,
    seed=7,
    max_report_nodes=256,
    data_seed=None,
):
    """Capture forward, activation backward, all adapter gradients and SGD.

    args order: (adapters, public_frozen_weights, public_buffers,
    secret_instruction_onehot, secret_pixels, secret_action_targets_onehot).
    The returned model uses dense one-hot embedding at its input boundary.
    Only frozen pretrained *architecture* is available here: tiny uses random
    weights, and full7b_meta carries shapes without checkpoint tensor contents.
    """
    if profile not in {"micro", "tiny", "full7b_meta"}:
        raise ValueError("profile must be micro, tiny or full7b_meta")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if max_report_nodes < 0:
        raise ValueError("max_report_nodes must be nonnegative")
    start = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, config, base_count = _build_model(str(snapshot), profile)
    meta = profile == "full7b_meta"
    device = "meta" if meta else "cpu"
    adapters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    frozen = {
        name: p.detach() for name, p in model.named_parameters() if not p.requires_grad
    }
    buffers = {name: b.detach() for name, b in model.named_buffers()}
    vocab = config.text_config.vocab_size
    ids = torch.tensor(
        [
            [
                1,
                2,
                3,
                vocab - 8,
                vocab - 7,
                vocab - 6,
                vocab - 5,
                vocab - 4,
                vocab - 3,
                vocab - 2,
                2,
            ]
        ]
    )
    labels = ids.clone()
    labels[:, :-8] = -100
    onehot = F.one_hot(ids, num_classes=vocab).float().to(device)
    targets = F.one_hot(ids[:, -8:], num_classes=vocab).float().to(device)
    resolution = config.image_sizes[0]
    pixels = public_pixel_fixture(
        (1, 6, resolution, resolution), device=device, data_seed=data_seed
    )
    names = tuple(adapters)
    native_result = None
    inactive = []
    if not meta:
        loss = functional_call(
            model,
            ({**frozen, **adapters}, buffers),
            (),
            dict(
                input_ids=ids,
                pixel_values=pixels,
                labels=labels,
                use_cache=False,
                return_dict=True,
            ),
        ).loss
        native_grads = torch.autograd.grad(
            loss, tuple(adapters.values()), allow_unused=True
        )
        inactive = [name for name, grad in zip(names, native_grads) if grad is None]
        grads = {
            name: torch.zeros_like(adapters[name]) if grad is None else grad.detach()
            for name, grad in zip(names, native_grads)
        }
        native_result = (
            loss.detach(),
            grads,
            {k: adapters[k].detach() - learning_rate * v for k, v in grads.items()},
        )

    # This algebraic input encoding is the only model-module boundary adaptation.
    embedding = model.get_input_embeddings()
    embedding.forward = MethodType(_dense_embedding, embedding)

    def loss_fn(trainable, fixed, state, instruction, image, action_targets):
        output = functional_call(
            model,
            ({**fixed, **trainable}, state),
            (),
            dict(
                input_ids=instruction,
                pixel_values=image,
                use_cache=False,
                return_dict=True,
            ),
        )
        prediction_logits = output.logits[:, -9:-1, :]
        log_prob = F.log_softmax(prediction_logits, dim=-1)
        return -(log_prob * action_targets).sum(dim=-1).mean()

    def step(*arguments):
        gradients, loss = grad_and_value(loss_fn)(*arguments)
        updated = {
            name: arguments[0][name] - learning_rate * gradient
            for name, gradient in gradients.items()
        }
        return loss, gradients, updated

    args = (adapters, frozen, buffers, onehot, pixels, targets)
    with _meta_disabled_autocast() if meta else nullcontext():
        eager = None if meta else step(*args)
        gm = make_fx(functionalize(step), record_module_stack=True)(*args)
    before_dce = len(list(gm.graph.nodes))
    rms_backward_rewrites = rewrite_rms_backward(gm)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    validation = None
    if not meta:
        actual = gm(*args)
        validation = _compare_result(actual, eager, adapters)
        reference = _compare_result(eager, native_result, adapters)
        reference["passed"] = (
            reference["loss_absolute_error"] <= 2e-6
            and reference["gradient_max_absolute_error"] <= 2e-5
            and reference["update_max_absolute_error"] <= 2e-7
        )
        reference["comparison"] = (
            "Original integer embedding and native masked cross entropy versus dense one-hot algebra"
        )
        reference["atol"] = {"loss": 2e-6, "gradient": 2e-5, "update": 2e-7}
        validation["native_reference"] = reference
        validation["inactive_adapter_tensors"] = inactive
        validation["passed"] = validation["passed"] and reference["passed"]
    metadata = _annotate(gm, args, names, max_report_nodes)
    base_model = model.base_model.model
    vision_modules = [base_model.vision_backbone.featurizer]
    if base_model.vision_backbone.use_fused_vision_backbone:
        vision_modules.append(base_model.vision_backbone.fused_featurizer)
    metadata.update(
        {
            "schema_version": 1,
            "profile": profile,
            "model_seed": seed,
            "data_seed": data_seed,
            "model_class": type(model.base_model.model).__name__,
            "checkpoint_weights_loaded": False,
            "encrypted_execution_performed": False,
            "base_parameter_count": base_count,
            "adapter_parameter_count": sum(p.numel() for p in adapters.values()),
            "adapter_tensor_count": len(adapters),
            "nodes_before_dce": before_dce,
            "rms_backward_rewrites": rms_backward_rewrites,
            "architecture": {
                "weights": "shape_only_without_tensor_contents"
                if meta
                else "seeded_random_weights",
                "decoder_layers": len(base_model.language_model.model.layers),
                "decoder_hidden_size": config.text_config.hidden_size,
                "vocabulary_size": vocab,
                "vision_backbones": [
                    {
                        "configured_blocks": len(vision.blocks),
                        "selected_output_block_zero_based": len(vision.blocks) - 2,
                        "feature_size": vision.embed_dim,
                        "patch_size": list(vision.patch_embed.patch_size),
                    }
                    for vision in vision_modules
                ],
                "completeness": "Both selected vision feature streams, multimodal projector, all decoder layers, masked action loss, activation backward and every adapter SGD update are captured. Official OpenVLA selects penultimate vision features; dead-code elimination removes unused final-block work without changing its outputs or gradients.",
            },
            "capture_seconds": time.perf_counter() - start,
            "data": (
                "Synthetic fixed-length instruction ids, "
                + (
                    "constant normalized"
                    if data_seed is None
                    else "separately seeded uniform [-0.5,0.5]"
                )
                + " six-channel RGB pair, seven action tokens and EOS; no robot accuracy evidence"
            ),
            "input_shapes": {
                "instruction_onehot": list(onehot.shape),
                "pixels": list(pixels.shape),
                "action_targets_onehot": list(targets.shape),
            },
            "input_privacy": {
                "adapters": "secret",
                "frozen_weights": "public",
                "buffers": "public",
                "instruction_onehot": "secret",
                "pixels": "secret",
                "action_targets_onehot": "secret",
                "lengths_positions_and_supervised_mask": "public fixed metadata",
            },
            "secret_indices": "Client encodes instructions and action targets as one-hot tensors before encryption; embedding is matrix multiplication and masked cross entropy uses a secret label dot product, never secret integer indexing.",
            "runtime_policy": {
                "intermediate_decryptions_allowed": 0,
                "final_egress_level": "minimum_supported_level",
                "final_egress_enforcement": "Required of encrypted runtime before every authorized final decryption; not executed by this plaintext graph capture.",
                "refresh": "Public bootstrap and slot/coefficient transforms require provider lowering; never substitute decrypt/re-encrypt.",
            },
            "validation": validation,
            "source_sha256": {
                name: hashlib.sha256((Path(snapshot) / name).read_bytes()).hexdigest()
                for name in (
                    "configuration_prismatic.py",
                    "modeling_prismatic.py",
                    "config.json",
                )
            },
        }
    )
    settings = {
        "snapshot": str(Path(snapshot).resolve()),
        "profile": profile,
        "learning_rate": learning_rate,
        "seed": seed,
        "max_report_nodes": 0,
    }
    return CapturedTrainingGraph(
        model, gm, args, names, eager, validation, metadata, settings
    )
