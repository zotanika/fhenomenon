"""Run the complete OpenVLA forward/backward/adapter update graph.

Reference modes are deliberately named noncryptographic. CKKS modes have no
intermediate decryption, host nonlinear evaluation, or silent provider fallback.
"""

from .source import DEFAULT_SNAPSHOT

import argparse
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from .graph import capture_training_graph
from .interpreter import RANGE_POLICY
from fhenomenon.ml import compile_model, native_tensor_registry, runtime_environment
from fhenomenon.ml.tensor import TensorEngine, ReferenceVectorBackend
from fhenomenon.ml.approximations import CONTRACTS


def _comparison_diagnostics(statistics):
    relative = cosine = None
    if statistics is not None:
        expected = statistics["expected_squared"]
        actual = statistics["actual_squared"]
        error = statistics["error_squared"]
        if expected > 0:
            relative = math.sqrt(error) / math.sqrt(expected)
            if not math.isfinite(relative):
                relative = None
        elif error == 0:
            relative = 0.0
        if actual > 0 and expected > 0:
            cosine = statistics["dot"] / math.sqrt(actual) / math.sqrt(expected)
            cosine = max(-1.0, min(1.0, cosine)) if math.isfinite(cosine) else None
    return {"relative_l2_error": relative, "cosine_similarity": cosine}


def _compare(engine, actual, expected):
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError("training output names changed")
        children = {k: _compare(engine, v, expected[k]) for k, v in actual.items()}
        results = {k: v[0] for k, v in children.items()}
        finite = all(x["all_finite"] for x in results.values())
        error = (
            max((x["max_abs_error"] for x in results.values()), default=0.0)
            if finite
            else None
        )
        expected_maxima = [x["expected_max_abs"] for x in results.values()]
        expected_max = (
            max(expected_maxima, default=0.0) if None not in expected_maxima else None
        )
        statistics = None
        if finite:
            # Combine globally scaled sums, not the per-tensor ratios/cosines.
            # Scaling keeps squared norms representable for very small gradients.
            terms = [v[1] for v in children.values()]
            scale = max((term["scale"] for term in terms), default=0.0)
            statistics = {"scale": scale}
            for key in ("error_squared", "actual_squared", "expected_squared", "dot"):
                statistics[key] = (
                    math.fsum(
                        term[key] * (term["scale"] / scale) ** 2 for term in terms
                    )
                    if scale
                    else 0.0
                )
        result = {
            "count": len(results),
            "max_abs_error": error,
            "expected_max_abs": expected_max,
            "all_finite": finite,
            "tensors": results,
            **_comparison_diagnostics(statistics),
        }
        return result, statistics
    got = np.asarray(engine.decrypt(actual), dtype=np.float64)
    want = expected.detach().cpu().numpy().astype(np.float64, copy=False)
    if got.shape != want.shape:
        raise ValueError("training output shape changed")
    expected_finite = bool(np.isfinite(want).all())
    expected_max = float(np.max(np.abs(want), initial=0.0)) if expected_finite else None
    finite = bool(np.isfinite(got).all() and expected_finite)
    with np.errstate(over="ignore", invalid="ignore"):
        error = float(np.max(np.abs(got - want))) if finite and got.size else 0.0
    finite = finite and math.isfinite(error)
    statistics = None
    if finite:
        scale = max(float(np.max(np.abs(got), initial=0.0)), expected_max)
        a, b = (got / scale, want / scale) if scale else (got, want)
        difference = a - b
        statistics = {
            "scale": scale,
            "error_squared": float(np.sum(difference * difference)),
            "actual_squared": float(np.sum(a * a)),
            "expected_squared": float(np.sum(b * b)),
            "dot": float(np.sum(a * b)),
        }
    result = {
        "max_abs_error": error if finite else None,
        "expected_max_abs": expected_max,
        "all_finite": finite,
        "shape": list(got.shape),
        **_comparison_diagnostics(statistics),
    }
    return result, statistics


def compare(engine, actual, expected):
    return _compare(engine, actual, expected)[0]


def _adapter_public_bounds(adapters, policy, uniform_bound):
    """Bounds from public initialization rules and shapes, never tensor data.

    Fresh PEFT initialization uses Kaiming uniform with a=sqrt(5) for A,
    hence 1/sqrt(fan_in), and exactly zero for B. Updated or imported
    adapters require the explicitly selected uniform policy instead.
    """
    if policy == "uniform":
        return {name: uniform_bound for name in adapters}
    if policy != "initialization":
        raise ValueError("unknown adapter bound policy")
    result = {}
    for name, value in adapters.items():
        if name.endswith(".lora_A.default.weight"):
            fan_in = math.prod(value.shape[1:])
            if len(value.shape) < 2 or fan_in <= 0:
                raise ValueError("unsupported LoRA A initialization shape: " + name)
            result[name] = 1 / math.sqrt(fan_in)
        elif name.endswith(".lora_B.default.weight"):
            result[name] = 0.0
        else:
            raise ValueError("unsupported adapter initialization rule: " + name)
    return result


def run(args):
    run_started = started = time.perf_counter()
    reference = args.backend.startswith("reference")
    backend = client = engine = program = None
    progress_state = {}
    last_report = [started]

    def progress(index, total, node):
        progress_state.update(
            {
                "completed_nodes": index,
                "total_nodes": total,
                "last_node": node.name,
                "phase": node.meta["training"].get("phase"),
                "activity": "reference_execution" if reference else "scope_capture",
            }
        )
        if time.perf_counter() - last_report[0] > 30:
            print(
                json.dumps(
                    {
                        "progress": progress_state,
                        "elapsed_seconds": time.perf_counter() - started,
                        "engine": engine.stats,
                    }
                ),
                flush=True,
            )
            last_report[0] = time.perf_counter()
        if args.max_seconds and time.perf_counter() - started > args.max_seconds:
            raise TimeoutError(
                "explicit execution time budget exhausted at a graph node boundary"
            )

    report = {
        "profile": args.profile,
        "backend": args.backend,
        "encrypted_execution": False,
        "execution_mode": "noncryptographic_reference"
        if reference
        else "single_fhenomenon_scope",
        "public_node_policy": "compile-time specialization of declared public graph nodes",
        "execution_completed": False,
        "passed": False,
        "aborted": False,
        "checkpoint_weights_loaded": False,
        "model": "actual OpenVLA architecture, reduced random configuration",
        "scope": "complete vision and language forward, action loss, activation/adapter backward, adapter SGD update",
        "intermediate_decryption_allowed": False,
        "approximation_contracts": CONTRACTS,
        "approximation_domain_guarantee": "declared requirements; not proven for arbitrary training data",
        "refresh_target": args.refresh_target,
        "slots_per_chunk": args.slots,
        "input_public_bound": None,
        "data_seed": getattr(args, "data_seed", None),
        "public_range_contract": None,
        "validation_conditioned_on_public_range_contract": False,
        # Criteria are declared before setup or execution and never relaxed.
        "validation_thresholds": {
            "loss_atol": 0.005,
            "gradient_atol": 0.005,
            "update_atol": 5e-6,
        },
        "validation_diagnostics": {
            "aggregation": "Dictionary diagnostics use all elements of all descendant tensors as one vector.",
            "relative_l2_error": "L2(actual-expected)/L2(expected); zero for two zero vectors, null for nonzero error with zero reference norm.",
            "cosine_similarity": "Dot product divided by both L2 norms; null if either vector has zero norm.",
            "criteria": "Diagnostics do not change the predeclared absolute-error pass/fail thresholds.",
        },
        "deployment": "in-process client and evaluator roles; not process/network isolation",
    }
    stage = "configuration"
    try:
        public_bound = float(getattr(args, "input_public_bound", 1.0))
        if not math.isfinite(public_bound) or public_bound < 1:
            raise ValueError("input public bound must be finite and at least one")
        report["input_public_bound"] = public_bound
        adapter_policy = getattr(args, "adapter_bound_policy", "initialization")
        crypto_profile = getattr(args, "crypto_profile", "Native128")
        if adapter_policy not in ("uniform", "initialization"):
            raise ValueError("unknown adapter bound policy")
        if crypto_profile not in ("Native128", "NativeTest"):
            raise ValueError("unsupported crypto profile")
        if not reference and (
            not isinstance(args.slots, int)
            or not 1 <= args.slots <= 256
            or args.slots & (args.slots - 1)
        ):
            raise ValueError("native slots must be a power of two between 1 and 256")
        if not reference and args.refresh_target not in range(9):
            raise ValueError("native refresh target must be between zero and eight")
        report.update(
            {
                "adapter_bound_policy": adapter_policy,
                "crypto_profile": crypto_profile,
                "cryptographic_test_only": crypto_profile == "NativeTest",
                "cryptographic_profile_scope": "Explicitly insecure small-ring numerical fixture; not production confidentiality."
                if crypto_profile == "NativeTest"
                else "Native128 enforces the native modulus-budget policy; independent security certification has not been performed.",
                "adapter_initialization_requirements": "Fresh PEFT default adapter initialization only: A is uniform within +/-1/sqrt(product(shape[1:])); B is exactly zero. Unknown parameter patterns fail closed. Client checks the declared bounds before encryption."
                if adapter_policy == "initialization"
                else "All adapter tensors must fit input_public_bound, including updated or imported adapters.",
            }
        )
        torch.set_num_threads(2)
        stage = "capture"
        capture_options = {
            "snapshot": args.snapshot,
            "profile": args.profile,
            "max_report_nodes": 0,
        }
        if report["data_seed"] is not None:
            capture_options["data_seed"] = report["data_seed"]
        captured = capture_training_graph(**capture_options)
        report["graph"] = captured.metadata
        range_contracts = None
        range_path = getattr(args, "public_range_contract", None)
        if range_path is not None:
            stage = "range_contract_validation"
            from .calibration import load_range_contract

            range_contracts = load_range_contract(range_path, captured)
            calibration_overlap = (
                report["data_seed"]
                in range_contracts["provenance"]["calibration_seeds"]
                if report["data_seed"] is not None
                else None
            )
            report["public_range_contract"] = {
                "source": range_contracts["source"],
                "provenance": range_contracts["provenance"],
                "empirical_not_guarantee": True,
                "calibration_overlap": calibration_overlap,
                "held_out": not calibration_overlap
                if calibration_overlap is not None
                else None,
                "held_out_scope": "Pixel fixture seed only; model/adapters and instruction/label fixtures are shared.",
                "validation_scope": "Terminal numerical agreement on this fixture is conditional on the supplied empirical public bounds; it does not prove those bounds for arbitrary private data.",
                "application": "Only tightens declared bounds; no value clamping or calibration from this run.",
            }
            report["validation_conditioned_on_public_range_contract"] = True
        stage = "setup"
        if reference:
            backend = ReferenceVectorBackend()
            client = backend
        else:
            from fhenomenon.backends.runtime import create_runtime_context

            client, backend = create_runtime_context(
                profile=crypto_profile,
                device=args.backend.split("-")[-1],
                rotations=[1 << i for i in range(args.slots.bit_length() - 1)],
                mod_raise_levels=[args.refresh_target],
                max_live_bytes=args.max_live_bytes,
            )
        engine = TensorEngine(
            backend,
            client=client,
            slots=args.slots,
            target_level=args.refresh_target,
            exact_reference=args.backend == "reference-exact",
        )
        stage = "model_compilation"
        program = compile_model(
            captured,
            registry=None
            if getattr(args, "decomposition_only", False)
            else native_tensor_registry(),
            environment=runtime_environment(engine),
        )
        report["kernel_requests"] = program.kernel_requests()
        if not program.supported:
            raise NotImplementedError(
                "unsupported secret operations: "
                + ", ".join(sorted({node.operation for node in program.unsupported}))
            )
        adapters, frozen, buffers, tokens, pixels, targets = captured.args
        # Keep the execution budget/timing boundary before input encryption;
        # setup failures still have the initial timer and a preserved report.
        started = time.perf_counter()
        last_report[0] = started
        report["setup_seconds"] = started - run_started
        stage = "input_encryption"
        adapter_bounds = _adapter_public_bounds(adapters, adapter_policy, public_bound)
        report["adapter_public_bounds"] = adapter_bounds
        bound = (
            {
                k: engine.encrypt(v, public_bound=adapter_bounds[k])
                for k, v in adapters.items()
            },
            frozen,
            buffers,
            engine.encrypt(tokens, public_bound=public_bound),
            engine.encrypt(pixels, public_bound=public_bound),
            engine.encrypt(targets, public_bound=public_bound),
        )
        stage = "graph_execution"
        report["encrypted_execution"] = not reference
        duration_ms = 0
        if args.max_seconds:
            remaining = args.max_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError(
                    "execution time budget exhausted before scope capture"
                )
            duration_ms = max(1, math.ceil(remaining * 1000))
        actual = program.execute(
            engine,
            bound,
            progress=progress,
            range_contracts=range_contracts,
            range_policy=RANGE_POLICY,
            scope_options={
                "max_commands": getattr(args, "scope_max_commands", 1000000),
                "max_values": getattr(args, "scope_max_values", 2000000),
                "max_input_edges": getattr(args, "scope_max_input_edges", 2000000),
                "max_exports": getattr(args, "scope_max_exports", 100000),
                "max_duration_ms": duration_ms,
            },
        )
        if not reference:
            scope = backend.runtime_stats["last_scope"]
            complete = scope["commands"] == scope["executed_commands"]
            report["scope_coverage"] = {**scope, "complete": complete}
            report["scope_timing"] = getattr(backend, "last_scope_timing", None)
            if not complete:
                raise RuntimeError(
                    "training scope did not execute every captured command"
                )
        report["execution_completed"] = True
        report["kernel_cost_measurement"] = program.measurement()
        stage = "terminal_validation"
        calls_before_egress = (
            None if reference else dict(backend.runtime_stats["recipe_calls"])
        )
        before = client.stats.copy()
        decrypt_count = before.get("decrypt_calls", before.get("terminal_decryptions"))
        if decrypt_count != 0:
            raise RuntimeError("decryption occurred before training outputs")
        report["seconds_before_egress"] = time.perf_counter() - started
        report["validation"] = {
            "loss": compare(engine, actual[0], captured.eager_result[0]),
            "gradients": compare(engine, actual[1], captured.eager_result[1]),
            "updates": compare(engine, actual[2], captured.eager_result[2]),
            "decryptions_before_egress": decrypt_count,
        }
        if not reference:
            calls_after_egress = backend.runtime_stats["recipe_calls"]
            report["validation"]["evaluator_calls_during_egress"] = sum(
                calls_after_egress.values()
            ) - sum(calls_before_egress.values())
            if calls_after_egress != calls_before_egress:
                raise RuntimeError(
                    "terminal validation performed evaluator work outside the training scope"
                )
        v = report["validation"]
        report["passed"] = all(
            v[k]["all_finite"] and v[k]["max_abs_error"] <= tol
            for k, tol in [("loss", 0.005), ("gradients", 0.005), ("updates", 5e-6)]
        )
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            {
                "passed": False,
                "failure": type(exc).__name__ + ": " + str(exc),
                "failure_stage": stage,
                "aborted": isinstance(exc, KeyboardInterrupt),
                "progress": progress_state,
                "traceback": traceback.format_exc(),
            }
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    report["total_elapsed_seconds"] = time.perf_counter() - run_started
    for name, owner, attribute in (
        ("engine", engine, "stats"),
        ("interpreter", program, "execution_report"),
        ("provider", backend, "stats"),
        ("client", client, "stats"),
        ("runtime", backend, "runtime_stats"),
    ):
        if owner is None:
            continue
        try:
            value = getattr(owner, attribute, None)
            if value is not None:
                report[name] = value
        except Exception as exc:
            failure = type(exc).__name__ + ": " + str(exc)
            report.setdefault("diagnostics_errors", {})[name] = failure
            report["passed"] = False
            if "failure" not in report:
                report.update({"failure": failure, "failure_stage": "diagnostics"})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    conditional_scope = (
        report["public_range_contract"]["validation_scope"]
        if report["public_range_contract"] is not None
        else None
    )
    print(
        json.dumps(
            {
                "output": str(path),
                "passed": report["passed"],
                "completed": report["execution_completed"],
                "cryptographic_test_only": report.get("cryptographic_test_only"),
                "validation_conditioned_on_public_range_contract": report[
                    "validation_conditioned_on_public_range_contract"
                ],
                "conditional_scope": conditional_scope,
                "seconds": report["elapsed_seconds"],
                "failure": report.get("failure"),
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    p.add_argument("--profile", choices=["micro", "tiny"], default="micro")
    p.add_argument(
        "--backend",
        choices=[
            "reference-exact",
            "reference-polynomial",
            "runtime-cpu",
            "runtime-cuda",
        ],
        default="runtime-cuda",
    )
    p.add_argument("--slots", type=int, default=256)
    p.add_argument(
        "--decomposition-only",
        action="store_true",
        help="Disable contributed semantic kernels for controlled baseline comparisons",
    )
    p.add_argument("--refresh-target", type=int, choices=range(9), default=8)
    p.add_argument(
        "--input-public-bound",
        type=float,
        default=1.0,
        help="Predeclared bound for private data and adapters under the uniform policy",
    )
    p.add_argument(
        "--adapter-bound-policy",
        choices=["uniform", "initialization"],
        default="initialization",
        help="Use the fresh LoRA initializer bounds, or a declared uniform adapter bound",
    )
    p.add_argument(
        "--crypto-profile",
        choices=["Native128", "NativeTest"],
        default="Native128",
        help="NativeTest is explicitly insecure and only for numerical tests",
    )
    p.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Optional pixel fixture seed; omitted preserves the original fixed fixture",
    )
    p.add_argument(
        "--public-range-contract",
        default=None,
        help="Explicit empirical public range profile; never calibrated from this run",
    )
    p.add_argument("--max-live-bytes", type=int, default=0)
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0,
        help="Input/capture/execution budget; checked at scope span boundaries, not inside a kernel",
    )
    p.add_argument("--scope-max-commands", type=int, default=1000000)
    p.add_argument("--scope-max-values", type=int, default=2000000)
    p.add_argument("--scope-max-input-edges", type=int, default=2000000)
    p.add_argument("--scope-max-exports", type=int, default=100000)
    p.add_argument("--output", required=True)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
