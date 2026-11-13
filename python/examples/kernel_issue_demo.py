"""Measure a small encrypted training Scope for the issue-automation example.

This uses the ordinary model in model_training.py, not a hard-coded request or
synthetic latency. NativeTest is insecure; this is not full OpenVLA training.
Run from the repository root with the SDK and native provider available.
"""

import argparse
import copy
import json
from pathlib import Path
import platform
import statistics

import torch

from fhenomenon.backends.runtime import create_runtime_context
from fhenomenon.ml import TensorEngine, price_kernel_requests
from python.examples.model_training import build_program


def run_demo(*, samples=3):
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    client, backend = create_runtime_context(
        profile="NativeTest", device="cpu", mod_raise_levels=[8]
    )
    try:
        engine = TensorEngine(backend, client=client, slots=16)
        program = build_program(engine)
        measurements = []
        maximum_error = 0.0

        def check(actual, expected):
            nonlocal maximum_error
            if isinstance(actual, dict):
                if actual.keys() != expected.keys():
                    raise ValueError("output parameter names differ")
                for name in actual:
                    check(actual[name], expected[name])
                return
            value = torch.tensor(engine.decrypt(actual), dtype=torch.float64)
            expected = expected.double()
            torch.testing.assert_close(value, expected, atol=2e-5, rtol=2e-5)
            maximum_error = max(maximum_error, float((value - expected).abs().max()))

        for _ in range(samples):
            encrypted = program.encrypt_inputs(engine, public_bound=2.0)
            before = client.stats["decrypt_calls"]
            outputs = program.execute(engine, encrypted)
            if (
                client.stats["decrypt_calls"] != before
                or backend.stats["evaluator_decrypt_calls"] != 0
            ):
                raise ValueError("unexpected intermediate decryption")
            scope = backend.runtime_stats["last_scope"]
            if (
                scope["status"] != "completed"
                or scope["commands"] != scope["executed_commands"]
                or program.execution_report["private_host_fallback_count"] != 0
            ):
                raise ValueError("training did not complete entirely in the Scope")
            evidence = program.measurement()
            if evidence is None:
                raise ValueError("provider did not supply a valid measurement")
            measurements.append(evidence)
            for actual, expected in zip(
                outputs, program.captured.eager_result, strict=True
            ):
                check(actual, expected)

        # All timings are arithmetic means from exactly the same sample cohort.
        combined = copy.deepcopy(measurements[0])
        for evidence in measurements[1:]:
            if evidence["request_fingerprint"] != combined["request_fingerprint"]:
                raise ValueError("request identity changed between samples")
        combined["sample_count"] = samples
        combined["whole_scope_wall_seconds"] = statistics.fmean(
            m["whole_scope_wall_seconds"] for m in measurements
        )
        for row in combined["node_measurements"]:
            values = []
            for evidence in measurements:
                found = next(
                    r for r in evidence["node_measurements"] if r["node"] == row["node"]
                )
                if (found["signature"], found["current_implementation"]) != (
                    row["signature"],
                    row["current_implementation"],
                ):
                    raise ValueError("measured realization changed between samples")
                values.append(found["seconds"])
            row["seconds"] = statistics.fmean(values)
        requests = program.kernel_requests()
        price_kernel_requests(requests, combined, [])
        return {
            "passed": True,
            "encrypted_execution": True,
            "kernel_requests": requests,
            "kernel_cost_measurement": combined,
            "scope_coverage": {**scope, "complete": True},
            "validation": {
                "samples_checked": samples,
                "max_absolute_error": maximum_error,
                "intermediate_decryptions": 0,
                "private_host_fallbacks": 0,
            },
            "issue_context": {
                "workload": "Ordinary PyTorch frozen linear + encrypted rank-1 adapters; forward, loss, backward and SGD in one Scope",
                "hardware": {
                    "machine": platform.machine(),
                    "system": platform.system(),
                    "torch": torch.__version__,
                    "device": "cpu",
                },
                "limitations": "Small random numerical fixture, NativeTest (insecure), 3 input and 2 output features; not OpenVLA/7B execution or a representative deployment benchmark. No warm-up samples are discarded.",
                "numerical_acceptance": {
                    "final_loss_gradient_update_atol": 2e-5,
                    "final_loss_gradient_update_rtol": 2e-5,
                    "encrypted_input_public_absolute_bound": 2.0,
                    "kernel_error_budget": "must be established within these final-output tolerances",
                },
                "reproduce": "Build fhn_tensor_runtime in Release; install ./python[ml] with torch==2.7.1; set FHN_LIBRARY_DIR to build/lib; python -m python.examples.kernel_issue_demo --output build/kernel-request-report.json",
            },
        }
    finally:
        client.close()
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--source-url")
    parser.add_argument("--evidence-url")
    args = parser.parse_args()
    report = run_demo(samples=args.samples)
    for key in ("source_url", "evidence_url"):
        if getattr(args, key):
            report["issue_context"][key] = getattr(args, key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "samples": args.samples,
                "scope_seconds": report["kernel_cost_measurement"][
                    "whole_scope_wall_seconds"
                ],
                "report": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
