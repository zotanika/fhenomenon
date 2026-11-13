"""Runner failure/report regressions without model loading or native key setup."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import run_training as runner


def fixture():
    adapters = {"adapter": torch.tensor([0.25])}
    data = torch.tensor([1.0])
    expected = (
        torch.tensor(0.5),
        {"adapter": torch.tensor([0.125])},
        {"adapter": torch.tensor([0.249875])},
    )
    return SimpleNamespace(
        args=(adapters, {}, {}, data, data, data),
        metadata={"profile": "runner_fixture"},
        eager_result=expected,
    )


def public_range_fixture():
    return {
        "provenance": {
            "kind": "public_empirical_calibration",
            "guarantee": "empirical_not_proof",
            "training_data_used": False,
            "calibration_seeds": [101, 102, 103, 104],
            "model_seed": 7,
            "profile": "micro",
            "headroom": 4.0,
            "floor": 0.01,
        }
    }


class FakeEngine:
    """Only external execution is replaced; run/compare/report writing are real."""

    def __init__(self, backend, *, client, **kwargs):
        self.backend, self.client = backend, client
        self.stats = {"encrypted_inputs": 0, "scoped_runs": 0}
        self.in_scope = False

    def run_scope(self, capture, **options):
        self.in_scope = True
        try:
            output = capture()
        finally:
            self.in_scope = False
        self.stats["scoped_runs"] += 1
        self.backend.runtime_stats["last_scope"] = {
            "commands": 1,
            "executed_commands": 1,
        }
        return output

    def encrypt(self, value, *, public_bound=None):
        if (
            self.backend.required_bound is not None
            and public_bound != self.backend.required_bound
        ):
            raise ValueError("missing predeclared input bound")
        self.stats["encrypted_inputs"] += 1
        if self.backend.encrypt_failure:
            raise RuntimeError("Runtime memory budget exceeded")
        return value

    def decrypt(self, value):
        self.client.stats["decrypt_calls"] += 1
        return np.asarray(value)


class FakeInterpreter:
    def __init__(self, engine):
        self.engine = engine
        self.stats = {}

    def run(self, captured, bound, progress, range_contracts=None):
        if not self.engine.in_scope:
            raise RuntimeError("training interpreter executed outside scope capture")
        if range_contracts != self.engine.backend.range_contract:
            raise ValueError("validated public range contract was not forwarded")
        self.stats["public_range_contract_supplied"] = range_contracts is not None
        if self.engine.backend.execution_failure is not None:
            raise self.engine.backend.execution_failure
        if self.engine.backend.output is not None:
            return self.engine.backend.output
        return captured.eager_result


class FakeProgram:
    supported = True

    def __init__(self, captured, **options):
        self.captured = captured
        self.execution_report = {}

    def kernel_requests(self):
        return {
            "requests": [],
            "latency_estimates": "unpriced_without_matching_measurements",
        }

    def measurement(self):
        return None

    def execute(
        self, engine, bound, *, progress, range_contracts, range_policy, scope_options
    ):
        interpreter = FakeInterpreter(engine)
        try:
            return engine.run_scope(
                lambda: interpreter.run(
                    self.captured, bound, progress, range_contracts
                ),
                **scope_options,
            )
        finally:
            self.execution_report = interpreter.stats


class RunnerTests(unittest.TestCase):
    def execute(
        self,
        *,
        capture_failure=None,
        setup_failure=None,
        encrypt_failure=False,
        output=None,
        expected_output=None,
        public_bound=1.0,
        crypto_profile="Native128",
        execution_failure=None,
        data_seed=None,
        range_contract=None,
        range_failure=None,
    ):
        captured = fixture()
        if expected_output is not None:
            captured.eager_result = expected_output
        client = SimpleNamespace(stats={"decrypt_calls": 0})
        backend = SimpleNamespace(
            stats={"native_calls": 0},
            runtime_stats={"live_bytes": 0, "recipe_calls": {}},
            encrypt_failure=encrypt_failure,
            output=output,
            execution_failure=execution_failure,
            range_contract=range_contract,
            required_bound=public_bound if public_bound != 1.0 else None,
        )

        def create_context(**options):
            if options.get("profile") != crypto_profile:
                raise ValueError("requested crypto profile was not forwarded")
            return client, backend

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            path = Path(directory) / "result.json"
            range_path = Path(directory) / "public_ranges.json"
            if range_contract is not None:
                range_path.write_text('{"fixture":"public calibration"}\n')
                range_contract["source"] = {
                    "path": str(range_path.resolve()),
                    "sha256": hashlib.sha256(range_path.read_bytes()).hexdigest(),
                }
            args = SimpleNamespace(
                snapshot="unused",
                profile="micro",
                backend="runtime-cpu",
                slots=2,
                refresh_target=8,
                max_live_bytes=1,
                max_seconds=0,
                output=str(path),
                input_public_bound=public_bound,
                adapter_bound_policy="uniform",
                crypto_profile=crypto_profile,
                data_seed=data_seed,
                public_range_contract=str(range_path)
                if range_contract is not None or range_failure
                else None,
            )

            def load_ranges(requested_path, requested_capture):
                self.assertEqual(requested_path, str(range_path))
                self.assertIs(requested_capture, captured)
                if range_failure is not None:
                    raise range_failure
                return range_contract

            loader = SimpleNamespace(load_range_contract=load_ranges)
            stack.enter_context(
                patch.dict(
                    "sys.modules", {"examples.openvla_training.calibration": loader}
                )
            )
            capture = stack.enter_context(
                patch.object(
                    runner,
                    "capture_training_graph",
                    return_value=captured,
                    side_effect=capture_failure,
                )
            )
            stack.enter_context(
                patch(
                    "fhenomenon.backends.runtime.create_runtime_context",
                    side_effect=setup_failure
                    if setup_failure is not None
                    else create_context,
                )
            )
            stack.enter_context(patch.object(runner, "TensorEngine", FakeEngine))
            stack.enter_context(patch.object(runner, "compile_model", FakeProgram))
            stack.enter_context(
                patch.object(runner, "runtime_environment", lambda engine: {})
            )
            output_text = stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                code = runner.run(args)
            except (Exception, KeyboardInterrupt) as exc:
                self.fail(f"runner failed without preserving its report: {exc}")
            if capture.called:
                if data_seed is None:
                    self.assertNotIn("data_seed", capture.call_args.kwargs)
                else:
                    self.assertEqual(capture.call_args.kwargs["data_seed"], data_seed)
            self.assertTrue(path.is_file(), "requested failure report was not written")
            self.last_stdout_summary = json.loads(
                output_text.getvalue().splitlines()[-1]
            )
            return code, json.loads(path.read_text())

    def assert_failure(self, result, message, stage):
        code, report = result
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertFalse(report["execution_completed"])
        self.assertIn(message, report["failure"])
        self.assertEqual(report["failure_stage"], stage)
        self.assertEqual(
            report["validation_thresholds"],
            {"loss_atol": 0.005, "gradient_atol": 0.005, "update_atol": 5e-6},
        )

    def test_capture_failure_writes_requested_report(self):
        self.assert_failure(
            self.execute(capture_failure=RuntimeError("snapshot unavailable")),
            "snapshot unavailable",
            "capture",
        )

    def test_runtime_setup_failure_writes_requested_report(self):
        self.assert_failure(
            self.execute(setup_failure=RuntimeError("CUDA unavailable")),
            "CUDA unavailable",
            "setup",
        )

    def test_input_encryption_budget_failure_writes_requested_report(self):
        result = self.execute(encrypt_failure=True)
        self.assert_failure(
            result, "Runtime memory budget exceeded", "input_encryption"
        )
        self.assertEqual(result[1]["client"]["decrypt_calls"], 0)
        self.assertEqual(result[1]["engine"]["encrypted_inputs"], 1)

    def test_explicit_public_bound_is_used_before_encryption(self):
        code, report = self.execute(public_bound=4.0)
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["input_public_bound"], 4.0)
        self.assertEqual(report["engine"]["encrypted_inputs"], 4)
        self.assertEqual(report["validation"]["decryptions_before_egress"], 0)

    def test_training_executes_through_one_scope_before_terminal_validation(self):
        code, report = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(report["execution_mode"], "single_fhenomenon_scope")
        self.assertEqual(report["engine"]["scoped_runs"], 1)
        self.assertTrue(report["scope_coverage"]["complete"])
        self.assertEqual(report["validation"]["evaluator_calls_during_egress"], 0)

    def test_insecure_test_crypto_profile_is_forwarded_and_reported(self):
        code, report = self.execute(crypto_profile="NativeTest")
        self.assertEqual(code, 0)
        self.assertEqual(report["crypto_profile"], "NativeTest")
        self.assertTrue(report["cryptographic_test_only"])
        self.assertTrue(self.last_stdout_summary["cryptographic_test_only"])
        self.assertIn("insecure", report["cryptographic_profile_scope"])
        self.assertEqual(report["adapter_bound_policy"], "uniform")

    def test_explicit_data_seed_is_forwarded_and_reported(self):
        code, report = self.execute(data_seed=2029)
        self.assertEqual(code, 0)
        self.assertEqual(report["data_seed"], 2029)

    def test_default_fixture_has_no_implicit_range_contract(self):
        code, report = self.execute()
        self.assertEqual(code, 0)
        self.assertIsNone(report["data_seed"])
        self.assertIsNone(report["public_range_contract"])
        self.assertFalse(report["validation_conditioned_on_public_range_contract"])
        self.assertFalse(report["interpreter"]["public_range_contract_supplied"])

    def test_explicit_empirical_range_profile_is_validated_forwarded_and_identified(
        self,
    ):
        profile = public_range_fixture()
        code, report = self.execute(range_contract=profile, data_seed=2029)
        self.assertEqual(code, 0)
        self.assertTrue(report["interpreter"]["public_range_contract_supplied"])
        self.assertTrue(report["validation_conditioned_on_public_range_contract"])
        recorded = report["public_range_contract"]
        self.assertEqual(recorded["source"], profile["source"])
        self.assertEqual(recorded["provenance"], profile["provenance"])
        self.assertTrue(recorded["empirical_not_guarantee"])
        self.assertFalse(recorded["calibration_overlap"])
        self.assertTrue(recorded["held_out"])
        self.assertIn("Pixel fixture seed only", recorded["held_out_scope"])
        self.assertTrue(
            self.last_stdout_summary["validation_conditioned_on_public_range_contract"]
        )
        self.assertEqual(
            self.last_stdout_summary["conditional_scope"], recorded["validation_scope"]
        )
        self.assertEqual(
            report["validation_thresholds"],
            {"loss_atol": 0.005, "gradient_atol": 0.005, "update_atol": 5e-6},
        )
        self.assertEqual(report["validation"]["gradients"]["relative_l2_error"], 0.0)

    def test_calibration_overlap_is_reported_without_rejecting_explicit_reuse(self):
        code, report = self.execute(
            range_contract=public_range_fixture(), data_seed=101
        )
        self.assertEqual(code, 0)
        self.assertTrue(report["public_range_contract"]["calibration_overlap"])
        self.assertFalse(report["public_range_contract"]["held_out"])

    def test_undefined_data_seed_is_not_described_as_held_out(self):
        code, report = self.execute(range_contract=public_range_fixture())
        self.assertEqual(code, 0)
        self.assertIsNone(report["public_range_contract"]["calibration_overlap"])
        self.assertIsNone(report["public_range_contract"]["held_out"])

    def test_mismatched_range_profile_fails_before_runtime_setup(self):
        result = self.execute(
            range_failure=ValueError("public weights fingerprint mismatch"),
            setup_failure=AssertionError("runtime setup must not begin"),
        )
        self.assert_failure(
            result, "public weights fingerprint mismatch", "range_contract_validation"
        )
        self.assertNotIn("engine", result[1])
        self.assertFalse(result[1]["encrypted_execution"])

    def test_keyboard_interrupt_preserves_abort_report(self):
        result = self.execute(execution_failure=KeyboardInterrupt())
        self.assert_failure(result, "KeyboardInterrupt", "graph_execution")
        self.assertTrue(result[1]["aborted"])

    def test_nonfinite_terminal_output_is_a_reported_validation_failure(self):
        output = (
            torch.tensor(0.5),
            {"adapter": torch.tensor([float("nan")])},
            {"adapter": torch.tensor([0.249875])},
        )
        code, report = self.execute(output=output)
        self.assertEqual(code, 1)
        self.assertTrue(report["execution_completed"])
        self.assertFalse(report["passed"])
        self.assertFalse(report["validation"]["gradients"]["all_finite"])
        self.assertIsNone(report["validation"]["gradients"]["max_abs_error"])

    def test_small_gradient_diagnostics_do_not_replace_absolute_pass_criteria(self):
        expected = (
            torch.tensor(0.5),
            {"adapter": torch.tensor([1e-8], dtype=torch.float64)},
            {"adapter": torch.tensor([0.25])},
        )
        actual = (expected[0], {"adapter": torch.zeros(1)}, expected[2])
        code, report = self.execute(output=actual, expected_output=expected)
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        gradient = report["validation"]["gradients"]
        self.assertEqual(gradient["relative_l2_error"], 1.0)
        self.assertIsNone(gradient["cosine_similarity"])
        self.assertEqual(report["validation_thresholds"]["gradient_atol"], 0.005)


class ComparisonTests(unittest.TestCase):
    def test_small_gradient_diagnostics_expose_complete_signal_loss(self):
        engine = SimpleNamespace(decrypt=lambda value: np.asarray(value))
        expected = torch.tensor([1e-8, -1e-8], dtype=torch.float64)
        result = runner.compare(engine, np.zeros(2), expected)
        self.assertLess(result["max_abs_error"], 0.005)
        self.assertEqual(result["expected_max_abs"], 1e-8)
        self.assertEqual(result["relative_l2_error"], 1.0)
        self.assertIsNone(result["cosine_similarity"])

    def test_dictionary_diagnostics_use_combined_vectors(self):
        engine = SimpleNamespace(decrypt=lambda value: np.asarray(value))
        got = {"small": np.array([-1e-8, 0.0]), "large": np.array([0.0, 3e-8])}
        expected = {
            "small": torch.tensor([1e-8, 0.0], dtype=torch.float64),
            "large": torch.tensor([0.0, 3e-8], dtype=torch.float64),
        }
        result = runner.compare(engine, got, expected)
        self.assertEqual(result["expected_max_abs"], 3e-8)
        self.assertAlmostEqual(result["relative_l2_error"], 0.6324555320336759)
        self.assertAlmostEqual(result["cosine_similarity"], 0.8)
        self.assertAlmostEqual(result["tensors"]["small"]["cosine_similarity"], -1.0)

    def test_zero_reference_diagnostics_are_json_safe(self):
        engine = SimpleNamespace(decrypt=lambda value: np.asarray(value))
        expected = torch.zeros(2, dtype=torch.float64)
        for actual, relative in ((np.zeros(2), 0.0), (np.ones(2), None)):
            with self.subTest(actual=actual.tolist()):
                result = runner.compare(engine, actual, expected)
                self.assertEqual(result["expected_max_abs"], 0.0)
                self.assertEqual(result["relative_l2_error"], relative)
                self.assertIsNone(result["cosine_similarity"])
                json.dumps(result, allow_nan=False)


class AdapterBoundTests(unittest.TestCase):
    def test_initialization_bounds_follow_public_linear_and_convolution_fan_in(self):
        adapters = {
            "linear.lora_A.default.weight": torch.full((2, 16), 99.0),
            "conv.lora_A.default.weight": torch.zeros(2, 3, 4, 4),
            "linear.lora_B.default.weight": torch.zeros(16, 2),
        }
        bounds = runner._adapter_public_bounds(adapters, "initialization", 7.0)
        self.assertEqual(bounds["linear.lora_A.default.weight"], 0.25)
        self.assertAlmostEqual(
            bounds["conv.lora_A.default.weight"], 0.14433756729740646
        )
        self.assertEqual(bounds["linear.lora_B.default.weight"], 0.0)

    def test_nonzero_lora_b_is_rejected_by_client_bound_validation(self):
        from fhenomenon.ml.tensor import ReferenceVectorBackend, TensorEngine

        name = "linear.lora_B.default.weight"
        adapters = {name: torch.full((4, 2), 0.125)}
        bounds = runner._adapter_public_bounds(adapters, "initialization", 1.0)
        engine = TensorEngine(ReferenceVectorBackend(), slots=8)
        with self.assertRaisesRegex(ValueError, "declared public bound"):
            engine.encrypt(adapters[name], public_bound=bounds[name])

    def test_unknown_initializer_fails_closed_but_uniform_policy_is_explicit(self):
        adapters = {"unknown.weight": torch.zeros(2, 3)}
        with self.assertRaisesRegex(ValueError, "initialization"):
            runner._adapter_public_bounds(adapters, "initialization", 1.0)
        self.assertEqual(
            runner._adapter_public_bounds(adapters, "uniform", 4.0),
            {"unknown.weight": 4.0},
        )

    def test_real_fresh_micro_capture_satisfies_initializer_bounds(self):
        from fhenomenon.ml.tensor import ReferenceVectorBackend, TensorEngine

        captured = runner.capture_training_graph(profile="micro", max_report_nodes=0)
        self.assertTrue(captured.validation["passed"])
        adapters = captured.args[0]
        bounds = runner._adapter_public_bounds(adapters, "initialization", 1.0)
        engine = TensorEngine(ReferenceVectorBackend(), slots=256)
        for name, value in adapters.items():
            with self.subTest(adapter=name):
                encrypted = engine.encrypt(value, public_bound=bounds[name])
                self.assertEqual(encrypted.public_bound, bounds[name])
        self.assertTrue(any(value == 0.0 for value in bounds.values()))
        self.assertTrue(any(0.0 < value < 1.0 for value in bounds.values()))

    def test_nonfinite_reference_and_overflow_do_not_break_dictionary_aggregation(self):
        engine = SimpleNamespace(decrypt=lambda value: np.asarray(value))
        for invalid in (float("nan"), float("inf"), -1e308):
            with self.subTest(expected=invalid):
                got = {"bad": np.array([1e308]), "valid": np.array([0.0])}
                expected = {
                    "bad": torch.tensor([invalid], dtype=torch.float64),
                    "valid": torch.tensor([0.0]),
                }
                result = runner.compare(engine, got, expected)
                self.assertFalse(result["all_finite"])
                self.assertIsNone(result["max_abs_error"])
                self.assertIsNone(result["relative_l2_error"])
                self.assertIsNone(result["cosine_similarity"])
                json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
