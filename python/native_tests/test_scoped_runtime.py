"""Real ciphertext whole-graph Scope tests; no mocked arithmetic or scheduler."""

import ctypes as C
import gc
import math
import os
import time
import traceback
import unittest

from fhenomenon.backends.runtime import create_runtime_context


class ScopedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client, cls.evaluator = create_runtime_context(
            profile="NativeTest",
            device=os.environ.get("OPENVLA_SCOPE_TEST_DEVICE", "cpu"),
            rotations=[1, -1],
            mod_raise_levels=[2],
        )

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.evaluator.close()

    def setUp(self):
        self.values = []
        self.assertTrue(
            callable(getattr(self.evaluator, "run_scope", None)),
            "Evaluator.run_scope must capture the full operation graph",
        )

    def tearDown(self):
        for value in reversed(self.values):
            value.close()
        gc.collect()
        self.assertEqual(self.evaluator.runtime_stats["live_resources"], 0)

    def keep(self, value):
        self.values.append(value)
        return value

    def source(self):
        return self.keep(self.client.encrypt([0.125, -0.25, 0.5, 0.375]))

    def assert_values(self, value, expected, bound=1.0):
        for actual, wanted in zip(
            self.client.decrypt(value, public_bound=bound), expected
        ):
            self.assertAlmostEqual(actual, wanted, delta=2e-5)

    def test_capture_defers_all_recipes_and_materializes_same_export_handles(self):
        e, client = self.evaluator, self.client
        x = self.source()
        y = self.keep(client.encrypt([0.5, 0.25, -0.5, 0.125]))
        source_info = x.info.copy()
        before = e.runtime_stats
        arithmetic_before = e.stats["arithmetic_calls"]
        decryptions_before = client.stats["decrypt_calls"]
        captured_outputs = []

        def capture():
            summed = e.add(x, y)
            restored = e.sub(summed, y)
            product = e.mul(restored, y)
            shifted = e.add_plain(product, 0.25)
            scaled = e.mul_plain(shifted, [2.0])
            rotated = e.rotate(e.neg(scaled), 1)
            coeff = e.slot_to_coeff(x)
            self.assertEqual(coeff.representation, "coefficients")
            slots = self.keep(e.coeff_to_slot(coeff))
            low = e.level_down(rotated, 0)
            raised = self.keep(e.mod_raise(low, 2, public_bound=2.0))
            self.assertEqual(raised.level, 2)
            self.assertEqual(raised.count, 4)
            self.assertEqual(e.stats["arithmetic_calls"], arithmetic_before)
            self.assertEqual(e.runtime_stats["recipe_calls"], before["recipe_calls"])
            self.assertEqual(client.stats["decrypt_calls"], decryptions_before)
            self.assertFalse(e._session.library.lib.fr_value_borrow(raised._handle))
            captured_outputs.extend([raised, slots, raised])
            return captured_outputs

        outputs = e.run_scope(capture)
        self.assertIs(outputs, captured_outputs)
        self.assertIs(outputs[0], outputs[2])
        self.assertEqual(client.stats["decrypt_calls"], decryptions_before)
        self.assertEqual(e.stats["evaluator_decrypt_calls"], 0)
        self.assertEqual(x.info, source_info)
        self.assert_values(outputs[0], [-0.375, 0.0, -0.59375, -0.625], bound=2.0)
        self.assert_values(outputs[1], [0.125, -0.25, 0.5, 0.375])
        scope = e.runtime_stats["last_scope"]
        self.assertEqual(scope["commands"], 11)
        self.assertEqual(scope["executed_commands"], 11)
        self.assertEqual(scope["values"], 13)
        self.assertEqual(scope["input_edges"], 14)
        self.assertEqual(scope["imports"], 2)
        self.assertEqual(scope["exports"], 3)
        self.assertEqual(scope["recipe_spans"], 11)
        self.assertEqual(
            e.runtime_stats["completed_scopes"], before["completed_scopes"] + 1
        )
        self.assertEqual(e.runtime_stats["last_mod_raise_public_bound"], 2.0)
        self.assertEqual(e.stats["last_refresh_public_bound"], 2.0)
        for operation in (
            "add",
            "sub",
            "mul",
            "add_plain",
            "mul_plain",
            "neg",
            "rotate",
            "slot_to_coeff",
            "coeff_to_slot",
            "level_down",
            "mod_raise",
        ):
            key = "native.ckks." + operation
            self.assertEqual(
                e.runtime_stats["recipe_calls"][key],
                before["recipe_calls"].get(key, 0) + 1,
            )
        timing = e.last_scope_timing
        self.assertGreaterEqual(timing["capture_seconds"], 0)
        self.assertGreaterEqual(timing["execution_and_finalize_seconds"], 0)
        self.assertAlmostEqual(
            timing["total_seconds"],
            timing["capture_seconds"] + timing["execution_and_finalize_seconds"],
            places=8,
        )

    def test_late_invalid_plaintext_does_not_execute_earlier_valid_commands(self):
        e, x = self.evaluator, self.source()
        before = e.stats["arithmetic_calls"]
        resources = e.runtime_stats["live_resources"]
        escaped = []

        def capture():
            value = self.keep(e.neg(x))
            escaped.append(value)
            e.add_plain(value, [1.0, 2.0, 3.0])
            return [value]

        with self.assertRaises(RuntimeError):
            e.run_scope(capture)
        self.assertEqual(e.stats["arithmetic_calls"], before)
        self.assertEqual(e.runtime_stats["last_scope"]["executed_commands"], 0)
        self.assertEqual(e.runtime_stats["live_resources"], resources)
        with self.assertRaises(RuntimeError):
            _ = escaped[0].info
        result = self.keep(e.run_scope(lambda: [e.neg(x)])[0])
        self.assert_values(result, [-0.125, 0.25, -0.5, -0.375])

    def test_symbolic_metadata_matches_actual_outputs_for_every_recipe(self):
        e, x = self.evaluator, self.source()
        y = self.keep(self.client.encrypt([0.5, 0.25, -0.5, 0.125]))
        symbolic = []

        def capture():
            values = []

            def record(value):
                values.append(self.keep(value))
                symbolic.append(value.info)
                return value

            summed = record(e.add(x, y))
            difference = record(e.sub(summed, y))
            product = record(e.mul(difference, y))
            shifted = record(e.add_plain(product, 0.25))
            scaled = record(e.mul_plain(shifted, 2.0))
            negated = record(e.neg(scaled))
            rotated = record(e.rotate(negated, 1))
            low = record(e.level_down(rotated, 0))
            record(e.mod_raise(low, 2, public_bound=2.0))
            coeff = record(e.slot_to_coeff(x))
            record(e.coeff_to_slot(coeff))
            return values

        outputs = e.run_scope(capture)
        for predicted, value in zip(symbolic, outputs):
            actual = value.info
            self.assertTrue(predicted["symbolic"])
            self.assertFalse(actual["symbolic"])
            for field in (
                "level",
                "count",
                "period",
                "modulus_count",
                "representation",
                "estimated_bytes",
            ):
                self.assertEqual(predicted[field], actual[field], field)
            self.assertGreater(predicted["scale"], 0)
            self.assertTrue(
                math.isclose(predicted["scale"], actual["scale"], rel_tol=1e-12)
            )

    def test_callback_baseexception_preserves_identity_traceback_and_releases_capture(
        self,
    ):
        class AbortCapture(BaseException):
            pass

        e, x = self.evaluator, self.source()
        error = AbortCapture("stop this capture")
        before = e.stats["arithmetic_calls"]
        escaped = []

        def capture():
            escaped.append(self.keep(e.neg(x)))
            raise error

        try:
            e.run_scope(capture)
        except AbortCapture as caught:
            self.assertIs(caught, error)
            self.assertIn(
                "capture",
                [frame.name for frame in traceback.extract_tb(caught.__traceback__)],
            )
        else:
            self.fail(
                "the Python exception was swallowed at the ctypes callback boundary"
            )
        self.assertEqual(e.stats["arithmetic_calls"], before)
        with self.assertRaises(RuntimeError):
            e.neg(escaped[0])
        self.assertGreaterEqual(e.last_scope_timing["total_seconds"], 0)
        result = self.keep(e.run_scope(lambda: [e.neg(x)])[0])
        self.assert_values(result, [-0.125, 0.25, -0.5, -0.375])

    def test_client_boundaries_and_nested_scopes_reject_during_capture(self):
        e, client, x = self.evaluator, self.client, self.source()
        decryptions = client.stats["decrypt_calls"]
        imports = e.runtime_stats["imports"]

        def capture():
            value = self.keep(e.neg(x))
            for action in (
                lambda: client.decrypt(x),
                lambda: client.decrypt(value),
                lambda: client.encrypt([0.25]),
                lambda: e.run_scope(lambda: [x]),
            ):
                with self.assertRaisesRegex(RuntimeError, "captur|scope"):
                    action()
            return [value]

        result = e.run_scope(capture)[0]
        self.assertEqual(client.stats["decrypt_calls"], decryptions)
        self.assertEqual(e.runtime_stats["imports"], imports)
        self.assert_values(result, [-0.125, 0.25, -0.5, -0.375])

    def test_unexported_symbols_expire_but_concrete_inputs_and_exports_can_be_reused(
        self,
    ):
        e, x = self.evaluator, self.source()
        escaped = []

        def capture():
            escaped.append(self.keep(e.neg(x)))
            return [self.keep(e.neg(escaped[0])), x]

        result, unchanged = e.run_scope(capture)
        self.assertIs(unchanged, x)
        for action in (
            lambda: escaped[0].info,
            lambda: self.client.decrypt(escaped[0]),
            lambda: e.run_scope(lambda: [escaped[0]]),
        ):
            with self.assertRaises(RuntimeError):
                action()
        reused = self.keep(e.run_scope(lambda: [e.add(result, x)])[0])
        self.assert_values(reused, [0.25, -0.5, 1.0, 0.75])

    def test_plaintext_arrays_are_owned_by_capture_and_temporary_handles_can_close(
        self,
    ):
        e, x = self.evaluator, self.source()

        def capture():
            weights = [0.5, -0.5, 0.25, 0.0]
            temporary = e.mul_plain(x, weights)
            weights[:] = [10.0] * 4
            result = self.keep(e.add_plain(temporary, 0.125))
            temporary.close()
            return (result,)

        outputs = e.run_scope(capture)
        self.assertIsInstance(outputs, list)
        self.assert_values(outputs[0], [0.1875, 0.25, 0.25, 0.125])

    def test_budget_failures_execute_nothing_and_leave_session_reusable(self):
        e, x = self.evaluator, self.source()

        def capture():
            a = self.keep(e.add(x, x))
            b = self.keep(e.neg(a))
            return [a, b]

        for cap in ("max_commands", "max_values", "max_input_edges", "max_exports"):
            with self.subTest(cap=cap):
                before = e.stats["arithmetic_calls"]
                resources = e.runtime_stats["live_resources"]
                with self.assertRaises(RuntimeError):
                    e.run_scope(capture, **{cap: 1})
                self.assertEqual(e.stats["arithmetic_calls"], before)
                self.assertEqual(e.runtime_stats["live_resources"], resources)
        self.keep(e.run_scope(lambda: [e.neg(x)])[0])

    def test_scope_arguments_and_exports_fail_before_execution(self):
        e, x = self.evaluator, self.source()
        calls = []
        for name in ("max_commands", "max_values", "max_input_edges", "max_exports"):
            for value in (0, -1, True, 1.5, C.c_size_t(-1).value + 1):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        e.run_scope(lambda: calls.append(True), **{name: value})
        self.assertEqual(calls, [])
        for value in (-1, True, 1.5, C.c_size_t(-1).value + 1):
            with self.subTest(max_duration_ms=value):
                with self.assertRaises(ValueError):
                    e.run_scope(lambda: calls.append(True), max_duration_ms=value)
        self.assertEqual(calls, [])
        with self.assertRaises(TypeError):
            e.run_scope(None)
        for outputs in (None, x, [[x]], [object()]):
            with self.subTest(outputs=type(outputs).__name__):
                before = e.stats["arithmetic_calls"]
                with self.assertRaises((TypeError, ValueError)):
                    e.run_scope(lambda: outputs)
                self.assertEqual(e.stats["arithmetic_calls"], before)

    def test_capture_deadline_aborts_before_any_encrypted_arithmetic(self):
        e, x = self.evaluator, self.source()
        before = e.stats["arithmetic_calls"]

        def capture():
            result = self.keep(e.neg(x))
            time.sleep(0.025)
            return [result]

        with self.assertRaisesRegex(RuntimeError, "deadline|duration|time"):
            e.run_scope(capture, max_duration_ms=1)
        self.assertEqual(e.stats["arithmetic_calls"], before)
        self.keep(e.run_scope(lambda: [e.neg(x)], max_duration_ms=0)[0])

    def test_caught_native_contract_error_still_invalidates_entire_scope(self):
        e, x = self.evaluator, self.source()
        before = e.stats["arithmetic_calls"]

        def capture():
            result = self.keep(e.neg(x))
            with self.assertRaises(RuntimeError):
                e.mul_plain(result, [1.0, 2.0, 3.0])
            return [result]

        with self.assertRaises(RuntimeError):
            e.run_scope(capture)
        self.assertEqual(e.stats["arithmetic_calls"], before)

    def test_representation_mismatch_rejects_whole_graph_without_arithmetic(self):
        e, x = self.evaluator, self.source()
        before = e.stats["arithmetic_calls"]

        def capture():
            coefficients = self.keep(e.slot_to_coeff(x))
            return [e.add(coefficients, x)]

        with self.assertRaises(RuntimeError):
            e.run_scope(capture)
        self.assertEqual(e.stats["arithmetic_calls"], before)

    def test_other_context_operand_and_export_reject_without_partial_execution(self):
        e, x = self.evaluator, self.source()
        other_client, other = create_runtime_context(
            profile="NativeTest",
            device=os.environ.get("OPENVLA_SCOPE_TEST_DEVICE", "cpu"),
            mod_raise_levels=[],
        )
        foreign = other_client.encrypt([0.125, -0.25, 0.5, 0.375])
        try:
            for capture in (lambda: [e.add(x, foreign)], lambda: [e.neg(x), foreign]):
                before = e.stats["arithmetic_calls"]
                with self.assertRaises(ValueError):
                    e.run_scope(capture)
                self.assertEqual(e.stats["arithmetic_calls"], before)
        finally:
            foreign.close()
            other_client.close()
            other.close()

    def test_execution_memory_budget_failure_discards_all_scoped_outputs(self):
        x = self.source()
        client, evaluator = create_runtime_context(
            profile="NativeTest",
            device=os.environ.get("OPENVLA_SCOPE_TEST_DEVICE", "cpu"),
            mod_raise_levels=[],
            max_live_bytes=2 * x.estimated_bytes,
        )
        source = client.encrypt([0.125, -0.25, 0.5, 0.375])
        symbols = []
        try:

            def capture():
                first = evaluator.neg(source)
                symbols.append(first)
                second = evaluator.neg(first)
                symbols.append(second)
                return [first, second]

            with self.assertRaisesRegex(RuntimeError, "budget"):
                evaluator.run_scope(capture)
            self.assertEqual(evaluator.runtime_stats["live_resources"], 1)
            self.assertEqual(
                evaluator.runtime_stats["live_bytes"], source.estimated_bytes
            )
            for value in symbols:
                with self.assertRaises(RuntimeError):
                    _ = value.info
            for actual, expected in zip(
                client.decrypt(source), [0.125, -0.25, 0.5, 0.375]
            ):
                self.assertAlmostEqual(actual, expected, delta=2e-5)
        finally:
            for value in symbols:
                value.close()
            source.close()
            client.close()
            evaluator.close()


if __name__ == "__main__":
    unittest.main()
