"""Real encrypted semantic-kernel contracts, publication and accounting tests."""

import ctypes as C
import gc
import math
import os
import unittest

from fhenomenon.backends.runtime import create_runtime_context


class TensorKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client, cls.evaluator = create_runtime_context(
            profile="NativeTest",
            device=os.environ.get("OPENVLA_SCOPE_TEST_DEVICE", "cpu"),
            mod_raise_levels=[],
        )

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.evaluator.close()

    def setUp(self):
        self.values = []
        self.assertTrue(
            callable(getattr(self.evaluator, "tensor_kernel", None)),
            "the Runtime must accept contracted tensor kernels",
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
        # Nonzero padding must never influence the logical 2x2 input.
        return self.keep(
            self.client.encrypt([0.25, -0.5, 0.75, 0.125, 0.9, -0.9, 0.8, -0.8])
        )

    @staticmethod
    def attributes():
        return {
            "rows": 2,
            "in_features": 2,
            "out_features": 3,
            "weights": [0.5, 0.25, -0.125, -0.25, 0.5, 0.75],
        }

    def assert_result(self, output):
        want = [0.25, -0.1875, -0.40625, 0.34375, 0.25, 0.0, 0.0, 0.0]
        got = self.client.decrypt(output)
        self.assertEqual(len(got), len(want))
        for actual, expected in zip(got, want):
            self.assertAlmostEqual(actual, expected, delta=2e-5)

    def test_eager_and_one_command_scope_compute_dense_linear_with_zero_padding(self):
        e, x = self.evaluator, self.source()
        initial = x.info.copy()
        eager = self.keep(
            e.tensor_kernel("tensor.public_linear", [x], self.attributes())[0]
        )
        self.assert_result(eager)
        arithmetic_before = e.stats["arithmetic_calls"]
        before = e.runtime_stats
        predicted = []

        def capture():
            output = self.keep(
                e.tensor_kernel("tensor.public_linear", [x], self.attributes())[0]
            )
            predicted.append(output.info)
            self.assertEqual(e.stats["arithmetic_calls"], arithmetic_before)
            return [output]

        output = e.run_scope(capture)[0]
        self.assert_result(output)
        self.assertEqual(x.info, initial)
        actual = output.info
        for field in (
            "count",
            "period",
            "level",
            "modulus_count",
            "estimated_bytes",
            "representation",
        ):
            self.assertEqual(predicted[0][field], actual[field])
        self.assertTrue(
            math.isclose(predicted[0]["scale"], actual["scale"], rel_tol=1e-12)
        )
        self.assertEqual(actual["level"], x.level - 1)
        stats = e.runtime_stats
        self.assertEqual(stats["last_scope"]["commands"], 1)
        self.assertEqual(stats["last_scope"]["kernel_spans"], 1)
        self.assertEqual(stats["last_scope"]["recipe_spans"], 0)
        self.assertEqual(stats["recipe_calls"], before["recipe_calls"])
        self.assertEqual(
            stats["kernel_calls"]["tensor.public_linear"],
            before["kernel_calls"]["tensor.public_linear"] + 1,
        )
        self.assertGreaterEqual(stats["kernel_seconds"]["tensor.public_linear"], 0)
        self.assertEqual(e.stats["evaluator_decrypt_calls"], 0)

    def test_invalid_contracts_preflight_before_any_arithmetic(self):
        e, x = self.evaluator, self.source()
        invalid = [
            {**self.attributes(), "rows": 0},
            {**self.attributes(), "rows": 1.5},
            {**self.attributes(), "out_features": 5},
            {**self.attributes(), "weights": [1.0]},
            {**self.attributes(), "weights": [float("nan")] * 6},
            {**self.attributes(), "weights": [float("inf")] * 6},
        ]
        for attributes in invalid:
            with self.subTest(attributes=attributes):
                before = e.stats["arithmetic_calls"]
                with self.assertRaises((RuntimeError, ValueError)):
                    e.run_scope(
                        lambda: e.tensor_kernel("tensor.public_linear", [x], attributes)
                    )
                self.assertEqual(e.stats["arithmetic_calls"], before)
                self.assertEqual(e.runtime_stats["last_scope"]["executed_commands"], 0)
        low = self.keep(e.level_down(x, 0))
        before = e.stats["arithmetic_calls"]
        with self.assertRaises(RuntimeError):
            e.tensor_kernel("tensor.public_linear", [low], self.attributes())
        self.assertEqual(e.stats["arithmetic_calls"], before)

    def test_caught_invalid_native_request_poisons_capture_and_exports(self):
        e, x = self.evaluator, self.source()
        escaped = []
        before = e.stats["arithmetic_calls"]

        def capture():
            result = self.keep(
                e.tensor_kernel("tensor.public_linear", [x], self.attributes())[0]
            )
            escaped.append(result)
            try:
                e.tensor_kernel("tensor.unknown", [x], {})
            except RuntimeError:
                pass
            return [result]

        with self.assertRaises(RuntimeError):
            e.run_scope(capture)
        self.assertEqual(e.stats["arithmetic_calls"], before)
        with self.assertRaises(RuntimeError):
            _ = escaped[0].info

    def test_nonexported_symbols_and_closed_inputs_are_rejected(self):
        e, x = self.evaluator, self.source()
        stale = []

        def capture():
            stale.extend(
                e.tensor_kernel("tensor.public_linear", [x], self.attributes())
            )
            self.values.extend(stale)
            return [x]

        e.run_scope(capture)
        before = e.stats["arithmetic_calls"]
        with self.assertRaises(RuntimeError):
            e.tensor_kernel("tensor.public_linear", stale, self.attributes())
        x.close()
        with self.assertRaises(RuntimeError):
            e.tensor_kernel("tensor.public_linear", [x], self.attributes())
        self.assertEqual(e.stats["arithmetic_calls"], before)

    def test_catalog_exposes_usable_contract_and_workspace_accounting(self):
        catalog = self.evaluator.kernel_catalog
        self.assertEqual(catalog["schema_version"], 1)
        kernel = next(
            item
            for item in catalog["kernels"]
            if item["operation"] == "tensor.public_linear"
        )
        self.assertEqual(kernel["input_arity"], 1)
        self.assertEqual(kernel["output_arity"], 1)
        self.assertEqual(kernel["level_consumption"], 1)
        self.assertEqual(kernel["representation"], "slots")
        self.assertIn("temporary_ciphertext_payload", kernel["workspace_accounting"])
        self.assertIn("rotation_key_basis", kernel["prepared_key_requirements"])

    def test_zero_matrix_still_consumes_one_level(self):
        e, x = self.evaluator, self.source()
        attributes = {**self.attributes(), "weights": [0.0] * 6}
        output = self.keep(
            e.run_scope(
                lambda: e.tensor_kernel("tensor.public_linear", [x], attributes)
            )[0]
        )
        self.assertEqual(output.level, x.level - 1)
        for actual in self.client.decrypt(output):
            self.assertAlmostEqual(actual, 0.0, delta=2e-5)

    def test_non_power_of_two_logical_count_and_last_level(self):
        e = self.evaluator
        source = self.keep(self.client.encrypt([0.25, -0.5, 0.75]))
        low = self.keep(e.level_down(source, 1))
        attributes = {
            "rows": 1,
            "in_features": 3,
            "out_features": 2,
            "weights": [0.5, 0.25, -0.25, 0.5, 0.125, -0.125],
        }
        output = self.keep(
            e.run_scope(
                lambda: e.tensor_kernel("tensor.public_linear", [low], attributes)
            )[0]
        )
        self.assertEqual(output.level, 0)
        self.assertEqual(output.count, 3)
        self.assertEqual(output.info["period"], 4)
        for actual, expected in zip(
            self.client.decrypt(output), [0.34375, -0.28125, 0.0]
        ):
            self.assertAlmostEqual(actual, expected, delta=2e-5)

    def test_insufficient_c_output_capacity_does_not_execute_or_publish(self):
        import json

        e, x = self.evaluator, self.source()
        before = e.stats["arithmetic_calls"]
        inputs = (C.c_void_p * 1)(x._handle)
        count = C.c_size_t(99)
        status = e._session.library.lib.fr_tensor_kernel(
            e._session.handle,
            b"tensor.public_linear",
            inputs,
            1,
            json.dumps(self.attributes()).encode(),
            None,
            0,
            C.byref(count),
        )
        self.assertNotEqual(status, 0)
        self.assertEqual(count.value, 0)
        self.assertEqual(e.stats["arithmetic_calls"], before)

    def test_workspace_budget_rejects_before_native_kernel_execution(self):
        x = self.source()
        # A second Runtime borrows the same evaluator library but owns its own
        # keys. The cap admits input+output, but not the diagonal workspace.
        client, e = create_runtime_context(
            profile="NativeTest",
            device=os.environ.get("OPENVLA_SCOPE_TEST_DEVICE", "cpu"),
            max_live_bytes=x.estimated_bytes * 2,
            mod_raise_levels=[],
        )
        source = client.encrypt([0.25, -0.5, 0.75, 0.125, 0.9, -0.9, 0.8, -0.8])
        try:
            before = e.stats["arithmetic_calls"]
            with self.assertRaises(RuntimeError):
                e.run_scope(
                    lambda: e.tensor_kernel(
                        "tensor.public_linear", [source], self.attributes()
                    )
                )
            self.assertEqual(e.stats["arithmetic_calls"], before)
            self.assertEqual(e.runtime_stats["live_resources"], 1)
        finally:
            source.close()
            client.close()
            e.close()

    def test_public_diagonal_buffers_are_also_budgeted(self):
        x = self.source()
        output_bytes = (
            x.estimated_bytes // x.info["modulus_count"] * (x.info["modulus_count"] - 1)
        )
        # This admits the declared input/output and ciphertext workspace exactly,
        # leaving no bytes for the kernel's own public diagonal buffers.
        client, e = create_runtime_context(
            profile="NativeTest",
            device=os.environ.get("OPENVLA_SCOPE_TEST_DEVICE", "cpu"),
            max_live_bytes=2 * x.estimated_bytes + 3 * output_bytes,
            mod_raise_levels=[],
        )
        source = client.encrypt([0.25, -0.5, 0.75, 0.125, 0.9, -0.9, 0.8, -0.8])
        try:
            before = e.stats["arithmetic_calls"]
            with self.assertRaises(RuntimeError):
                e.run_scope(
                    lambda: e.tensor_kernel(
                        "tensor.public_linear", [source], self.attributes()
                    )
                )
            self.assertEqual(e.stats["arithmetic_calls"], before)
            self.assertEqual(e.runtime_stats["live_resources"], 1)
        finally:
            source.close()
            client.close()
            e.close()


if __name__ == "__main__":
    unittest.main()
