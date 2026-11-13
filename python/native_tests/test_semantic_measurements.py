"""Semantic-node timing uses executed encrypted work, never capture duration."""

import gc
import os
import time
import unittest

from fhenomenon.backends.runtime import create_runtime_context


class SemanticMeasurementTests(unittest.TestCase):
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
            callable(getattr(self.evaluator, "semantic_node", None)),
            "runtime must preserve per-semantic-node origin during capture",
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
        return self.keep(self.client.encrypt([0.25, -0.5, 0.75, 0.125]))

    def test_capture_time_and_unattributed_output_work_are_excluded(self):
        e, x = self.evaluator, self.source()

        def capture():
            with e.semantic_node("encoder_linear", "shape-hash-a", "diagonal-native"):
                linear = self.keep(
                    e.tensor_kernel(
                        "tensor.public_linear",
                        [x],
                        {
                            "rows": 2,
                            "in_features": 2,
                            "out_features": 2,
                            "weights": [0.5, 0.25, -0.25, 0.5],
                        },
                    )[0]
                )
                time.sleep(0.08)
                self.assertEqual(e.runtime_stats["semantic_measurements"], [])
            with e.semantic_node("decoder", "shape-hash-b", "primitive-lowering"):
                half = self.keep(e.mul_plain(linear, 0.5))
                shifted = self.keep(e.add_plain(half, 0.125))
            return [self.keep(e.neg(shifted))]

        output = e.run_scope(capture)[0]
        records = {
            item["name"]: item for item in e.runtime_stats["semantic_measurements"]
        }
        self.assertEqual(set(records), {"encoder_linear", "decoder"})
        self.assertEqual(records["encoder_linear"]["signature"], "shape-hash-a")
        self.assertEqual(
            records["encoder_linear"]["current_implementation"], "diagonal-native"
        )
        self.assertEqual(records["encoder_linear"]["command_count"], 1)
        self.assertEqual(records["decoder"]["command_count"], 2)
        measured = 0.0
        for record in records.values():
            self.assertEqual(record["timing_scope"], "exclusive_synchronized_wall")
            self.assertGreater(record["seconds"], 0)
            measured += record["seconds"]
        self.assertGreaterEqual(e.last_scope_timing["capture_seconds"], 0.08)
        self.assertLess(measured, e.last_scope_timing["total_seconds"] - 0.05)
        self.assertEqual(e.runtime_stats["last_scope"]["executed_commands"], 4)
        for actual, expected in zip(
            self.client.decrypt(output), [-0.25, -0.03125, -0.296875, -0.25]
        ):
            self.assertAlmostEqual(actual, expected, delta=2e-5)

    def test_nested_origins_restore_and_failure_cleanup_does_not_tag_later_work(self):
        e, x = self.evaluator, self.source()

        def capture():
            with e.semantic_node("outer", "outer-hash", "primitive"):
                first = self.keep(e.neg(x))
                with e.semantic_node("inner", "inner-hash", "primitive"):
                    second = self.keep(e.neg(first))
                third = self.keep(e.neg(second))
            try:
                with e.semantic_node("failed-context", "failure-hash", "primitive"):
                    raise ValueError("frontend lowering failed before submission")
            except ValueError:
                pass
            return [self.keep(e.neg(third))]

        e.run_scope(capture)
        records = {
            item["name"]: item for item in e.runtime_stats["semantic_measurements"]
        }
        self.assertEqual(set(records), {"outer", "inner"})
        self.assertEqual(records["outer"]["command_count"], 2)
        self.assertEqual(records["inner"]["command_count"], 1)
        # A fresh, unattributed scope resets prior observations.
        self.keep(e.run_scope(lambda: [e.neg(x)])[0])
        self.assertEqual(e.runtime_stats["semantic_measurements"], [])

    def test_invalid_capture_publishes_no_measurements_and_next_scope_recovers(self):
        e, x = self.evaluator, self.source()

        def invalid():
            with e.semantic_node("invalid", "failure", "primitive"):
                self.keep(e.neg(x))
                return [e.add_plain(x, [1, 2, 3])]

        with self.assertRaises(RuntimeError):
            e.run_scope(invalid)
        self.assertEqual(e.runtime_stats["semantic_measurements"], [])

        def valid():
            with e.semantic_node("valid", "recovered", "primitive"):
                return [self.keep(e.neg(x))]

        e.run_scope(valid)
        records = e.runtime_stats["semantic_measurements"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["name"], "valid")
        self.assertEqual(records[0]["command_count"], 1)

    def test_successful_zero_command_nodes_appear_only_after_scope_completion(self):
        e, x = self.evaluator, self.source()

        def capture():
            with e.semantic_node("view", "shape-only", "layout-view"):
                pass
            self.assertEqual(e.runtime_stats["semantic_measurements"], [])
            return [x]

        e.run_scope(capture)
        self.assertEqual(
            e.runtime_stats["semantic_measurements"],
            [
                {
                    "name": "view",
                    "signature": "shape-only",
                    "current_implementation": "layout-view",
                    "seconds": 0.0,
                    "command_count": 0,
                    "timing_scope": "exclusive_synchronized_wall",
                }
            ],
        )

    def test_binary_identity_matches_loaded_files_and_is_stable(self):
        import hashlib

        identity = self.evaluator.implementation_identity
        self.assertEqual(identity, self.evaluator.implementation_identity)
        for key, library in (
            ("adapter_sha256", self.evaluator._session.library.lib),
            ("provider_sha256", self.evaluator._session.provider_library.lib),
        ):
            with open(library._name, "rb") as source:
                expected = hashlib.file_digest(source, "sha256").hexdigest()
            self.assertEqual(identity[key], expected)
        self.assertEqual(len(identity["identity_sha256"]), 64)
        self.assertNotIn("/", str(identity))

    def test_identity_failure_releases_native_runtime_exactly_once(self):
        import subprocess
        import sys

        # Inject one filesystem-identity failure; all provider allocation and
        # Runtime ownership transfer are real, in an isolated process.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from unittest.mock import patch
from fhenomenon.backends import runtime as backend
with patch.object(backend, "_loaded_binary_digest", side_effect=RuntimeError("identity unavailable")):
    try:
        backend.create_runtime_context(profile="NativeTest", mod_raise_levels=[])
    except RuntimeError as error:
        assert str(error) == "identity unavailable"
    else:
        raise AssertionError("identity failure was ignored")
client, evaluator = backend.create_runtime_context(profile="NativeTest", mod_raise_levels=[])
client.close(); evaluator.close()
""",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
