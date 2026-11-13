"""Synthetic arithmetic fixtures for conditional pricing, never benchmarks."""

import copy
import math
import unittest

from fhenomenon.ml.costs import price_kernel_requests


def artifact():
    return {
        "schema_version": 1,
        "environment": {
            "runtime": {
                "backend_kind": "fixture-provider",
                "device": {"name": "fixture-cpu"},
                "profile": {"degree": 1024, "levels": 3},
                "build": "fixture-v1",
            },
        },
        "requests": [
            {
                "signature": "matmul-shape-a",
                "nodes": ["mm_0", "mm_1"],
                "occurrences": 2,
                "current_implementation": "decomposition",
            },
            {
                "signature": "normalization-shape-b",
                "nodes": ["norm_0"],
                "occurrences": 1,
                "current_implementation": "norm-v1",
            },
        ],
    }


def measurement(request):
    return {
        "schema_version": 1,
        "environment": copy.deepcopy(request["environment"]),
        "request_fingerprint": price_kernel_requests(request, None, [])[
            "request_fingerprint"
        ],
        "timing_boundary": "synchronized_scope_execution",
        "attribution_kind": "exclusive_synchronized_wall",
        "sample_count": 7,
        "whole_scope_wall_seconds": 10.0,
        "node_measurements": [
            {
                "node": "mm_0",
                "signature": "matmul-shape-a",
                "current_implementation": "decomposition",
                "seconds": 2.0,
            },
            {
                "node": "mm_1",
                "signature": "matmul-shape-a",
                "current_implementation": "decomposition",
                "seconds": 1.0,
            },
            {
                "node": "norm_0",
                "signature": "normalization-shape-b",
                "current_implementation": "norm-v1",
                "seconds": 1.0,
            },
        ],
    }


def offer(signature="matmul-shape-a"):
    return {
        "signature": signature,
        "assumed_seconds_per_invocation": 0.4,
        "setup_seconds": 0.1,
        "transfer_seconds": 0.2,
    }


class KernelPricingTests(unittest.TestCase):
    def test_whole_scope_estimate_retains_uncovered_work_and_charges_overheads_once(
        self,
    ):
        request = artifact()
        evidence = measurement(request)
        proposed = offer()
        result = price_kernel_requests(request, evidence, [proposed])
        self.assertEqual(result["status"], "conditional_estimate")
        self.assertEqual(result["measured_baseline_seconds"], 10.0)
        self.assertEqual(result["sample_count"], 7)
        priced = result["offers"][0]
        self.assertEqual(priced["status"], "conditional_estimate")
        self.assertEqual(priced["measured_covered_seconds"], 3.0)
        self.assertEqual(priced["assumed_kernel_seconds"], 0.8)
        self.assertAlmostEqual(priced["conditional_whole_scope_seconds"], 8.1)
        self.assertAlmostEqual(priced["conditional_speedup"], 10 / 8.1)
        self.assertEqual(result["combined"]["priced_signatures"], ["matmul-shape-a"])
        self.assertAlmostEqual(
            result["combined"]["conditional_whole_scope_seconds"], 8.1
        )
        self.assertIn("not a measured speedup", result["qualification"])

    def test_two_disjoint_offers_combine_measured_savings_without_counting_scope_twice(
        self,
    ):
        request = artifact()
        proposals = [offer(), offer("normalization-shape-b")]
        result = price_kernel_requests(request, measurement(request), proposals)
        self.assertAlmostEqual(result["combined"]["measured_covered_seconds"], 4.0)
        self.assertAlmostEqual(
            result["combined"]["conditional_whole_scope_seconds"], 7.8
        )
        self.assertAlmostEqual(result["combined"]["conditional_speedup"], 10 / 7.8)

    def test_no_measurement_stays_unpriced_even_when_offer_has_a_latency(self):
        request = artifact()
        request["environment"] = {}
        result = price_kernel_requests(request, None, [offer()])
        self.assertEqual(result["status"], "unpriced")
        self.assertIsNone(result["measured_baseline_seconds"])
        self.assertIsNone(result["combined"])
        self.assertEqual(result["offers"][0]["reason"], "missing_measurement")
        self.assertNotIn("conditional_whole_scope_seconds", result["offers"][0])

    def test_incomplete_node_measurements_never_extrapolate_remaining_occurrences(self):
        request = artifact()
        evidence = measurement(request)
        del evidence["node_measurements"][1]
        result = price_kernel_requests(
            request, evidence, [offer(), offer("normalization-shape-b")]
        )
        self.assertEqual(result["status"], "partially_priced")
        self.assertEqual(result["offers"][0]["status"], "unpriced")
        self.assertEqual(result["offers"][0]["missing_nodes"], ["mm_1"])
        self.assertEqual(
            result["combined"]["priced_signatures"], ["normalization-shape-b"]
        )
        self.assertAlmostEqual(
            result["combined"]["conditional_whole_scope_seconds"], 9.7
        )

    def test_mixed_current_implementations_bind_each_node(self):
        request = artifact()
        request["requests"][0]["current_implementation"] = "mixed"
        request["requests"][0]["current_implementations"] = {
            "mm_0": "kernel-v2",
            "mm_1": "decomposition",
        }
        evidence = measurement(request)
        evidence["node_measurements"][0]["current_implementation"] = "kernel-v2"
        self.assertEqual(
            price_kernel_requests(request, evidence, [offer()])["status"],
            "conditional_estimate",
        )
        evidence["node_measurements"][0]["current_implementation"] = "kernel-v1"
        with self.assertRaisesRegex(ValueError, "implementation"):
            price_kernel_requests(request, evidence, [offer()])

    def test_profile_device_backend_and_timing_provenance_must_match(self):
        for key, value in [
            ("profile", {"degree": 2048, "levels": 3}),
            ("device", {"name": "another-cpu"}),
            ("backend_kind", "other-provider"),
        ]:
            with self.subTest(environment_key=key):
                request = artifact()
                evidence = measurement(request)
                evidence["environment"]["runtime"][key] = value
                with self.assertRaisesRegex(ValueError, "environment"):
                    price_kernel_requests(request, evidence, [offer()])
        for key, value in [
            ("timing_boundary", "scope_capture"),
            ("attribution_kind", "overlapping_gpu_events"),
        ]:
            with self.subTest(timing_key=key):
                request = artifact()
                evidence = measurement(request)
                evidence[key] = value
                with self.assertRaisesRegex(ValueError, "timing|attribution"):
                    price_kernel_requests(request, evidence, [offer()])
        request = artifact()
        del request["environment"]["runtime"]["profile"]
        with self.assertRaisesRegex(ValueError, "environment"):
            price_kernel_requests(request, measurement(request), [offer()])

    def test_measurement_identity_and_coverage_are_checked_before_pricing(self):
        for key, value in [
            ("node", "not_in_graph"),
            ("signature", "wrong-shape"),
            ("current_implementation", "other-kernel"),
        ]:
            with self.subTest(key=key):
                request = artifact()
                evidence = measurement(request)
                evidence["node_measurements"][0][key] = value
                with self.assertRaises(ValueError):
                    price_kernel_requests(request, evidence, [offer()])

    def test_baseline_binds_the_entire_request_graph_even_with_partial_timings(self):
        request = artifact()
        evidence = measurement(request)
        evidence["node_measurements"] = evidence["node_measurements"][:2]
        request["requests"][1]["nodes"] = ["different_normalization"]
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            price_kernel_requests(request, evidence, [offer()])

    def test_graph_identity_change_rejects_an_otherwise_identical_baseline(self):
        request = artifact()
        request["graph_identity"] = "first-topology"
        evidence = measurement(request)
        request["graph_identity"] = "other-topology"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            price_kernel_requests(request, evidence, [offer()])

    def test_reference_measurement_is_never_presented_as_encrypted_speedup(self):
        request = artifact()
        request["environment"]["runtime"]["backend_kind"] = "noncryptographic_reference"
        result = price_kernel_requests(request, measurement(request), [offer()])
        self.assertEqual(
            result["measurement_backend_kind"], "noncryptographic_reference"
        )
        self.assertIn("do not predict encrypted execution", result["qualification"])

    def test_malformed_coverage_and_runtime_metadata_raise_value_error(self):
        request = artifact()
        request["requests"][0]["nodes"] = [{}, "mm_1"]
        request["requests"][0]["current_implementations"] = {
            "mm_0": "decomposition",
            "mm_1": "decomposition",
        }
        with self.assertRaises(ValueError):
            price_kernel_requests(request, None, [offer()])
        request = artifact()
        request["environment"]["runtime"] = []
        with self.assertRaises(ValueError):
            price_kernel_requests(request, None, [offer()])

    def test_duplicate_node_coverage_and_duplicate_offers_are_rejected(self):
        request = artifact()
        evidence = measurement(request)
        evidence["node_measurements"].append(
            copy.deepcopy(evidence["node_measurements"][0])
        )
        with self.assertRaisesRegex(ValueError, "duplicate|overlap"):
            price_kernel_requests(request, evidence, [offer()])
        with self.assertRaisesRegex(ValueError, "duplicate|overlap"):
            price_kernel_requests(request, measurement(request), [offer(), offer()])
        request["requests"][1]["nodes"] = ["mm_0"]
        with self.assertRaisesRegex(ValueError, "duplicate|overlap"):
            price_kernel_requests(request, None, [offer()])

    def test_inconsistent_occurrences_and_unknown_signature_are_rejected(self):
        request = artifact()
        request["requests"][0]["occurrences"] = 9
        with self.assertRaisesRegex(ValueError, "occurrences|coverage"):
            price_kernel_requests(request, None, [offer()])
        with self.assertRaisesRegex(ValueError, "signature"):
            price_kernel_requests(artifact(), None, [offer("unknown")])

    def test_nonfinite_negative_boolean_and_missing_costs_are_rejected(self):
        for field in (
            "assumed_seconds_per_invocation",
            "setup_seconds",
            "transfer_seconds",
        ):
            for invalid in (-1.0, math.inf, -math.inf, math.nan, True, "1"):
                with self.subTest(field=field, invalid=invalid):
                    proposed = offer()
                    proposed[field] = invalid
                    with self.assertRaises(ValueError):
                        price_kernel_requests(artifact(), None, [proposed])
            proposed = offer()
            del proposed[field]
            with self.assertRaises(ValueError):
                price_kernel_requests(artifact(), None, [proposed])
        for invalid in (-1.0, math.inf, math.nan, True):
            request = artifact()
            evidence = measurement(request)
            evidence["node_measurements"][0]["seconds"] = invalid
            with self.assertRaises(ValueError):
                price_kernel_requests(request, evidence, [offer()])
        for invalid in (0, -1, 1.5, True):
            request = artifact()
            evidence = measurement(request)
            evidence["sample_count"] = invalid
            with self.assertRaises(ValueError):
                price_kernel_requests(request, evidence, [offer()])

    def test_exclusive_attributions_cannot_exceed_measured_wall_time(self):
        request = artifact()
        evidence = measurement(request)
        evidence["whole_scope_wall_seconds"] = 3.0
        with self.assertRaisesRegex(ValueError, "wall|attribution"):
            price_kernel_requests(request, evidence, [offer()])

    def test_zero_latency_assumption_does_not_emit_infinite_speedup(self):
        request = artifact()
        request["requests"] = request["requests"][:1]
        evidence = measurement(request)
        evidence["node_measurements"] = evidence["node_measurements"][:2]
        evidence["whole_scope_wall_seconds"] = 3.0
        proposed = offer()
        for key in (
            "assumed_seconds_per_invocation",
            "setup_seconds",
            "transfer_seconds",
        ):
            proposed[key] = 0.0
        result = price_kernel_requests(request, evidence, [proposed])
        self.assertEqual(result["combined"]["conditional_whole_scope_seconds"], 0.0)
        self.assertIsNone(result["combined"]["conditional_speedup"])

    def test_inputs_are_not_modified_and_slow_offers_are_reported_honestly(self):
        request = artifact()
        evidence = measurement(request)
        proposed = offer()
        proposed["assumed_seconds_per_invocation"] = 5.0
        before = copy.deepcopy((request, evidence, proposed))
        result = price_kernel_requests(request, evidence, [proposed])
        self.assertEqual((request, evidence, proposed), before)
        self.assertLess(result["combined"]["conditional_speedup"], 1.0)
        result["environment"]["runtime"]["device"] = {"name": "edited-output"}
        self.assertEqual((request, evidence, proposed), before)


if __name__ == "__main__":
    unittest.main()
