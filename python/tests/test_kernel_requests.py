"""Engineering requests derived from bound scope evidence, not estimated timings."""

import copy
import unittest

from fhenomenon.ml.requests import derive_kernel_issues
from test_costs import artifact, measurement


def request_artifact():
    result = artifact()
    result["contains_model_weights_or_training_values"] = False
    for index, request in enumerate(result["requests"]):
        request.update(
            operation="aten.mm.default" if index == 0 else "aten.rsqrt.default",
            input_contracts=[{"shape": [2, 4], "visibility": "secret"}],
            keyword_contracts={},
            output_contract={"shape": [2, 4], "visibility": "secret"},
            selected_requirements=None,
            declined_candidates=[],
            acceptance={"intermediate_decryptions": 0},
        )
    return result


class EngineeringIssueTests(unittest.TestCase):
    def test_ranks_complete_requests_and_reports_bound_not_speedup(self):
        requests = request_artifact()
        before = copy.deepcopy(requests)
        drafts = derive_kernel_issues(requests, measurement(requests), limit=2)
        self.assertEqual(len(drafts), 2)
        first = drafts[0]
        self.assertEqual(first["operation"], "aten.mm.default")
        self.assertEqual(first["measured_seconds"], 3)
        self.assertAlmostEqual(first["scope_fraction"], 0.3)
        self.assertIn("7.000000", first["body"])
        self.assertIn("not a measured speedup", first["body"])
        self.assertIn("synchronized_scope_execution", first["body"])
        self.assertIn("intermediate_decryptions", first["body"])
        self.assertEqual(requests, before)

    def test_partial_coverage_is_excluded_without_extrapolation(self):
        requests = request_artifact()
        evidence = measurement(requests)
        evidence["node_measurements"].pop(0)
        drafts = derive_kernel_issues(requests, evidence, limit=10)
        self.assertEqual([d["operation"] for d in drafts], ["aten.rsqrt.default"])

    def test_bad_or_absent_measurement_and_unsafe_artifact_are_rejected(self):
        requests = request_artifact()
        evidence = measurement(requests)
        evidence["request_fingerprint"] = "different-graph"
        for invalid in (evidence, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                derive_kernel_issues(requests, invalid)
        requests["contains_model_weights_or_training_values"] = True
        with self.assertRaises(ValueError):
            derive_kernel_issues(requests, measurement(requests))

    def test_identity_survives_new_measurement_binary_and_graph_node_names(self):
        requests = request_artifact()
        before = derive_kernel_issues(requests, measurement(requests))[0]
        requests["graph_identity"] = "another-model"
        requests["environment"]["runtime"]["implementation_identity"] = {
            "sha256": "new"
        }
        requests["environment"]["runtime"]["kernel_catalog"] = {"kernels": ["new"]}
        requests["requests"][0]["current_implementation"] = "new-kernel"
        evidence = measurement(requests)
        for row in evidence["node_measurements"][:2]:
            row["current_implementation"] = "new-kernel"
            row["seconds"] *= 1.1
        after = derive_kernel_issues(requests, evidence)[0]
        self.assertEqual(before["request_id"], after["request_id"])
        self.assertNotEqual(before["body"], after["body"])

    def test_identity_includes_contract_runtime_and_application_accuracy(self):
        requests = request_artifact()
        original = derive_kernel_issues(requests, measurement(requests))[0][
            "request_id"
        ]
        mutations = (
            lambda r: r["requests"][0]["output_contract"].update(shape=[3, 4]),
            lambda r: r["environment"]["runtime"]["profile"].update(levels=5),
            lambda r: r["requests"][0]["acceptance"].update(intermediate_decryptions=1),
        )
        for mutate in mutations:
            changed = copy.deepcopy(requests)
            mutate(changed)
            self.assertNotEqual(
                original,
                derive_kernel_issues(changed, measurement(changed))[0]["request_id"],
            )
        with_accuracy = derive_kernel_issues(
            requests,
            measurement(requests),
            context={"numerical_acceptance": {"atol": 1e-5}},
        )[0]
        self.assertNotEqual(original, with_accuracy["request_id"])

    def test_only_requested_contract_and_allowlisted_context_are_disclosed(self):
        requests = request_artifact()
        requests["weights"] = "DO_NOT_PUBLISH"
        requests["requests"][0]["debug_values"] = "DO_NOT_PUBLISH"
        draft = derive_kernel_issues(
            requests,
            measurement(requests),
            context={"workload": "public fixture", "training_values": "DO_NOT_PUBLISH"},
        )[0]
        self.assertNotIn("DO_NOT_PUBLISH", draft["body"])
        self.assertIn("public fixture", draft["body"])

    def test_missing_contract_is_not_silently_published(self):
        requests = request_artifact()
        del requests["requests"][0]["output_contract"]
        with self.assertRaisesRegex(ValueError, "output_contract"):
            derive_kernel_issues(requests, measurement(requests))

    def test_zero_cost_candidates_are_not_engineering_priorities(self):
        requests = request_artifact()
        evidence = measurement(requests)
        for row in evidence["node_measurements"]:
            row["seconds"] = 0
        self.assertEqual(derive_kernel_issues(requests, evidence), [])

    def test_limit_must_be_positive_integer(self):
        requests = request_artifact()
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                derive_kernel_issues(requests, measurement(requests), limit=limit)


if __name__ == "__main__":
    unittest.main()
