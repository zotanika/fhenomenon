"""Exercise real Scope measurement -> validated report -> engineering draft."""

import unittest

from fhenomenon.ml import derive_kernel_issues
from python.examples.kernel_issue_demo import run_demo


class KernelIssueDemoTests(unittest.TestCase):
    def test_real_encrypted_training_emits_valid_bound_request(self):
        report = run_demo(samples=2)
        self.assertTrue(report["passed"])
        self.assertEqual(report["kernel_cost_measurement"]["sample_count"], 2)
        self.assertEqual(report["validation"]["intermediate_decryptions"], 0)
        self.assertEqual(report["validation"]["private_host_fallbacks"], 0)
        self.assertTrue(report["scope_coverage"]["complete"])
        drafts = derive_kernel_issues(
            report["kernel_requests"],
            report["kernel_cost_measurement"],
            context=report["issue_context"],
        )
        self.assertEqual(len(drafts), 1)
        self.assertGreater(drafts[0]["measured_seconds"], 0)
        self.assertIn("insecure numerical fixture", drafts[0]["body"])
        self.assertIn("2e-05", drafts[0]["body"])


if __name__ == "__main__":
    unittest.main()
