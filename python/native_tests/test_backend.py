"""Real native ABI checks. A missing native build is a failure, never a skip."""

import subprocess
import sys
import textwrap
import math
import unittest

from fhenomenon.backends.ckks import NativeError, create_context


class NativeBoundaryTests(unittest.TestCase):
    def test_public_plaintext_cache_is_reused_without_confusing_values_or_levels(self):
        client, evaluator = create_context(profile="NativeTest", mod_raise_levels=[])
        values = []
        source = [0.1, 0.2, 0.3, 0.4]
        mask = [1.0, 0.0, -1.0, 0.5]
        other_mask = [2.0, -1.0, 0.5, -0.25]
        try:
            x = client.encrypt(source)
            values.append(x)
            first = evaluator.mul_plain(x, mask)
            values.append(first)
            hits = evaluator.stats["plaintext_cache_hits"]
            repeated = evaluator.mul_plain(x, mask)
            values.append(repeated)
            self.assertGreater(evaluator.stats["plaintext_cache_hits"], hits)
            changed = evaluator.mul_plain(x, other_mask)
            values.append(changed)
            lower = evaluator.level_down(x, 7)
            values.append(lower)
            at_lower_scale = evaluator.mul_plain(lower, mask)
            values.append(at_lower_scale)
            self.assertNotEqual(x.info["scale"], lower.info["scale"])
            for actual, weights in (
                (first, mask),
                (repeated, mask),
                (changed, other_mask),
                (at_lower_scale, mask),
            ):
                for got, a, b in zip(client.decrypt(actual), source, weights):
                    self.assertAlmostEqual(got, a * b, delta=2e-5)
            self.assertGreater(evaluator.stats["plaintext_cache_bytes"], 0)
            self.assertLessEqual(
                evaluator.stats["plaintext_cache_bytes"], 256 * 1024 * 1024
            )
        finally:
            for value in values:
                value.close()
            client.close()
            evaluator.close()

    def test_non_ciphertext_operands_reject_without_crashing_the_process(self):
        # ctypes accepts integers for void pointers. Exercise the real boundary
        # in a subprocess so a regression cannot terminate the test runner.
        code = textwrap.dedent("""
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            from fhenomenon.backends.ckks import NativeError, create_context
            c, e = create_context(profile="NativeTest", mod_raise_levels=[2])
            x = c.encrypt([.1, .2])
            calls = [lambda: e.neg(7), lambda: e.add(7, x), lambda: e.sub(x, 7),
                     lambda: e.mul(x, 7), lambda: e.rotate(7, 1),
                     lambda: e.level_down(7, 0), lambda: e.add_plain(7, 1.),
                     lambda: e.mul_plain(7, 1.), lambda: e.mod_raise(7, 2),
                     lambda: e.slot_to_coeff(7), lambda: e.coeff_to_slot(7)]
            try:
                for call in calls:
                    try: call()
                    except (NativeError, ValueError): pass
                    else: raise AssertionError("non-ciphertext operand was accepted")
                x.close()
                try: e.neg(x)
                except NativeError: pass
                else: raise AssertionError("closed ciphertext was accepted")
            finally:
                x.close(); c.close(); e.close()
            print("all invalid operands rejected")
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], text=True, capture_output=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("all invalid operands rejected", result.stdout)

    def test_native_profile_arithmetic_identity_and_minimum_egress(self):
        client, evaluator = create_context(
            profile="NativeTest", rotations=[1], mod_raise_levels=[]
        )
        values = []
        try:
            self.assertTrue(evaluator.profile["test_only"])
            self.assertFalse(evaluator.profile["production_security_claim"])
            self.assertEqual(evaluator.profile["ring_degree"], 1024)
            self.assertEqual(evaluator.profile["top_level"], 8)
            self.assertEqual(evaluator.profile["slots"], 256)
            self.assertFalse(hasattr(evaluator, "decrypt"))
            x = client.encrypt([0.125, -0.25, 0.5, 0.375])
            values.append(x)
            original = x.info.copy()
            y = evaluator.mul_plain(x, [0.5])
            values.append(y)
            z = evaluator.add(x, y)
            values.append(z)
            rotated = evaluator.rotate(z, 1)
            values.append(rotated)
            for actual, expected in zip(
                client.decrypt(rotated), [-0.375, 0.75, 0.5625, 0.1875]
            ):
                self.assertAlmostEqual(actual, expected, delta=2e-5)
            self.assertEqual(x.info, original)
            self.assertEqual(client.stats["last_egress_modulus_count"], 1)
            self.assertEqual(client.stats["last_egress_level"], 0)
            self.assertEqual(evaluator.stats["evaluator_decrypt_calls"], 0)
            client.close()
            retained = evaluator.neg(x)
            values.append(retained)
            self.assertEqual(retained.level, 8)
            with self.assertRaises(NativeError):
                client.encrypt([0.1])
        finally:
            for value in values:
                value.close()
            client.close()
            evaluator.close()

    def test_refresh_and_real_representation_roundtrip(self):
        client, evaluator = create_context(profile="NativeTest", mod_raise_levels=[2])
        values = []
        try:
            x = client.encrypt([0.125, -0.25, 0.5, 0.375])
            values.append(x)
            coeff = evaluator.slot_to_coeff(x)
            values.append(coeff)
            self.assertEqual(coeff.representation, "coefficients")
            self.assertEqual(coeff.level, x.level - 1)
            with self.assertRaises(NativeError):
                client.decrypt(coeff)
            with self.assertRaises(NativeError):
                evaluator.add(coeff, x)
            slots = evaluator.coeff_to_slot(coeff)
            values.append(slots)
            self.assertEqual(slots.level, x.level - 2)
            low = evaluator.level_down(x, 0)
            values.append(low)
            with self.assertRaises(NativeError):
                evaluator.slot_to_coeff(low)
            before = client.stats["decrypt_calls"]
            raised = evaluator.mod_raise(low, 2)
            values.append(raised)
            self.assertEqual(client.stats["decrypt_calls"], before)
            self.assertEqual(raised.level, 2)
            expected_gain = 2 ** math.floor(
                math.log2(evaluator.profile["refresh_maximum_public_bound"])
            )
            self.assertEqual(evaluator.stats["last_refresh_public_gain"], expected_gain)
            for output in (slots, raised):
                for actual, expected in zip(
                    client.decrypt(output), [0.125, -0.25, 0.5, 0.375]
                ):
                    self.assertAlmostEqual(actual, expected, delta=2e-5)
            short, full = evaluator.refresh_plan(2), evaluator.refresh_plan(8)
            self.assertFalse(short["full_refresh_then_drop"])
            self.assertEqual(short["unit_bound_public_gain"], expected_gain)
            self.assertLess(
                short["scheduled_modulus_sum"], full["scheduled_modulus_sum"]
            )
        finally:
            for value in values:
                value.close()
            client.close()
            evaluator.close()

    def test_distinct_contexts_and_closed_handles_are_rejected(self):
        c1, e1 = create_context(profile="NativeTest", mod_raise_levels=[])
        c2, e2 = create_context(profile="NativeTest", mod_raise_levels=[])
        x, y = c1.encrypt([0.1]), c2.encrypt([0.2])
        try:
            with self.assertRaises(NativeError):
                e1.add(x, y)
            with self.assertRaises(NativeError):
                c2.decrypt(x)
            x.close()
            with self.assertRaises(NativeError):
                e1.neg(x)
        finally:
            x.close()
            y.close()
            c1.close()
            c2.close()
            e1.close()
            e2.close()


if __name__ == "__main__":
    unittest.main()
