"""Real ciphertext tests for the private Fhenomenon Runtime adapter."""

import gc
import unittest


class RuntimeBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fhenomenon.backends.runtime import create_runtime_context

        cls.client, cls.evaluator = create_runtime_context(
            profile="NativeTest", rotations=[1, -1], mod_raise_levels=[2]
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "client"):
            cls.client.close()
            cls.evaluator.close()

    def _ready(self):
        self.assertIsNotNone(self.evaluator)

    def test_arithmetic_refresh_and_egress_use_tracked_immutable_values(self):
        self._ready()
        client, evaluator = self.client, self.evaluator
        self.assertEqual(evaluator.kind, "ckks")
        self.assertFalse(hasattr(evaluator, "decrypt"))
        self.assertFalse(hasattr(evaluator, "encrypt"))
        before = evaluator.runtime_stats
        x = client.encrypt([0.125, -0.25, 0.5, 0.375])
        y = client.encrypt([0.5, 0.25, -0.5, 0.125])
        initial = x.info.copy()
        summed = evaluator.add(x, y)
        product = evaluator.mul(summed, x)
        low = evaluator.level_down(product, 0)
        raised = evaluator.mod_raise(low, 2)
        values = [x, y, summed, product, low, raised]
        self.assertEqual(raised.level, 2)
        self.assertEqual(x.info, initial)
        self.assertEqual(
            evaluator.runtime_stats["live_resources"] - before["live_resources"], 6
        )
        self.assertEqual(
            evaluator.runtime_stats["live_bytes"] - before["live_bytes"],
            sum(value.info["runtime_bytes"] for value in values),
        )
        for got, want in zip(client.decrypt(raised), [0.078125, 0.0, 0.0, 0.1875]):
            self.assertAlmostEqual(got, want, delta=2e-5)
        self.assertEqual(raised.level, 2)
        self.assertEqual(client.stats["last_egress_modulus_count"], 1)
        for got, want in zip(client.decrypt(x), [0.125, -0.25, 0.5, 0.375]):
            self.assertAlmostEqual(got, want, delta=2e-5)
        calls = evaluator.runtime_stats["recipe_calls"]
        for operation in ("add", "mul", "level_down", "mod_raise"):
            self.assertGreater(calls["native.ckks." + operation], 0)
        self.assertEqual(evaluator.stats["evaluator_decrypt_calls"], 0)
        self.assertEqual(x.info["runtime_format"], "fhenomenon.native.ckks")
        for value in values:
            value.close()
        self.assertEqual(evaluator.runtime_stats["live_bytes"], before["live_bytes"])
        self.assertEqual(
            evaluator.runtime_stats["live_resources"], before["live_resources"]
        )

    def test_other_recipes_and_invalid_requests_have_no_fallback(self):
        self._ready()
        client, evaluator = self.client, self.evaluator
        x = client.encrypt([0.125, -0.25, 0.5, 0.375])
        a = evaluator.add_plain(x, [0.25])
        b = evaluator.mul_plain(a, [0.5, -0.5, 0.5, -0.5])
        negated = evaluator.neg(b)
        rotated = evaluator.rotate(negated, 1)
        result = evaluator.sub(rotated, negated)
        coeff = evaluator.slot_to_coeff(x)
        slots = evaluator.coeff_to_slot(coeff)
        low = evaluator.level_down(x, 0)
        before = evaluator.runtime_stats.copy()
        with self.assertRaises(RuntimeError):
            evaluator.mul(low, low)
        with self.assertRaises(ValueError):
            evaluator.rotate(x, 2**31)
        with self.assertRaises((ValueError, RuntimeError)):
            evaluator.add(coeff, x)
        with self.assertRaises(ValueError):
            evaluator.mod_raise(low, 3)
        with self.assertRaises(ValueError):
            evaluator.add(x, object())
        after = evaluator.runtime_stats
        self.assertEqual(after["live_bytes"], before["live_bytes"])
        self.assertEqual(after["live_resources"], before["live_resources"])
        self.assertGreater(after["failed_calls"], before["failed_calls"])
        for got, want in zip(client.decrypt(result), [0.1875, -0.375, 0.6875, -0.5]):
            self.assertAlmostEqual(got, want, delta=2e-5)
        for got, want in zip(client.decrypt(slots), [0.125, -0.25, 0.5, 0.375]):
            self.assertAlmostEqual(got, want, delta=2e-5)
        for value in [x, a, b, negated, rotated, result, coeff, slots, low]:
            value.close()
        self.assertEqual(evaluator.runtime_stats["live_resources"], 0)

    def test_public_bound_reaches_refresh_and_minimum_level_egress(self):
        import ctypes as C

        client, evaluator = self.client, self.evaluator
        x = client.encrypt([32.0, -24.0, 16.0, -8.0])
        low = evaluator.level_down(x, 0)
        source_info = low.info.copy()
        before = evaluator.runtime_stats
        decryptions = client.stats["decrypt_calls"]
        raised = None
        try:
            for bound in (0.5, float("nan"), float("inf"), -1.0):
                with self.subTest(bound=bound):
                    with self.assertRaisesRegex(ValueError, "public bound"):
                        evaluator.mod_raise(low, 2, public_bound=bound)
                    with self.assertRaisesRegex(ValueError, "public bound"):
                        client.decrypt(low, public_bound=bound)
            self.assertEqual(client.stats["decrypt_calls"], decryptions)
            self.assertEqual(
                evaluator.runtime_stats["recipe_calls"], before["recipe_calls"]
            )
            # The native entry point must validate too, without publishing a Value.
            session = evaluator._session
            out = C.c_void_p()
            status = session.library.lib.fr_mod_raise_bounded(
                session.handle, low._handle, 2, float("nan"), C.byref(out)
            )
            self.assertNotEqual(status, 0)
            self.assertFalse(out.value)
            self.assertEqual(
                evaluator.runtime_stats["live_bytes"], before["live_bytes"]
            )
            raised = evaluator.mod_raise(low, 2, public_bound=64.0)
            self.assertEqual(raised.level, 2)
            self.assertEqual(evaluator.stats["last_refresh_public_bound"], 64.0)
            self.assertEqual(
                evaluator.runtime_stats["last_mod_raise_public_bound"], 64.0
            )
            self.assertEqual(
                evaluator.runtime_stats["recipe_calls"]["native.ckks.mod_raise"],
                before["recipe_calls"].get("native.ckks.mod_raise", 0) + 1,
            )
            for got, want in zip(
                client.decrypt(raised, public_bound=64.0), [32.0, -24.0, 16.0, -8.0]
            ):
                self.assertAlmostEqual(got, want, delta=3e-4)
            self.assertEqual(client.stats["last_egress_public_bound"], 64.0)
            self.assertEqual(client.stats["last_egress_modulus_count"], 1)
            self.assertEqual(client.stats["decrypt_calls"], decryptions + 1)
            self.assertEqual(evaluator.stats["evaluator_decrypt_calls"], 0)
            self.assertEqual(low.info, source_info)
            self.assertEqual(raised.level, 2)
        finally:
            if raised is not None:
                raised.close()
            low.close()
            x.close()
        self.assertEqual(evaluator.runtime_stats["live_resources"], 0)

    def test_budget_failure_releases_client_import_and_closed_handles_reject(self):
        self._ready()
        from fhenomenon.backends.runtime import create_runtime_context

        client, evaluator = create_runtime_context(
            profile="NativeTest", mod_raise_levels=[], max_live_bytes=1
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "budget"):
                client.encrypt([0.125])
            self.assertEqual(evaluator.runtime_stats["live_resources"], 0)
            self.assertEqual(evaluator.runtime_stats["live_bytes"], 0)
            other_context_value = self.client.encrypt([0.125])
            try:
                with self.assertRaises(ValueError):
                    evaluator.neg(other_context_value)
            finally:
                other_context_value.close()
        finally:
            client.close()
            evaluator.close()
        with self.assertRaises(RuntimeError):
            client.encrypt([0.125])
        with self.assertRaises(RuntimeError):
            _ = evaluator.runtime_stats
        x = self.client.encrypt([0.125])
        x.close()
        with self.assertRaises(RuntimeError):
            self.evaluator.neg(x)
        with self.assertRaises(RuntimeError):
            self.client.decrypt(x)
        gc.collect()
        self.assertEqual(self.evaluator.runtime_stats["live_resources"], 0)


class NativeTestProfileRuntimeTests(unittest.TestCase):
    def test_explicit_test_profile_keeps_runtime_contract(self):
        from fhenomenon.backends.runtime import create_runtime_context

        client, evaluator = create_runtime_context(profile="NativeTest")
        values = []
        try:
            self.assertEqual(evaluator.profile["name"], "NativeTest")
            self.assertTrue(evaluator.profile["test_only"])
            self.assertEqual(evaluator.profile["top_level"], 8)
            self.assertEqual(evaluator.profile["mod_raise_targets"], [2, 8])
            self.assertEqual(evaluator.profile["execution_runtime"], "Fhenomenon")
            x = client.encrypt([16.0, -8.0, 4.0, -2.0])
            values.append(x)
            low = evaluator.level_down(x, 0)
            values.append(low)
            raised = evaluator.mod_raise(low, 2, public_bound=32.0)
            values.append(raised)
            for got, want in zip(
                client.decrypt(raised, public_bound=32.0), [16.0, -8.0, 4.0, -2.0]
            ):
                self.assertAlmostEqual(got, want, delta=2e-5)
            self.assertEqual(raised.level, 2)
            self.assertEqual(client.stats["last_egress_modulus_count"], 1)
            self.assertEqual(evaluator.stats["evaluator_decrypt_calls"], 0)
        finally:
            for value in values:
                value.close()
            self.assertEqual(evaluator.runtime_stats["live_resources"], 0)
            client.close()
            evaluator.close()


if __name__ == "__main__":
    unittest.main()
