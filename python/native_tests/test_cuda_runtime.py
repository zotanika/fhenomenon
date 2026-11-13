"""Standalone real CUDA tests; missing library or GPU is a failure, not a skip."""

import json
import unittest


class CudaRuntimeTests(unittest.TestCase):
    def test_gpu_recipe_chain_and_minimum_level_client_egress(self):
        from fhenomenon.backends.runtime import create_runtime_context

        client, evaluator = create_runtime_context(
            profile="NativeTest", device="cuda", rotations=[1], mod_raise_levels=[2]
        )
        values = []
        try:
            self.assertEqual(evaluator.device_info["device"], "cuda")
            self.assertFalse(evaluator.device_info["host_fallback_allowed"])
            a = client.encrypt([8.0, -16.0, 32.0, 24.0])
            values.append(a)
            b = evaluator.mul_plain(a, 0.5)
            values.append(b)
            c = evaluator.add(a, b)
            values.append(c)
            low = evaluator.level_down(c, 0)
            values.append(low)
            source_info = low.info.copy()
            refreshed = evaluator.mod_raise(low, 2, public_bound=64.0)
            values.append(refreshed)
            result = evaluator.rotate(refreshed, 1)
            values.append(result)
            evaluator.synchronize()
            self.assertTrue(all(value.info["device"] == "cuda" for value in values))
            self.assertEqual(result.level, 2)
            for got, expected in zip(
                client.decrypt(result, public_bound=64.0), [-24.0, 48.0, 36.0, 12.0]
            ):
                self.assertAlmostEqual(got, expected, delta=3e-4)
            self.assertEqual(client.stats["decrypt_calls"], 1)
            self.assertEqual(client.stats["last_egress_modulus_count"], 1)
            self.assertEqual(client.stats["last_egress_public_bound"], 64.0)
            self.assertEqual(evaluator.stats["last_refresh_public_bound"], 64.0)
            self.assertEqual(
                evaluator.runtime_stats["last_mod_raise_public_bound"], 64.0
            )
            self.assertEqual(low.info, source_info)
            self.assertEqual(result.level, 2)
            self.assertEqual(evaluator.stats["evaluator_decrypt_calls"], 0)
            self.assertEqual(
                evaluator.runtime_stats["recipe_calls"]["native.ckks.mod_raise"], 1
            )
            print(
                json.dumps(
                    {
                        "device": evaluator.device_info,
                        "runtime": evaluator.runtime_stats,
                        "client": client.stats,
                        "provider": evaluator.stats,
                    }
                ),
                flush=True,
            )
            with self.assertRaises(RuntimeError):
                create_runtime_context(
                    profile="NativeTest", device="cpu", mod_raise_levels=[]
                )
        finally:
            for value in values:
                value.close()
            self.assertEqual(evaluator.runtime_stats["live_resources"], 0)
            client.close()
            evaluator.close()


if __name__ == "__main__":
    unittest.main()
