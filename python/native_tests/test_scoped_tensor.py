"""Scope boundaries must include output layouts and invalidate capture caches."""

import gc
import unittest

import numpy as np
import torch

from fhenomenon.ml.tensor import TensorEngine
from fhenomenon.backends.runtime import create_runtime_context


class ScopedTensorTests(unittest.TestCase):
    def setUp(self):
        self.client, self.backend = create_runtime_context(
            profile="NativeTest", rotations=[1, 2], mod_raise_levels=[2]
        )
        self.engine = TensorEngine(
            self.backend, client=self.client, slots=4, target_level=2
        )

    def tearDown(self):
        self.client.close()
        self.backend.close()

    def test_nested_output_views_are_exported_before_execution(self):
        x = self.engine.encrypt(torch.tensor([[0.125, 0.25], [0.375, 0.5]]))
        before = self.backend.runtime_stats["recipe_calls"].copy()

        def capture():
            result = self.engine.add(x, 0.125)
            self.assertEqual(self.backend.runtime_stats["recipe_calls"], before)
            view = self.engine.permute(result, (1, 0))
            return {
                "loss": self.engine.sum(result),
                "gradient": view,
                "updates": (result, view),
                "public": torch.tensor(0.0),
            }

        output = self.engine.run_scope(capture)
        calls = self.backend.runtime_stats["recipe_calls"].copy()
        self.assertGreater(sum(calls.values()), sum(before.values()))
        self.assertAlmostEqual(
            float(self.engine.decrypt(output["loss"])), 1.75, delta=1e-5
        )
        np.testing.assert_allclose(
            self.engine.decrypt(output["gradient"]),
            [[0.25, 0.5], [0.375, 0.625]],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            self.engine.decrypt(output["updates"][0]),
            [[0.25, 0.375], [0.5, 0.625]],
            atol=1e-5,
        )
        self.assertIs(output["gradient"], output["updates"][1])
        self.assertEqual(self.backend.runtime_stats["recipe_calls"], calls)
        self.assertEqual(self.client.stats["decrypt_calls"], 3)
        scope = self.backend.runtime_stats["last_scope"]
        self.assertEqual(scope["commands"], scope["executed_commands"])
        self.assertGreater(scope["commands"], 1)
        output = x = None
        gc.collect()
        self.assertEqual(self.backend.runtime_stats["live_resources"], 0)

    def test_failed_capture_cannot_poison_reusable_input_views(self):
        x = self.engine.encrypt(torch.tensor([[0.125, 0.25], [0.375, 0.5]]))
        view = self.engine.permute(x, (1, 0))
        before = self.backend.runtime_stats["recipe_calls"].copy()

        def broken():
            self.engine.add(view, 0.125)
            raise ValueError("late frontend rejection")

        with self.assertRaisesRegex(ValueError, "late frontend rejection"):
            self.engine.run_scope(broken)
        self.assertEqual(self.backend.runtime_stats["recipe_calls"], before)
        result = self.engine.run_scope(lambda: self.engine.add(view, 0.125))
        np.testing.assert_allclose(
            self.engine.decrypt(result), [[0.25, 0.5], [0.375, 0.625]], atol=1e-5
        )
        result = view = x = None
        gc.collect()
        self.assertEqual(self.backend.runtime_stats["live_resources"], 0)

    def test_refresh_cache_is_local_to_one_capture(self):
        x = self.engine.encrypt(torch.tensor([0.125, 0.25, 0.375, 0.5]))
        low = self.engine._new(
            [self.backend.level_down(x.chunks[0], 0)], (4,), public_bound=1.0
        )
        for _ in range(2):
            result = self.engine.run_scope(lambda: self.engine.mul(low, 2.0))
            np.testing.assert_allclose(
                self.engine.decrypt(result), [0.25, 0.5, 0.75, 1.0], atol=2e-5
            )
            del result
        low = x = None
        gc.collect()
        self.assertEqual(self.backend.runtime_stats["live_resources"], 0)


if __name__ == "__main__":
    unittest.main()
