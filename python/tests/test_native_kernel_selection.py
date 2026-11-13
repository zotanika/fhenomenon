import unittest

from fhenomenon.ml import NodeContract, TensorContract
from fhenomenon.ml.kernels import native_tensor_registry
from fhenomenon.ml.registry import select_kernel


class NativeSelectionTests(unittest.TestCase):
    def node(self, *, width=3, privacy="public"):
        return NodeContract(
            "linear",
            "aten.mm.default",
            "training",
            (
                TensorContract((2, 3), "torch.float32", "secret"),
                TensorContract((3, width), "torch.float32", privacy),
            ),
            TensorContract((2, width), "torch.float32", "secret"),
            "secret",
        )

    def select(self, node, *, slots=8, available=True):
        environment = {
            "runtime": {
                "slots": slots,
                "kernel_catalog": {
                    "kernels": [{"operation": "tensor.public_linear"}]
                    if available
                    else []
                },
            }
        }
        return select_kernel(node, environment, native_tensor_registry().snapshot())

    def test_selects_supported_public_linear_geometry(self):
        selected, _ = self.select(self.node())
        self.assertEqual(selected.name, "native.tensor.public_linear")

    def test_declines_private_weights_and_oversized_layout(self):
        for node in (self.node(privacy="secret"), self.node(width=5)):
            selected, reasons = self.select(node)
            self.assertIsNone(selected)
            self.assertTrue(reasons)

    def test_does_not_assume_provider_has_native_implementation(self):
        selected, reasons = self.select(self.node(), available=False)
        self.assertIsNone(selected)
        self.assertIn("catalog", reasons[0]["reason"])
