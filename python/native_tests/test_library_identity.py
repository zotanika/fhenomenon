"""Checks provider instance identity without requiring a CUDA device."""

import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class ProviderAbiTests(unittest.TestCase):
    def test_incompatible_abi_rejected_before_handle_creation(self):
        from fhenomenon.backends import ckks, runtime

        for module, symbol in (
            (ckks, "fn_get_abi_version"),
            (runtime, "fr_get_abi_version"),
        ):
            version = lambda: 999
            stub = SimpleNamespace(**{symbol: version})
            with (
                self.subTest(symbol=symbol),
                patch.object(module.C, "CDLL", return_value=stub),
            ):
                with self.assertRaisesRegex(RuntimeError, "incompatible.*ABI"):
                    module._Library("unused")


class ProviderLibraryIdentityTests(unittest.TestCase):
    def run_isolated(self, body):
        setup = """
import ctypes as C
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch
from fhenomenon.backends import ckks as native
from fhenomenon.backends import runtime
# Load the Runtime with its regular provider, then a distinct copy of the same ABI.
library = runtime._Library(runtime.LIBRARY)
temporary = tempfile.TemporaryDirectory()
foreign = Path(temporary.name) / native.LIBRARY.name
shutil.copyfile(native.LIBRARY, foreign)
native.LIBRARY = foreign
cpu_library = native._Library(foreign)
assert library.lib.fr_provider_library_identity() != cpu_library.lib.fn_library_identity()
"""
        result = subprocess.run(
            [sys.executable, "-c", setup + textwrap.dedent(body)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mismatched_provider_rejects_before_key_generation(self):
        self.run_isolated("""
            with patch.object(native, "create_context", side_effect=AssertionError("key generation began before identity validation")):
                try:
                    runtime.create_runtime_context(profile="NativeTest", device="cpu",
                        runtime_library_path=runtime.LIBRARY, mod_raise_levels=[])
                except runtime.RuntimeError as error:
                    assert "provider library" in str(error), str(error)
                else:
                    raise AssertionError("mismatched provider was accepted")
        """)

    def test_direct_runtime_abi_rejects_foreign_evaluator_without_ownership_transfer(
        self,
    ):
        self.run_isolated("""
            client, evaluator = native.create_context(profile="NativeTest", mod_raise_levels=[])
            result = C.c_void_p()
            try:
                status = library.lib.fr_create(evaluator._handle, 0, C.byref(result))
                if status == 0:
                    evaluator._handle = None  # A successful call transfers ownership.
                    library.lib.fr_runtime_free(result)
                    raise AssertionError("a foreign provider evaluator was accepted")
                assert not result.value, "failure returned an owning Runtime handle"
                assert "library" in library.lib.fr_last_error().decode()
                # A failed transfer leaves the original handle usable and owned.
                assert evaluator.profile["name"] == "NativeTest"
            finally:
                client.close(); evaluator.close()
        """)


if __name__ == "__main__":
    unittest.main()
