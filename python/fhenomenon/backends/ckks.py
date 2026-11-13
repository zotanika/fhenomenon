"""Native CKKS CPU/CUDA provider with separate client and keyless evaluator.

Values are periodic across physical CKKS slots, padded to the next power of two.
NativeTest is an explicitly insecure numerical fixture. Native128 enforces the
engine modulus-budget policy; it is not an independently certified security claim.
Conversion consumes levels and is distinct from semantic refresh.
"""

from __future__ import annotations
import ctypes as C
import json
import math
import threading
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
_LIBRARY_DIR = Path(os.environ.get("FHN_LIBRARY_DIR", ROOT / "build/lib"))
_LIBRARY_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
LIBRARY = _LIBRARY_DIR / ("libfhn_ckks_tensor" + _LIBRARY_SUFFIX)
CUDA_LIBRARY = LIBRARY
_LOADED_DEVICE = None
SUPPORTED_PROFILES = ("Native128", "NativeTest")


class NativeError(RuntimeError):
    pass


def _values(values):
    if isinstance(values, (float, int)):
        values = [values]
    data = [float(v) for v in values]
    if not data or not all(math.isfinite(value) for value in data):
        raise ValueError("plaintext must be nonempty and finite")
    return (C.c_double * len(data))(*data)


class _Library:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        version = self.lib.fn_get_abi_version
        version.argtypes, version.restype = [], C.c_uint32
        if version() != 1:
            raise NativeError("incompatible tensor provider ABI; expected version 1")
        self.lock = threading.RLock()
        ptr = C.c_void_p
        args = {
            "fn_create": [
                C.POINTER(C.c_int32),
                C.c_size_t,
                C.POINTER(ptr),
                C.POINTER(ptr),
            ],
            "fn_create_profile": [
                C.c_int,
                C.POINTER(C.c_int32),
                C.c_size_t,
                C.POINTER(ptr),
                C.POINTER(ptr),
            ],
            "fn_select_device": [ptr, C.c_int],
            "fn_synchronize": [ptr],
            "fn_prepare_mod_raise": [ptr, C.c_uint32],
            "fn_encrypt": [ptr, C.POINTER(C.c_double), C.c_size_t, C.POINTER(ptr)],
            "fn_decrypt": [ptr, ptr, C.POINTER(C.c_double), C.c_size_t],
            "fn_decrypt_bounded": [
                ptr,
                ptr,
                C.POINTER(C.c_double),
                C.c_size_t,
                C.c_double,
            ],
            "fn_binary": [ptr, C.c_int, ptr, ptr, C.POINTER(ptr)],
            "fn_plain": [
                ptr,
                C.c_int,
                ptr,
                C.POINTER(C.c_double),
                C.c_size_t,
                C.POINTER(ptr),
            ],
            "fn_neg": [ptr, ptr, C.POINTER(ptr)],
            "fn_rotate": [ptr, ptr, C.c_int32, C.POINTER(ptr)],
            "fn_level_down": [ptr, ptr, C.c_uint32, C.POINTER(ptr)],
            "fn_bootstrap": [ptr, ptr, C.POINTER(ptr)],
            "fn_mod_raise": [ptr, ptr, C.c_uint32, C.POINTER(ptr)],
            "fn_mod_raise_bounded": [ptr, ptr, C.c_uint32, C.c_double, C.POINTER(ptr)],
            "fn_slot_to_coeff": [ptr, ptr, C.POINTER(ptr)],
            "fn_coeff_to_slot": [ptr, ptr, C.POINTER(ptr)],
        }
        for name, types in args.items():
            fn = getattr(self.lib, name)
            fn.argtypes = types
            fn.restype = C.c_int
        for name in (
            "fn_profile",
            "fn_stats",
            "fn_client_stats",
            "fn_ciphertext_info",
            "fn_device_info",
        ):
            fn = getattr(self.lib, name)
            fn.argtypes = [ptr]
            fn.restype = C.c_char_p
        self.lib.fn_refresh_plan.argtypes = [ptr, C.c_uint32]
        self.lib.fn_refresh_plan.restype = C.c_char_p
        self.lib.fn_last_error.argtypes = []
        self.lib.fn_last_error.restype = C.c_char_p
        self.lib.fn_library_identity.argtypes = []
        self.lib.fn_library_identity.restype = ptr
        for name in ("fn_client_free", "fn_evaluator_free", "fn_ciphertext_free"):
            fn = getattr(self.lib, name)
            fn.argtypes = [ptr]
            fn.restype = None

    def check(self, status):
        if status:
            raise NativeError(self.lib.fn_last_error().decode())

    def call(self, name, *arguments):
        with self.lock:
            status = getattr(self.lib, name)(*arguments)
            if status is not None:
                self.check(status)

    def info(self, name, handle):
        with self.lock:
            result = getattr(self.lib, name)(handle)
            if result is None:
                raise NativeError(self.lib.fn_last_error().decode())
            return json.loads(result)


class Ciphertext:
    def __init__(self, library, handle):
        self._library, self._handle = library, handle

    def close(self):
        with self._library.lock:
            if getattr(self, "_handle", None):
                self._library.call("fn_ciphertext_free", self._handle)
                self._handle = None

    def __del__(self):
        self.close()

    @property
    def info(self):
        with self._library.lock:
            return self._library.info("fn_ciphertext_info", self._handle)

    @property
    def level(self):
        return self.info["level"]

    @property
    def representation(self):
        return self.info["representation"]

    @property
    def count(self):
        return self.info["count"]

    @property
    def estimated_bytes(self):
        return self.info["estimated_bytes"]


class Client:
    """Only this object owns a native handle containing secret-key material."""

    def __init__(self, library, handle):
        self._library, self._handle = library, handle

    def close(self):
        with self._library.lock:
            if getattr(self, "_handle", None):
                self._library.call("fn_client_free", self._handle)
                self._handle = None

    def __del__(self):
        self.close()

    def encrypt(self, values: Iterable[float] | float):
        data = _values(values)
        result = C.c_void_p()
        with self._library.lock:
            self._library.call(
                "fn_encrypt", self._handle, data, len(data), C.byref(result)
            )
        return Ciphertext(self._library, result)

    def decrypt(self, ciphertext: Ciphertext, public_bound=1.0):
        """Clone and lower to one prime, then decrypt immutably.

        No ciphertext range is inspected. The declared bound must cover the output's
        magnitude and fit the minimum-prime dynamic range; otherwise egress fails.
        """
        public_bound = float(public_bound)
        if not math.isfinite(public_bound) or public_bound < 1:
            raise ValueError("public bound must be finite and at least one")
        with self._library.lock:
            if (
                not isinstance(ciphertext, Ciphertext)
                or ciphertext._library is not self._library
            ):
                raise NativeError("ciphertext belongs to another key context")
            data = (C.c_double * ciphertext.count)()
            self._library.call(
                "fn_decrypt_bounded",
                self._handle,
                ciphertext._handle,
                data,
                len(data),
                public_bound,
            )
        return list(data)

    @property
    def stats(self):
        with self._library.lock:
            return self._library.info("fn_client_stats", self._handle)


class Evaluator:
    """Evaluation keys and ciphertexts only; no decrypt or encrypt capability."""

    def __init__(self, library, handle):
        self._library, self._handle = library, handle

    def close(self):
        with self._library.lock:
            if getattr(self, "_handle", None):
                self._library.call("fn_evaluator_free", self._handle)
                self._handle = None

    def __del__(self):
        self.close()

    def _operand(self, value):
        if not isinstance(value, Ciphertext) or value._library is not self._library:
            raise NativeError("expected a ciphertext from this key context")
        if not value._handle:
            raise NativeError("ciphertext is closed")
        return value._handle

    def _call(self, name, *args):
        with self._library.lock:
            operands = {
                "fn_binary": (1, 2),
                "fn_plain": (1,),
                "fn_neg": (0,),
                "fn_rotate": (0,),
                "fn_level_down": (0,),
                "fn_mod_raise": (0,),
                "fn_mod_raise_bounded": (0,),
                "fn_bootstrap": (0,),
                "fn_slot_to_coeff": (0,),
                "fn_coeff_to_slot": (0,),
            }
            if name not in operands:
                raise NativeError("unknown evaluator operation")
            converted = list(args)
            for index in operands[name]:
                if index >= len(converted):
                    raise NativeError("missing ciphertext operand")
                converted[index] = self._operand(args[index])
            result = C.c_void_p()
            self._library.call(name, self._handle, *converted, C.byref(result))
            return Ciphertext(self._library, result)

    def add(self, a, b):
        return self._call("fn_binary", 0, a, b)

    def sub(self, a, b):
        return self._call("fn_binary", 1, a, b)

    def mul(self, a, b):
        return self._call("fn_binary", 2, a, b)

    def add_plain(self, a, values):
        data = _values(values)
        return self._call("fn_plain", 0, a, data, len(data))

    def mul_plain(self, a, values):
        data = _values(values)
        return self._call("fn_plain", 2, a, data, len(data))

    def neg(self, a):
        return self._call("fn_neg", a)

    def rotate(self, a, step):
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not -(2**31) <= step < 2**31
        ):
            raise ValueError("rotation must be an int32")
        return self._call("fn_rotate", a, step)

    def level_down(self, a, level):
        with self._library.lock:
            self._operand(a)
            if (
                isinstance(level, bool)
                or not isinstance(level, int)
                or not 0 <= level <= a.level
            ):
                raise ValueError(
                    "level-down target must be between zero and the input level"
                )
            return self._call("fn_level_down", a, level)

    def mod_raise(self, a, target_level, public_bound=1.0):
        """Refresh directly to target_level with its shorter bootstrap RNS chain.

        Refresh is unconditional, including equal/lower targets, using a shorter
        active RNS prefix for smaller targets. Public bounds are declarations,
        never derived from encrypted values. Bounds outside the implemented
        refresh interval fail; this is not a relative-accuracy guarantee.
        """
        public_bound = float(public_bound)
        if not math.isfinite(public_bound) or public_bound < 1:
            raise ValueError("public bound must be finite and at least one")
        if (
            isinstance(target_level, bool)
            or not isinstance(target_level, int)
            or not 0 <= target_level <= self.profile["top_level"]
        ):
            raise ValueError(
                "requested level is outside the native CKKS evaluation chain"
            )
        if target_level not in self.profile["mod_raise_targets"]:
            raise ValueError("requested target was not prepared at client setup")
        return self._call("fn_mod_raise_bounded", a, target_level, public_bound)

    def synchronize(self):
        with self._library.lock:
            self._library.call("fn_synchronize", self._handle)

    @property
    def device_info(self):
        with self._library.lock:
            return self._library.info("fn_device_info", self._handle)

    def refresh_plan(self, target_level):
        if (
            isinstance(target_level, bool)
            or not isinstance(target_level, int)
            or not 0 <= target_level <= self.profile["top_level"]
        ):
            raise ValueError("invalid refresh target")
        with self._library.lock:
            result = self._library.lib.fn_refresh_plan(self._handle, target_level)
            if result is None:
                raise NativeError(self._library.lib.fn_last_error().decode())
            return json.loads(result)

    def slot_to_coeff(self, a):
        return self._call("fn_slot_to_coeff", a)

    def coeff_to_slot(self, a):
        return self._call("fn_coeff_to_slot", a)

    @property
    def profile(self):
        with self._library.lock:
            return self._library.info("fn_profile", self._handle)

    @property
    def stats(self):
        with self._library.lock:
            return self._library.info("fn_stats", self._handle)


def create_context(
    profile="Native128",
    rotations=(),
    library_path=None,
    mod_raise_levels=None,
    device="cpu",
):
    global _LOADED_DEVICE
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if _LOADED_DEVICE is not None and _LOADED_DEVICE != device:
        raise RuntimeError(
            "CPU and CUDA native CKKS builds use the same library identity; select one device per process"
        )
    if profile not in SUPPORTED_PROFILES:
        raise ValueError("unsupported native CKKS profile")
    if mod_raise_levels is None:
        mod_raise_levels = (2, 8)
    rotations, mod_raise_levels = tuple(rotations), tuple(mod_raise_levels)
    if any(
        isinstance(step, bool)
        or not isinstance(step, int)
        or not -(2**31) <= step < 2**31
        for step in rotations
    ):
        raise ValueError("rotation steps must be int32 values")
    if any(
        isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 8
        for level in mod_raise_levels
    ):
        raise ValueError("invalid ModRaise target level")
    library = _Library(library_path or (CUDA_LIBRARY if device == "cuda" else LIBRARY))
    _LOADED_DEVICE = device
    steps = (C.c_int32 * len(rotations))(*rotations)
    client, evaluator = C.c_void_p(), C.c_void_p()
    library.call(
        "fn_create_profile",
        SUPPORTED_PROFILES.index(profile),
        steps,
        len(steps),
        C.byref(client),
        C.byref(evaluator),
    )
    c, e = Client(library, client), Evaluator(library, evaluator)
    try:
        library.call("fn_select_device", client, device == "cuda")
        for target in sorted(set(mod_raise_levels)):
            library.call("fn_prepare_mod_raise", client, target)
        return c, e
    except BaseException:
        c.close()
        e.close()
        raise
