"""Fhenomenon Runtime recipes over the native CKKS ciphertext ABI.

The evaluator contains no client handle. Python ciphertexts retain their owning
Runtime session; only the explicit client facade may borrow one for terminal
minimum-level decryption. All calls on the shared provider are serialized.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import ctypes as C
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import threading
import time

from fhenomenon.backends import ckks as native

ROOT = Path(__file__).resolve().parents[3]
_LIBRARY_DIR = Path(os.environ.get("FHN_LIBRARY_DIR", ROOT / "build/lib"))
_LIBRARY_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
LIBRARY = _LIBRARY_DIR / ("libfhn_tensor_runtime" + _LIBRARY_SUFFIX)
CUDA_LIBRARY = LIBRARY
_LOADED_DEVICE = None
_ScopeCallback = C.CFUNCTYPE(C.c_int, C.c_void_p)
_BINARY_DIGESTS = {}


class _DlInfo(C.Structure):
    _fields_ = [
        ("filename", C.c_char_p),
        ("base", C.c_void_p),
        ("symbol", C.c_char_p),
        ("address", C.c_void_p),
    ]


def _loaded_binary_digest(function):
    """Hash the file actually backing a loaded symbol, once per file identity."""
    address = C.cast(function, C.c_void_p).value
    process = C.CDLL(None)
    lookup = process.dladdr
    lookup.argtypes, lookup.restype = [C.c_void_p, C.POINTER(_DlInfo)], C.c_int
    info = _DlInfo()
    if not lookup(address, C.byref(info)) or not info.filename:
        raise RuntimeError("cannot resolve the loaded implementation binary")
    path = Path(os.fsdecode(info.filename)).resolve(strict=True)
    mapped = None
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        lower, upper = (int(value, 16) for value in fields[0].split("-"))
        if lower <= address < upper:
            major, minor = (int(value, 16) for value in fields[3].split(":"))
            mapped = (major, minor, int(fields[4]))
            break
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if mapped != (os.major(before.st_dev), os.minor(before.st_dev), before.st_ino):
            raise RuntimeError(
                "loaded implementation was replaced; restart before collecting binary identity"
            )
        key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        digest = _BINARY_DIGESTS.get(key)
        if digest is None:
            value = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                value.update(chunk)
            after = os.fstat(source.fileno())
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != key:
                raise RuntimeError(
                    "implementation binary changed while computing its identity"
                )
            digest = value.hexdigest()
            _BINARY_DIGESTS[key] = digest
    return digest


class RuntimeError(native.NativeError):
    """Native Runtime/provider rejection; execution never falls back to plaintext."""


def _values(values):
    if isinstance(values, (float, int)):
        values = [values]
    data = [float(value) for value in values]
    if not data or not all(math.isfinite(value) for value in data):
        raise ValueError("plaintext values must be nonempty and finite")
    return (C.c_double * len(data))(*data)


def _public_bound(value):
    value = float(value)
    if not math.isfinite(value) or value < 1:
        raise ValueError("public bound must be finite and at least one")
    return value


class _Library:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        version = self.lib.fr_get_abi_version
        version.argtypes, version.restype = [], C.c_uint32
        if version() != 1:
            raise RuntimeError("incompatible tensor runtime ABI; expected version 1")
        pointer = C.c_void_p
        declarations = {
            "fr_create": [pointer, C.c_size_t, C.POINTER(pointer)],
            "fr_import": [pointer, pointer, C.POINTER(pointer)],
            "fr_binary": [pointer, C.c_int, pointer, pointer, C.POINTER(pointer)],
            "fr_plain": [
                pointer,
                C.c_int,
                pointer,
                C.POINTER(C.c_double),
                C.c_size_t,
                C.POINTER(pointer),
            ],
            "fr_unary": [pointer, C.c_int, pointer, C.c_int64, C.POINTER(pointer)],
            "fr_mod_raise_bounded": [
                pointer,
                pointer,
                C.c_uint32,
                C.c_double,
                C.POINTER(pointer),
            ],
            "fr_synchronize": [pointer],
            "fr_run_scope": [
                pointer,
                _ScopeCallback,
                pointer,
                C.c_size_t,
                C.c_size_t,
                C.c_size_t,
                C.c_size_t,
                C.c_size_t,
            ],
            "fr_scope_export": [pointer, pointer],
            "fr_tensor_kernel": [
                pointer,
                C.c_char_p,
                C.POINTER(pointer),
                C.c_size_t,
                C.c_char_p,
                C.POINTER(pointer),
                C.c_size_t,
                C.POINTER(C.c_size_t),
            ],
            "fr_set_semantic_origin": [pointer, C.c_char_p, C.c_char_p, C.c_char_p],
            "fr_complete_semantic_origin": [pointer],
        }
        for name, arguments in declarations.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, C.c_int
        for name in (
            "fr_profile",
            "fr_provider_stats",
            "fr_runtime_stats",
            "fr_value_info",
            "fr_device_info",
            "fr_kernel_catalog",
        ):
            function = getattr(self.lib, name)
            function.argtypes, function.restype = [pointer], C.c_char_p
        self.lib.fr_refresh_plan.argtypes = [pointer, C.c_uint32]
        self.lib.fr_refresh_plan.restype = C.c_char_p
        self.lib.fr_last_error.argtypes = []
        self.lib.fr_last_error.restype = C.c_char_p
        self.lib.fr_provider_library_identity.argtypes = []
        self.lib.fr_provider_library_identity.restype = pointer
        self.lib.fr_value_borrow.argtypes = [pointer]
        self.lib.fr_value_borrow.restype = pointer
        for name in ("fr_runtime_free", "fr_value_free"):
            function = getattr(self.lib, name)
            function.argtypes, function.restype = [pointer], None

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.fr_last_error().decode(errors="replace"))

    def info(self, name, *arguments):
        result = getattr(self.lib, name)(*arguments)
        if result is None:
            self.check(-1)
        return json.loads(result)


class _Session:
    def __init__(self, library, handle, provider_library):
        # Ownership transfers only once all fallible identity checks succeed.
        # create_runtime_context remains responsible for handle on init failure.
        self.library, self.handle = library, None
        # Retain both loaded libraries; this holds no client/key handle.
        self.provider_library = provider_library
        self.lock = threading.RLock()
        self.capturing = False
        self.scope_active = False
        self.semantic_origin = None
        adapter = _loaded_binary_digest(library.lib.fr_create)
        provider = _loaded_binary_digest(provider_library.lib.fn_create_profile)
        self.implementation_identity = {
            "adapter_sha256": adapter,
            "provider_sha256": provider,
            "identity_sha256": hashlib.sha256(
                (adapter + ":" + provider).encode()
            ).hexdigest(),
        }
        self.handle = handle

    def __del__(self):
        handle = getattr(self, "handle", None)
        if handle:
            self.library.lib.fr_runtime_free(handle)
            self.handle = None


class Ciphertext:
    def __init__(self, session, handle):
        self._session, self._handle = session, handle

    def _check(self):
        if not self._handle:
            raise RuntimeError("ciphertext is closed")

    def close(self):
        with self._session.lock:
            if self._handle:
                self._session.library.lib.fr_value_free(self._handle)
                self._handle = None

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()

    @property
    def info(self):
        with self._session.lock:
            self._check()
            return self._session.library.info("fr_value_info", self._handle)

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
    """Explicit client boundary; secret material remains in the native CKKS client."""

    def __init__(self, native_client, session):
        self._native_client, self._session = native_client, session
        self._closed = False

    def _check(self):
        if self._closed:
            raise RuntimeError("client is closed")

    def encrypt(self, values):
        with self._session.lock:
            self._check()
            if self._session.capturing:
                raise RuntimeError(
                    "client encryption is forbidden during scope capture"
                )
            data = _values(values)
            source = self._native_client.encrypt(data)
            try:
                result = C.c_void_p()
                self._session.library.check(
                    self._session.library.lib.fr_import(
                        self._session.handle, source._handle, C.byref(result)
                    )
                )
                return Ciphertext(self._session, result)
            finally:
                source.close()

    def decrypt(self, ciphertext, public_bound=1.0):
        with self._session.lock:
            self._check()
            if self._session.capturing:
                raise RuntimeError(
                    "client decryption is forbidden during scope capture"
                )
            public_bound = _public_bound(public_bound)
            _operand(self._session, ciphertext)
            data = (C.c_double * ciphertext.count)()
            borrowed = self._session.library.lib.fr_value_borrow(ciphertext._handle)
            if not borrowed:
                self._session.library.check(-1)
            native = self._native_client
            # ciphertext and session are strongly held for this entire call.
            native._library.check(
                native._library.lib.fn_decrypt_bounded(
                    native._handle, borrowed, data, len(data), public_bound
                )
            )
            return list(data)

    @property
    def stats(self):
        with self._session.lock:
            self._check()
            return self._native_client.stats

    def close(self):
        with self._session.lock:
            if not self._closed:
                self._native_client.close()
                self._closed = True

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            self.close()


def _operand(session, value):
    if not isinstance(value, Ciphertext) or value._session is not session:
        raise ValueError("ciphertext belongs to another Runtime context")
    value._check()
    return value._handle


class Evaluator:
    kind = "ckks"

    def __init__(self, session):
        self._session, self._closed = session, False
        self.last_scope_timing = None

    def _check(self):
        if self._closed:
            raise RuntimeError("evaluator is closed")

    def _call(self, name, *arguments):
        with self._session.lock:
            self._check()
            operand_indices = (
                (1, 2)
                if name == "fr_binary"
                else (0,)
                if name == "fr_mod_raise_bounded"
                else (1,)
            )
            converted = list(arguments)
            for index in operand_indices:
                converted[index] = _operand(self._session, arguments[index])
            result = C.c_void_p()
            function = getattr(self._session.library.lib, name)
            self._session.library.check(
                function(self._session.handle, *converted, C.byref(result))
            )
            return Ciphertext(self._session, result)

    def add(self, a, b):
        return self._call("fr_binary", 0, a, b)

    def sub(self, a, b):
        return self._call("fr_binary", 1, a, b)

    def mul(self, a, b):
        return self._call("fr_binary", 2, a, b)

    def add_plain(self, a, values):
        data = _values(values)
        return self._call("fr_plain", 0, a, data, len(data))

    def mul_plain(self, a, values):
        data = _values(values)
        return self._call("fr_plain", 2, a, data, len(data))

    def neg(self, a):
        return self._call("fr_unary", 0, a, 0)

    def rotate(self, a, step):
        if not isinstance(step, int) or not -(2**31) <= step < 2**31:
            raise ValueError("rotation must be an int32")
        return self._call("fr_unary", 1, a, step)

    def level_down(self, a, level):
        if not isinstance(level, int) or not 0 <= level <= a.level:
            raise ValueError(
                "level-down target must be between zero and the input level"
            )
        return self._call("fr_unary", 2, a, level)

    def mod_raise(self, a, target_level, public_bound=1.0):
        """Refresh with a declared magnitude bound; never inspect secret values."""
        public_bound = _public_bound(public_bound)
        if (
            not isinstance(target_level, int)
            or target_level not in self.profile["mod_raise_targets"]
        ):
            raise ValueError(
                "requested ModRaise target was not prepared at client setup"
            )
        return self._call("fr_mod_raise_bounded", a, target_level, public_bound)

    def slot_to_coeff(self, a):
        return self._call("fr_unary", 4, a, 0)

    def coeff_to_slot(self, a):
        return self._call("fr_unary", 5, a, 0)

    @property
    def kernel_catalog(self):
        """JSON-only provider kernel contracts; no model or ciphertext contents."""
        return self._info("fr_kernel_catalog")

    @contextmanager
    def semantic_node(self, name, signature, current_implementation):
        """Attribute subsequent commands to a public semantic origin.

        Scopes copy this metadata while capturing. Timing is collected only
        during actual synchronized command execution and resets per scope.
        Nested contexts restore their previous origin, including on exceptions.
        No model parameters or ciphertext/plaintext contents are recorded.
        """
        origin = (name, signature, current_implementation)
        if any(
            not isinstance(value, str) or not value or "\0" in value for value in origin
        ):
            raise ValueError(
                "semantic origin fields must be nonempty strings without NUL"
            )
        with self._session.lock:
            self._check()
            previous = self._session.semantic_origin
            library = self._session.library
            library.check(
                library.lib.fr_set_semantic_origin(
                    self._session.handle, *(value.encode() for value in origin)
                )
            )
            self._session.semantic_origin = origin
            try:
                yield
            except BaseException:
                raise
            else:
                library.check(
                    library.lib.fr_complete_semantic_origin(self._session.handle)
                )
            finally:
                restored = (
                    tuple(value.encode() for value in previous)
                    if previous
                    else (None, None, None)
                )
                library.check(
                    library.lib.fr_set_semantic_origin(self._session.handle, *restored)
                )
                self._session.semantic_origin = previous

    @property
    def implementation_identity(self):
        """Stable SHA-256 identities of the actual loaded adapter/provider."""
        with self._session.lock:
            self._check()
            return self._session.implementation_identity.copy()

    def tensor_kernel(self, operation, inputs, attributes):
        """Submit one contracted tensor operation, eager or inside run_scope.

        Contributors declare their input/output metadata, temporary ciphertext
        reservation and native executor in the registry. Capture performs only
        metadata inference; one ScopeKernel executes the semantic command. The
        returned list owns every output handle, and only scope exports survive
        successful capture. Native failures poison capture even when caught.
        """
        with self._session.lock:
            self._check()
            if not isinstance(operation, str) or "\0" in operation:
                raise ValueError("tensor kernel operation must be a string without NUL")
            inputs = tuple(inputs)
            handles = (C.c_void_p * len(inputs))(
                *(_operand(self._session, value) for value in inputs)
            )
            # Strict JSON parsing and finite-number validation happen natively,
            # so caught validation errors still invalidate the entire capture.
            serialized = json.dumps(attributes, separators=(",", ":")).encode()
            entry = next(
                (
                    item
                    for item in self.kernel_catalog["kernels"]
                    if item["operation"] == operation
                ),
                None,
            )
            capacity = entry["output_arity"] if entry else 0
            outputs = (C.c_void_p * capacity)()
            output_count = C.c_size_t()
            self._session.library.check(
                self._session.library.lib.fr_tensor_kernel(
                    self._session.handle,
                    operation.encode(),
                    handles,
                    len(inputs),
                    serialized,
                    outputs,
                    capacity,
                    C.byref(output_count),
                )
            )
            result = []
            try:
                for index in range(output_count.value):
                    result.append(Ciphertext(self._session, C.c_void_p(outputs[index])))
                    outputs[index] = None
                return result
            except BaseException:
                for value in result:
                    value.close()
                for handle in outputs:
                    if handle:
                        self._session.library.lib.fr_value_free(handle)
                raise

    def run_scope(
        self,
        capture,
        *,
        max_commands=1_000_000,
        max_values=1_100_000,
        max_input_edges=2_000_000,
        max_exports=10_000,
        max_duration_ms=0,
    ):
        """Capture one full encrypted operation graph, then execute its exports.

        ``capture`` returns a flat sequence of ciphertexts from this context.
        Inputs must already be encrypted. During the callback, evaluator calls
        record commands and return symbolic handles with public metadata. Only
        exported handles become concrete when the complete graph succeeds;
        other captured handles expire. The client cannot encrypt or decrypt
        during capture. Plaintext vectors are copied by the native bridge.

        An existing output list is returned unchanged. Other sequences become
        lists containing the same Ciphertext instances. Timing separates Python
        capture from native finalization, validation and execution; it is not a
        measurement of device execution alone. A nonzero ``max_duration_ms``
        includes capture and is checked between native execution spans; it
        cannot preempt an individual encrypted operation.
        """
        if not callable(capture):
            raise TypeError("scope capture must be callable")
        capacities = (max_commands, max_values, max_input_edges, max_exports)
        for name, value in zip(
            ("max_commands", "max_values", "max_input_edges", "max_exports"), capacities
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value <= C.c_size_t(-1).value
            ):
                raise ValueError(name + " must be a positive size_t")
        if (
            isinstance(max_duration_ms, bool)
            or not isinstance(max_duration_ms, int)
            or not 0 <= max_duration_ms <= C.c_size_t(-1).value
        ):
            raise ValueError("max_duration_ms must be a nonnegative size_t")
        session = self._session
        with session.lock:
            self._check()
            if session.scope_active:
                raise RuntimeError(
                    "nested Runtime scopes are forbidden during scope capture"
                )
            outputs, failure = None, None
            capture_seconds = 0.0

            @_ScopeCallback
            def callback(_):
                nonlocal outputs, failure, capture_seconds
                capture_start = time.perf_counter()
                session.capturing = True
                try:
                    returned = capture()
                    if not isinstance(returned, Sequence):
                        raise TypeError(
                            "scope capture must return a flat sequence of Ciphertext exports"
                        )
                    outputs = returned if isinstance(returned, list) else list(returned)
                    # Validate the full Python export list before registering any
                    # exports. Native validation also checks symbolic lifetimes.
                    handles = [_operand(session, value) for value in outputs]
                    for handle in handles:
                        session.library.check(
                            session.library.lib.fr_scope_export(session.handle, handle)
                        )
                    return 0
                except BaseException as error:
                    # ctypes cannot propagate callback exceptions into C++.
                    # Preserve the original object and traceback for the caller.
                    failure = (error, error.__traceback__)
                    return -1
                finally:
                    session.capturing = False
                    capture_seconds += time.perf_counter() - capture_start

            session.scope_active = True
            started = time.perf_counter()
            try:
                status = session.library.lib.fr_run_scope(
                    session.handle, callback, None, *capacities, max_duration_ms
                )
            finally:
                elapsed = time.perf_counter() - started
                session.capturing = False
                session.scope_active = False
                self.last_scope_timing = {
                    "capture_seconds": capture_seconds,
                    "execution_and_finalize_seconds": max(
                        0.0, elapsed - capture_seconds
                    ),
                    "total_seconds": elapsed,
                }
            if failure is not None:
                error, original_traceback = failure
                raise error.with_traceback(original_traceback)
            session.library.check(status)
            return outputs

    def refresh_plan(self, target_level):
        with self._session.lock:
            self._check()
            if (
                not isinstance(target_level, int)
                or not 0 <= target_level <= self.profile["top_level"]
            ):
                raise ValueError("requested level is outside the evaluation chain")
            return self._session.library.info(
                "fr_refresh_plan", self._session.handle, target_level
            )

    def synchronize(self):
        with self._session.lock:
            self._check()
            self._session.library.check(
                self._session.library.lib.fr_synchronize(self._session.handle)
            )

    @property
    def device_info(self):
        return self._info("fr_device_info")

    def _info(self, name):
        with self._session.lock:
            self._check()
            return self._session.library.info(name, self._session.handle)

    @property
    def profile(self):
        return self._info("fr_profile")

    @property
    def stats(self):
        return self._info("fr_provider_stats")

    @property
    def runtime_stats(self):
        return self._info("fr_runtime_stats")

    def close(self):
        with self._session.lock:
            self._closed = True


def create_runtime_context(
    profile="Native128",
    rotations=(),
    library_path=None,
    mod_raise_levels=None,
    *,
    max_live_bytes=0,
    runtime_library_path=None,
    device="cpu",
):
    """Create client/setup keys, then transfer only the evaluator to Runtime."""
    global _LOADED_DEVICE
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    for loaded in (_LOADED_DEVICE, native._LOADED_DEVICE):
        if loaded is not None and loaded != device:
            raise RuntimeError(
                "CPU and CUDA native CKKS builds share a library identity; select one device per process"
            )
    if (
        isinstance(max_live_bytes, bool)
        or not isinstance(max_live_bytes, int)
        or not 0 <= max_live_bytes <= C.c_size_t(-1).value
    ):
        raise ValueError("max_live_bytes must be a nonnegative size_t")
    if profile not in ("Native128", "NativeTest"):
        raise ValueError("unsupported native CKKS profile")
    top_level = 8
    if mod_raise_levels is None:
        mod_raise_levels = (2, 8)
    rotations, mod_raise_levels = tuple(rotations), tuple(mod_raise_levels)
    if any(
        not isinstance(step, int) or not -(2**31) <= step < 2**31 for step in rotations
    ):
        raise ValueError("rotation steps must be int32 values")
    if any(
        not isinstance(level, int) or not 0 <= level <= top_level
        for level in mod_raise_levels
    ):
        raise ValueError("invalid ModRaise target level")
    library = _Library(
        runtime_library_path or (CUDA_LIBRARY if device == "cuda" else LIBRARY)
    )
    # Check the actual loaded implementation before costly key generation or
    # transferring any C++ handles. Paths and requested device labels alone do
    # not establish that the Runtime and client share a provider ABI instance.
    provider_library = native._Library(
        library_path or (native.CUDA_LIBRARY if device == "cuda" else native.LIBRARY)
    )
    if (
        library.lib.fr_provider_library_identity()
        != provider_library.lib.fn_library_identity()
    ):
        raise RuntimeError(
            "Runtime and client use different provider library instances"
        )
    _LOADED_DEVICE = device
    client, evaluator = native.create_context(
        profile=profile,
        rotations=rotations,
        library_path=library_path,
        mod_raise_levels=mod_raise_levels,
        device=device,
    )
    result = C.c_void_p()
    try:
        library.check(
            library.lib.fr_create(evaluator._handle, max_live_bytes, C.byref(result))
        )
        evaluator._handle = None  # Ownership transferred only after native success.
        session = _Session(library, result, client._library)
        result = None  # The session now exclusively owns the native Runtime.
        return Client(client, session), Evaluator(session)
    except BaseException:
        client.close()
        evaluator.close()
        if result:
            library.lib.fr_runtime_free(result)
        raise


create_context = create_runtime_context
