"""Packed tensor lowering for runtime-backed model programs. Shapes/index maps are public; values are opaque.

ReferenceVectorBackend is explicitly NONCRYPTOGRAPHIC. It is used to validate
lowering algebra; it never constitutes an encrypted execution measurement.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import weakref
import numpy as np
import torch


def array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass(eq=False)
class ReferenceCipher:
    value: np.ndarray
    level: int = 17
    representation: str = "slots"

    @property
    def count(self):
        return self.value.size


class ReferenceVectorBackend:
    kind = "noncryptographic_reference"
    profile = {"evaluation_top_level": 17, "mod_raise_targets": [2, 8, 17]}

    def __init__(self):
        self.stats = {
            "intermediate_decryptions": 0,
            "terminal_decryptions": 0,
            "mod_raise_calls": 0,
            "arithmetic_calls": 0,
        }

    def encrypt(self, x):
        return ReferenceCipher(np.array(x, dtype=float, copy=True))

    def decrypt(self, x, public_bound=1.0):
        self.stats["terminal_decryptions"] += 1
        return x.value.copy()

    def _binary(self, x, y, op, depth=0):
        self.stats["arithmetic_calls"] += 1
        return ReferenceCipher(op(x.value, y.value), min(x.level, y.level) - depth)

    def add(self, x, y):
        return self._binary(x, y, np.add)

    def sub(self, x, y):
        return self._binary(x, y, np.subtract)

    def mul(self, x, y):
        return self._binary(x, y, np.multiply, 1)

    def add_plain(self, x, y):
        return ReferenceCipher(x.value + array(y), x.level)

    def mul_plain(self, x, y):
        return ReferenceCipher(x.value * array(y), x.level - 1)

    def neg(self, x):
        return ReferenceCipher(-x.value, x.level)

    def rotate(self, x, k):
        return ReferenceCipher(np.roll(x.value, -k), x.level)

    def mod_raise(self, x, target_level, public_bound=1.0):
        if target_level not in self.profile["mod_raise_targets"]:
            raise ValueError("unprepared target level")
        self.stats["mod_raise_calls"] += 1
        return ReferenceCipher(x.value.copy(), target_level)

    def exact_unary(self, x, fn):
        return ReferenceCipher(fn(x.value), x.level)


class Tensor:
    """Immutable public layout over immutable encrypted chunks."""

    def __init__(self, engine, chunks, indices, dtype=torch.float32, public_bound=1.0):
        self.engine = engine
        self.chunks = tuple(chunks)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = tuple(self.indices.shape)
        self.dtype = dtype
        self._packed = None
        self.public_bound = float(public_bound)
        if not math.isfinite(self.public_bound) or self.public_bound < 0:
            raise ValueError("public magnitude bound must be finite and nonnegative")

    def __array__(self, *args, **kwargs):
        raise TypeError(
            "encrypted tensors cannot be converted to plaintext arrays; use explicit client egress"
        )

    def numel(self):
        return self.indices.size

    @property
    def ndim(self):
        return len(self.shape)


class TensorEngine:
    def __init__(
        self, backend, *, client=None, slots=256, target_level=8, exact_reference=False
    ):
        if slots < 1 or slots & (slots - 1):
            raise ValueError("slots must be a positive power of two")
        self.backend = backend
        self.client = client or backend
        self.slots = slots
        self.target_level = target_level
        self.exact_reference = exact_reference
        self._raised = weakref.WeakKeyDictionary()
        self._scope_packed_views = None
        if exact_reference and backend.kind != "noncryptographic_reference":
            raise ValueError(
                "exact host nonlinear evaluation is forbidden for encrypted backends"
            )
        self.stats = {
            "gather_rotations": 0,
            "gather_masks": 0,
            "tensor_matmuls": 0,
            "mod_raise_requests": 0,
            "maximum_refresh_public_bound": 1.0,
            "maximum_egress_public_bound": 1.0,
        }

    def run_scope(self, capture, **options):
        """Lower one computation and its output layouts into a single native Scope.

        The backend returns metadata-bearing symbols during capture. Rebuild the
        result tree over explicitly exported, contiguous chunks so later client
        egress cannot accidentally perform layout arithmetic outside the scope.
        Public-only leaves remain compile-time results.
        """
        from torch.utils._pytree import tree_flatten, tree_unflatten

        if self._scope_packed_views is not None:
            raise ValueError("nested tensor scopes are not supported")
        if self.backend.kind == "noncryptographic_reference":
            raise ValueError("reference execution is not a Fhenomenon scope")
        self._raised.clear()
        self._scope_packed_views = []
        output = None

        def record():
            nonlocal output
            leaves, spec = tree_flatten(capture())
            materialized = []
            exports = []
            tensors = {}
            for leaf in leaves:
                if not isinstance(leaf, Tensor):
                    materialized.append(leaf)
                    continue
                if leaf.engine is not self:
                    raise ValueError("scope output belongs to another tensor engine")
                if id(leaf) not in tensors:
                    chunks = self._pack(leaf)
                    tensors[id(leaf)] = self._new(
                        chunks, leaf.shape, leaf.dtype, leaf.public_bound
                    )
                    exports.extend(chunks)
                materialized.append(tensors[id(leaf)])
            output = tree_unflatten(materialized, spec)
            return exports

        try:
            self.backend.run_scope(record, **options)
            return output
        finally:
            # A view passed into capture may outlive a failed or successful
            # scope. Its temporary symbolic packing must never escape as a
            # reusable cache. Fresh materialized exports have separate wrappers.
            for reference in self._scope_packed_views:
                view = reference()
                if view is not None:
                    view._packed = None
            self._scope_packed_views = None
            self._raised.clear()

    def from_public(self, x, dtype=None):
        return torch.as_tensor(x, dtype=dtype) if dtype else torch.as_tensor(x)

    constant = from_public

    def encrypt(self, x, public_bound=None):
        a = array(x)
        flat = a.astype(float).ravel()
        chunks = []
        if not np.isfinite(flat).all():
            raise ValueError("client input must be finite")
        if public_bound is None:
            # Auto-range is available only to the explicitly noncryptographic
            # oracle. Real encryption never reveals a data-derived range.
            public_bound = (
                max(1.0, float(np.max(np.abs(flat), initial=0.0)))
                if self.backend.kind == "noncryptographic_reference"
                else 1.0
            )
        if (
            not math.isfinite(public_bound)
            or public_bound < 0
            or np.max(np.abs(flat), initial=0.0) > public_bound
        ):
            raise ValueError("client input exceeds its declared public bound")
        for i in range(0, len(flat), self.slots):
            v = np.zeros(self.slots)
            v[: len(flat[i : i + self.slots])] = flat[i : i + self.slots]
            chunks.append(self.client.encrypt(v.tolist()))
        dtype = torch.as_tensor(a).dtype
        t = Tensor(
            self, chunks, np.arange(len(flat)).reshape(a.shape), dtype, public_bound
        )
        t._packed = tuple(chunks)
        return t

    def decrypt(self, x):
        if not isinstance(x, Tensor):
            return array(x)
        self.stats["maximum_egress_public_bound"] = max(
            self.stats["maximum_egress_public_bound"], x.public_bound
        )
        v = (
            np.concatenate(
                [
                    array(self.client.decrypt(c, public_bound=max(1.0, x.public_bound)))
                    for c in self._pack(x)
                ]
            )
            if x.numel()
            else np.empty(0)
        )
        return v[: x.numel()].reshape(x.shape)

    def _ready(self, c, depth=1, public_bound=1.0):
        if c.level < depth:
            public_bound = max(1.0, float(public_bound))
            cache = self._raised.setdefault(c, {})
            cached = cache.get(public_bound)
            if cached is None:
                self.stats["mod_raise_requests"] += 1
                self.stats["maximum_refresh_public_bound"] = max(
                    self.stats["maximum_refresh_public_bound"], public_bound
                )
                cached = self.backend.mod_raise(
                    c, self.target_level, public_bound=public_bound
                )
                cache[public_bound] = cached
            return cached
        return c

    def _rot(self, c, step):
        step %= self.slots
        bit = 1
        while step:
            if step & 1:
                c = self.backend.rotate(c, bit)
                self.stats["gather_rotations"] += 1
            bit *= 2
            step >>= 1
        return c

    def _pack(self, x):
        if x._packed is not None:
            return x._packed
        if self._scope_packed_views is not None:
            self._scope_packed_views.append(weakref.ref(x))
        out = []
        flat = x.indices.ravel()
        for start in range(0, len(flat), self.slots):
            index = flat[start : start + self.slots]
            groups = {}
            for dst, src in enumerate(index):
                if src >= 0:
                    key = (
                        int(src) // self.slots,
                        (int(src) % self.slots - dst) % self.slots,
                    )
                    groups.setdefault(key, []).append(dst)
            total = None
            for (chunk, shift), positions in groups.items():
                c = self._rot(
                    self._ready(x.chunks[chunk], public_bound=x.public_bound)
                    if len(positions) != self.slots
                    else x.chunks[chunk],
                    shift,
                )
                if len(positions) != self.slots:
                    mask = np.zeros(self.slots)
                    mask[positions] = 1.0
                    c = self.backend.mul_plain(
                        self._ready(c, public_bound=x.public_bound), mask.tolist()
                    )
                    self.stats["gather_masks"] += 1
                total = c if total is None else self.backend.add(total, c)
            if total is None:
                if not x.chunks:
                    raise ValueError(
                        "cannot materialize private zero without an encrypted source"
                    )
                total = self.backend.sub(x.chunks[0], x.chunks[0])
            out.append(total)
        x._packed = tuple(out)
        return x._packed

    def _new(self, chunks, shape, dtype=torch.float32, public_bound=1.0):
        chunks = tuple(chunks)
        expected = (math.prod(shape) + self.slots - 1) // self.slots
        if len(chunks) != expected or any(
            chunk.count != self.slots for chunk in chunks
        ):
            raise ValueError(
                "output ciphertext count/packing does not cover the declared tensor"
            )
        t = Tensor(
            self,
            chunks,
            np.arange(math.prod(shape)).reshape(shape),
            dtype,
            public_bound,
        )
        t._packed = tuple(chunks)
        return t

    def _binary(self, a, b, op):
        if not isinstance(a, Tensor) and not isinstance(b, Tensor):
            return getattr(torch, op)(torch.as_tensor(a), torch.as_tensor(b))
        shape = np.broadcast_shapes(
            a.shape if isinstance(a, Tensor) else array(a).shape,
            b.shape if isinstance(b, Tensor) else array(b).shape,
        )
        aa = self.expand(a, shape)
        bb = self.expand(b, shape)
        ac = self._pack(aa) if isinstance(aa, Tensor) else None
        bc = self._pack(bb) if isinstance(bb, Tensor) else None
        av = array(aa).ravel() if ac is None else None
        bv = array(bb).ravel() if bc is None else None
        result = []
        for i in range((math.prod(shape) + self.slots - 1) // self.slots):
            if ac is not None and bc is not None:
                x, y = ac[i], bc[i]
                if op == "mul":
                    x, y = (
                        self._ready(x, public_bound=aa.public_bound),
                        self._ready(y, public_bound=bb.public_bound),
                    )
                c = getattr(self.backend, op)(x, y)
            else:
                private = ac[i] if ac is not None else bc[i]
                public = (bv if ac is not None else av)[
                    i * self.slots : (i + 1) * self.slots
                ]
                p = np.zeros(self.slots)
                p[: len(public)] = public
                if op == "sub":
                    if ac is not None:
                        p = -p
                    else:
                        private = self.backend.neg(private)
                if op == "mul":
                    private = self._ready(
                        private,
                        public_bound=(aa if ac is not None else bb).public_bound,
                    )
                c = getattr(self.backend, "mul_plain" if op == "mul" else "add_plain")(
                    private, p.tolist()
                )
            result.append(c)
        a_bound = (
            a.public_bound
            if isinstance(a, Tensor)
            else float(np.max(np.abs(array(a)), initial=0.0))
        )
        b_bound = (
            b.public_bound
            if isinstance(b, Tensor)
            else float(np.max(np.abs(array(b)), initial=0.0))
        )
        bound = a_bound * b_bound if op == "mul" else a_bound + b_bound
        return self._new(
            result, shape, (a if isinstance(a, Tensor) else b).dtype, bound
        )

    def add(self, a, b):
        return self._binary(a, b, "add")

    def sub(self, a, b):
        return self._binary(a, b, "sub")

    def mul(self, a, b):
        return self._binary(a, b, "mul")

    def neg(self, a):
        return self.mul(a, -1)

    def div(self, a, b):
        return (
            self.mul(a, self.nonlinear("reciprocal", (b,)))
            if isinstance(b, Tensor)
            else self.mul(a, 1.0 / array(b))
        )

    def pow(self, a, p):
        if not isinstance(p, (int, float)) or p < 0 or int(p) != p:
            if p == -1:
                return self.nonlinear("reciprocal", (a,))
            if p == -0.5:
                return self.nonlinear("rsqrt", (a,))
            raise NotImplementedError(
                "only public nonnegative integer powers and reciprocal/rsqrt"
            )
        out = torch.ones(a.shape if isinstance(a, Tensor) else array(a).shape)
        while p:
            if int(p) & 1:
                out = self.mul(out, a)
            p = int(p) // 2
            if p:
                a = self.mul(a, a)
        return out

    def reshape(self, a, shape):
        return (
            Tensor(
                self, a.chunks, a.indices.reshape(tuple(shape)), a.dtype, a.public_bound
            )
            if isinstance(a, Tensor)
            else torch.as_tensor(a).reshape(tuple(shape))
        )

    def permute(self, a, dims):
        return (
            Tensor(
                self,
                a.chunks,
                a.indices.transpose(tuple(dims)),
                a.dtype,
                a.public_bound,
            )
            if isinstance(a, Tensor)
            else torch.as_tensor(a).permute(tuple(dims))
        )

    def expand(self, a, shape):
        shape = tuple(shape)
        if isinstance(a, Tensor):
            shape = tuple(
                a.shape[i - len(shape) + len(a.shape)] if n == -1 else n
                for i, n in enumerate(shape)
            )
            return Tensor(
                self,
                a.chunks,
                np.broadcast_to(a.indices, shape),
                a.dtype,
                a.public_bound,
            )
        return torch.broadcast_to(torch.as_tensor(a), shape)

    def slice(self, a, dim, start, end, step=1):
        shape = a.shape if hasattr(a, "shape") else array(a).shape
        selector = [slice(None)] * len(shape)
        selector[dim] = slice(start, end, step)
        if isinstance(a, Tensor):
            return Tensor(
                self, a.chunks, a.indices[tuple(selector)], a.dtype, a.public_bound
            )
        return torch.as_tensor(a)[tuple(selector)]

    def slice_backward(self, g, shape, dim, start, end, step=1):
        selector = [slice(None)] * len(shape)
        selector[dim] = slice(start, end, step)
        if isinstance(g, Tensor):
            indices = np.full(shape, -1, dtype=np.int64)
            indices[tuple(selector)] = g.indices
            return Tensor(self, g.chunks, indices, g.dtype, g.public_bound)
        result = torch.zeros(tuple(shape), dtype=torch.as_tensor(g).dtype)
        result[tuple(selector)] = torch.as_tensor(g)
        return result

    def cat(self, values, dim=0):
        if all(not isinstance(v, Tensor) for v in values):
            return torch.cat([torch.as_tensor(v) for v in values], dim)
        # Public elements can be injected using an existing encrypted zero.
        seed = next(v for v in values if isinstance(v, Tensor))
        chunks = []
        maps = []
        bounds = []
        for v in values:
            if not isinstance(v, Tensor):
                zero = self.mul(
                    self.reshape(self.slice(self.reshape(seed, (-1,)), 0, 0, 1), (1,)),
                    0.0,
                )
                v = self.add(self.expand(zero, array(v).shape), v)
            bounds.append(v.public_bound)
            maps.append(
                np.where(v.indices < 0, -1, v.indices + len(chunks) * self.slots)
            )
            chunks.extend(v.chunks)
        return Tensor(
            self, chunks, np.concatenate(maps, axis=dim), seed.dtype, max(bounds)
        )

    def stack(self, values, dim=0):
        shape = list(values[0].shape)
        dim %= len(shape) + 1
        shape.insert(dim, 1)
        return self.cat([self.reshape(v, shape) for v in values], dim)

    def sum(self, a, dims=None, keepdim=False):
        ndim = len(a.shape)
        dims = (
            tuple(range(ndim))
            if dims is None or len(dims) == 0
            else tuple(sorted({int(d) % ndim for d in dims}))
        )
        if not isinstance(a, Tensor):
            return torch.as_tensor(a).sum(dims, keepdim=keepdim)
        kept = [d for d in range(ndim) if d not in dims]
        arranged = self.permute(a, kept + list(dims))
        width = math.prod(a.shape[d] for d in dims)
        arranged = self.reshape(arranged, (-1, width))
        total = None
        for j in range(width):
            item = self.reshape(self.slice(arranged, 1, j, j + 1), (-1,))
            total = item if total is None else self.add(total, item)
        shape = (
            tuple(1 if d in dims else a.shape[d] for d in range(ndim))
            if keepdim
            else tuple(a.shape[d] for d in kept)
        )
        return self.reshape(total, shape) if total is not None else torch.zeros(shape)

    def mean(self, a, dims=None, keepdim=False):
        dims = (
            tuple(range(len(a.shape)))
            if dims is None or len(dims) == 0
            else tuple(dims)
        )
        return self.div(self.sum(a, dims, keepdim), math.prod(a.shape[d] for d in dims))

    def matmul(self, a, b):
        self.stats["tensor_matmuls"] += 1
        ash, bsh = tuple(a.shape), tuple(b.shape)
        if len(ash) < 2 or len(bsh) < 2:
            raise NotImplementedError("matrix operands need at least two dimensions")
        if ash[-1] != bsh[-2]:
            raise ValueError("matrix dimensions do not match")
        batch = np.broadcast_shapes(ash[:-2], bsh[:-2])
        m, k, n = ash[-2], ash[-1], bsh[-1]
        a = self.expand(a, (*batch, m, k))
        b = self.expand(b, (*batch, k, n))
        total = None
        for j in range(k):
            aa = self.reshape(self.slice(a, -1, j, j + 1), (*batch, m, 1))
            bb = self.reshape(self.slice(b, -2, j, j + 1), (*batch, 1, n))
            p = self.mul(aa, bb)
            total = p if total is None else self.add(total, p)
        return total if total is not None else torch.zeros((*batch, m, n))

    def zeros_like(self, a):
        return torch.zeros(a.shape, dtype=a.dtype)

    def clone(self, a):
        return a

    def conv2d(self, x, w, bias, stride, padding, dilation, groups):
        n, c, h, ww = x.shape
        out, per, kh, kw = w.shape
        sy, sx = stride
        py, px = padding
        dy, dx = dilation
        oh = (h + 2 * py - dy * (kh - 1) - 1) // sy + 1
        ow = (ww + 2 * px - dx * (kw - 1) - 1) // sx + 1
        if groups != 1:
            raise NotImplementedError(
                "grouped convolution requires a separate lowering"
            )
        if c != per:
            raise ValueError("convolution channel mismatch")
        # im2col is a public index mapping, including padding sentinels.
        idx = np.full((n, oh, ow, c, kh, kw), -1, dtype=np.int64)
        for iy in range(oh):
            for ix in range(ow):
                for ky in range(kh):
                    for kx in range(kw):
                        yy = iy * sy - py + ky * dy
                        xx = ix * sx - px + kx * dx
                        if 0 <= yy < h and 0 <= xx < ww:
                            for bn in range(n):
                                for ch in range(c):
                                    idx[bn, iy, ix, ch, ky, kx] = (
                                        (bn * c + ch) * h + yy
                                    ) * ww + xx
        if isinstance(x, Tensor):
            mapped = np.where(idx < 0, -1, x.indices.ravel()[np.maximum(idx, 0)])
            col = Tensor(self, x.chunks, mapped, x.dtype, x.public_bound)
        else:
            data = array(x).ravel()
            col = torch.as_tensor(np.where(idx < 0, 0, data[np.maximum(idx, 0)]))
        col = self.reshape(col, (n * oh * ow, c * kh * kw))
        weights = self.permute(self.reshape(w, (out, -1)), (1, 0))
        result = self.matmul(col, weights)
        if bias is not None:
            result = self.add(result, bias)
        return self.permute(self.reshape(result, (n, oh, ow, out)), (0, 3, 1, 2))

    def conv2d_backward(self, g, x, w, stride, padding, dilation, groups, output_mask):
        if groups != 1:
            raise NotImplementedError("grouped convolution backward")
        n, c, h, ww = x.shape
        out, _, kh, kw = w.shape
        _, _, oh, ow = g.shape
        sy, sx = stride
        py, px = padding
        dy, dx = dilation
        dxout = None
        dwout = None
        db = self.sum(g, (0, 2, 3), False) if output_mask[2] else None
        if output_mask[0]:
            # Scatter each public kernel position; sums overlap without revealing values.
            for ky in range(kh):
                for kx in range(kw):
                    wij = self.reshape(
                        self.slice(self.slice(w, 2, ky, ky + 1), 3, kx, kx + 1),
                        (out, c),
                    )
                    gg = self.reshape(self.permute(g, (0, 2, 3, 1)), (-1, out))
                    vals = self.permute(
                        self.reshape(self.matmul(gg, wij), (n, oh, ow, c)), (0, 3, 1, 2)
                    )
                    ids = np.full((n, c, h, ww), -1, dtype=np.int64)
                    for iy in range(oh):
                        for ix in range(ow):
                            yy = iy * sy - py + ky * dy
                            xx = ix * sx - px + kx * dx
                            if 0 <= yy < h and 0 <= xx < ww:
                                ids[:, :, yy, xx] = (
                                    np.arange(n * c).reshape(n, c) * oh * ow
                                    + iy * ow
                                    + ix
                                )
                    if isinstance(vals, Tensor):
                        part = Tensor(
                            self,
                            vals.chunks,
                            np.where(
                                ids < 0, -1, vals.indices.ravel()[np.maximum(ids, 0)]
                            ),
                            vals.dtype,
                            vals.public_bound,
                        )
                    else:
                        part = torch.as_tensor(
                            np.where(
                                ids < 0, 0, array(vals).ravel()[np.maximum(ids, 0)]
                            )
                        )
                    dxout = part if dxout is None else self.add(dxout, part)
        if output_mask[1]:
            rows = []
            gg = self.permute(
                self.reshape(self.permute(g, (0, 2, 3, 1)), (-1, out)), (1, 0)
            )
            for ky in range(kh):
                cols = []
                for kx in range(kw):
                    ids = np.full((n, oh, ow, c), -1, dtype=np.int64)
                    for iy in range(oh):
                        for ix in range(ow):
                            yy = iy * sy - py + ky * dy
                            xx = ix * sx - px + kx * dx
                            if 0 <= yy < h and 0 <= xx < ww:
                                for bn in range(n):
                                    ids[bn, iy, ix, :] = (
                                        (bn * c + np.arange(c)) * h + yy
                                    ) * ww + xx
                    if isinstance(x, Tensor):
                        part = Tensor(
                            self,
                            x.chunks,
                            np.where(
                                ids < 0, -1, x.indices.ravel()[np.maximum(ids, 0)]
                            ),
                            x.dtype,
                            x.public_bound,
                        )
                    else:
                        part = torch.as_tensor(
                            np.where(ids < 0, 0, array(x).ravel()[np.maximum(ids, 0)])
                        )
                    cols.append(self.matmul(gg, self.reshape(part, (-1, c))))
                rows.append(self.stack(cols, -1))
            dwout = self.stack(rows, -2)
        return dxout, dwout, db

    def require_bound(self, x, public_bound):
        """Attach a public mathematical bound; encrypted values are not inspected.

        Callers must supply an independently justified input requirement or
        recurrence invariant. This is not a range clamp or an encrypted check.
        """
        if not isinstance(x, Tensor):
            return x
        chunks = self._pack(x)
        if self.backend.kind == "noncryptographic_reference":
            maximum = max(
                (float(np.max(np.abs(c.value), initial=0.0)) for c in chunks),
                default=0.0,
            )
            if maximum > public_bound * (1 + 1e-7) + 1e-10:
                raise ValueError(
                    f"public bound {public_bound} violated in reference validation: {maximum}"
                )
        return self._new(chunks, x.shape, x.dtype, public_bound)

    def nonlinear(self, name, args):
        from .approximations import evaluate

        return evaluate(self, name, args)
