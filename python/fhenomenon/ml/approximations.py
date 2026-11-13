"""Explicit public approximation contracts; no decryption or secret range tests.

Intervals below are declared requirements, not secretly calibrated guarantees.
Checking model-wide range validity is a separate client-side validation task.
The exact reference mode is noncryptographic and forbidden for real providers.
"""

from functools import lru_cache
import math
import numpy as np
import torch
from .tensor import Tensor, array

CONTRACTS = {
    "exp": {
        "interval": [-8.0, 8.0],
        "method": "degree8 Taylor at x/64 then six squarings",
    },
    "gelu": {"interval": [-4.0, 4.0], "method": "degree24 public Chebyshev fit"},
    "gelu_backward": {
        "interval": [-4.0, 4.0],
        "method": "derivative of forward degree24 polynomial",
    },
    "reciprocal": {
        "interval": [1e-6, 131072.0],
        "method": "Newton iteration; public positive interval determines steps (at most 48)",
    },
    "rsqrt": {
        "interval": [1e-6, 16.0],
        "method": "Newton inverse square root; 32 steps",
    },
    "log": {
        "interval": [1e-6, 131072.0],
        "method": "Newton log using reciprocal and repeated square roots",
    },
    "sigmoid": {"interval": [-4.0, 4.0], "method": "degree25 public Chebyshev fit"},
}


def _exact(name, values):
    x = values[0]
    if name == "exp":
        return np.exp(x)
    if name == "log":
        return np.log(x)
    if name == "reciprocal":
        return 1 / x
    if name == "rsqrt":
        return 1 / np.sqrt(x)
    if name == "sigmoid":
        return 1 / (1 + np.exp(-x))
    if name == "gelu":
        return torch.nn.functional.gelu(
            torch.as_tensor(x), approximate=values[1] if len(values) > 1 else "none"
        ).numpy()
    if name == "gelu_backward":
        grad, x = values[:2]
        xx = torch.tensor(x, requires_grad=True)
        torch.nn.functional.gelu(
            xx, approximate=values[2] if len(values) > 2 else "none"
        ).backward(torch.as_tensor(grad))
        return xx.grad.numpy()
    raise NotImplementedError("nonlinear operation " + name)


@lru_cache(None)
def coefficients(name):
    x = np.cos(np.pi * (np.arange(256) + 0.5) / 256) * 4
    y = _exact(name, [x])
    return np.polynomial.chebyshev.chebfit(x / 4, y, 24 if name == "gelu" else 25)


def chebyshev(e, x, c):
    # Clenshaw recurrence, entirely ciphertext arithmetic.
    t = e.div(e.require_bound(x, 4.0), 4.0)
    b1 = 0.0
    b2 = 0.0
    for k in range(len(c) - 1, 0, -1):
        b0 = e.add(e.sub(e.mul(e.mul(t, 2.0), b1), b2), float(c[k]))
        # b_k=sum(c_j U_{j-k}(t)); |U_n(t)|<=n+1 on [-1,1].
        b0 = e.require_bound(b0, sum(abs(c[j]) * (j - k + 1) for j in range(k, len(c))))
        b2, b1 = b1, b0
    return e.require_bound(
        e.add(e.sub(e.mul(t, b1), b2), float(c[0])), float(np.sum(np.abs(c)))
    )


def inverse_sqrt(e, x, low, high, steps):
    """Public positive-domain Newton invariant, independent of secret values.

    Starting below 1/sqrt(x) keeps y below that limit; x*y<=sqrt(high),
    (x*y)*y<=1 and y grows by at most 1.5 each iteration. Reassociation
    avoids materializing y*y, which can be orders of magnitude larger.
    Numerical error remains a separate requirement, not proved by these bounds.
    """
    y = 1 / math.sqrt(high)
    for k in range(steps):
        xy = e.require_bound(e.mul(x, y), math.sqrt(high))
        xyy = e.require_bound(e.mul(xy, y), 1.0)
        y = e.mul(e.mul(y, 0.5), e.sub(3.0, xyy))
        y = e.require_bound(
            y, min(1 / math.sqrt(low), 1.5 ** (k + 1) / math.sqrt(high))
        )
    return y


def evaluate(e, name, args):
    if name not in CONTRACTS:
        raise NotImplementedError("unregistered private nonlinear " + name)
    interval = CONTRACTS[name]["interval"]
    if name in ("reciprocal", "rsqrt", "log") and len(args) > 1:
        if len(args) != 3:
            raise ValueError("positive interval requires public lower and upper bounds")
        low, high = args[1:]
        if (
            not all(
                isinstance(v, (float, int)) and math.isfinite(v) for v in (low, high)
            )
            or not interval[0] <= low <= high <= interval[1]
        ):
            raise ValueError("positive interval outside the supported public domain")
        interval = (float(low), float(high))
    secret = any(isinstance(x, Tensor) for x in args)
    if not secret:
        return torch.as_tensor(
            _exact(name, [array(x) if not isinstance(x, str) else x for x in args])
        )
    if e.exact_reference:
        if e.backend.kind != "noncryptographic_reference":
            raise RuntimeError("host nonlinear path forbidden")
        # Reference provider values only, not a decryption method. This path
        # validates lowering and is always reported NONCRYPTOGRAPHIC.
        values = []
        for x in args:
            if isinstance(x, Tensor):
                chunks = e._pack(x)
                values.append(
                    np.concatenate([c.value for c in chunks])[: x.numel()].reshape(
                        x.shape
                    )
                )
            else:
                values.append(x if isinstance(x, str) else array(x))
        result = e.encrypt(_exact(name, values))
        dtype = next(x.dtype for x in args if isinstance(x, Tensor))
        return e._new(result.chunks, result.shape, dtype, result.public_bound)
    x = args[0]
    if name != "gelu_backward":
        declared = max(abs(v) for v in interval)
        x = (
            e.require_bound(x, min(x.public_bound, declared))
            if isinstance(x, Tensor)
            else x
        )
    if name == "exp":
        t = e.div(x, 64.0)
        out = 1 / math.factorial(8)
        for k in range(7, -1, -1):
            out = e.add(e.mul(out, t), 1 / math.factorial(k))
        for _ in range(6):
            out = e.mul(out, out)
        return out
    if name in ("gelu", "sigmoid"):
        if name == "gelu" and len(args) > 1 and args[1] != "none":
            raise NotImplementedError(
                "GELU tanh approximation needs its own polynomial contract"
            )
        return chebyshev(e, x, coefficients(name))
    if name == "gelu_backward":
        if len(args) > 2 and args[2] != "none":
            raise NotImplementedError("GELU tanh derivative")
        derivative = np.polynomial.chebyshev.chebder(coefficients("gelu")) / 4.0
        return e.mul(args[0], chebyshev(e, args[1], derivative))
    if name == "reciprocal":
        low, high = interval
        y = 1 / high
        steps = (
            1
            if low == high
            else max(1, math.ceil(math.log2(math.log(1e-13) / math.log1p(-low / high))))
            + 1
        )
        # For positive x and y0<=1/x: 0<xy<=1 and y increases
        # monotonically to 1/x; y_(k+1)<=2*y_k.
        for k in range(steps):
            xy = e.require_bound(e.mul(x, y), 1.0)
            y = e.mul(y, e.sub(2.0, xy))
            y = e.require_bound(y, min(1 / low, 2.0 ** (k + 1) / high))
        return y
    if name == "rsqrt":
        low, high = interval
        return inverse_sqrt(
            e, x, low, high, math.ceil(math.log(math.sqrt(high / low), 1.5)) + 8
        )
    if name == "log":
        # Eight square roots reduce the declared positive interval near one.
        # Each inverse square root has a PUBLIC interval-dependent iteration
        # count. No data-dependent normalization, range probe or decryption.
        low, high = interval
        z = x
        for _ in range(8):
            steps = math.ceil(math.log(math.sqrt(high / low), 1.5)) + 8
            y = inverse_sqrt(e, z, low, high, steps)
            z = e.require_bound(e.mul(z, y), math.sqrt(high))
            low, high = math.sqrt(low), math.sqrt(high)
        denominator = e.add(z, 1.0)
        inv = 1 / (1 + high)
        for k in range(8):
            product = e.require_bound(e.mul(denominator, inv), 1.0)
            inv = e.require_bound(e.mul(inv, e.sub(2.0, product)), 1 / (1 + low))
        t = e.mul(e.sub(z, 1.0), inv)
        t2 = e.mul(t, t)
        term = t
        result = t
        for k in range(1, 8):
            term = e.mul(term, t2)
            result = e.add(result, e.div(term, 2 * k + 1))
        return e.mul(result, 512.0)
    raise NotImplementedError(name)
