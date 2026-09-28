"""Host ops: the Python twins of ``backend/runtime/mx_host.h``.

The mesh does matmuls. Everything else in a layer — RMSNorm, SiLU, the elementwise product,
softmax, RoPE — has no hardware here, so it runs on the scalar core in fp32. Each such op exists
**twice**: once in C, emitted into the driver and executed on device, and once here, used for the
fp32 reference and the golden.

Two implementations of one thing is a drift risk, so it is a **tested** invariant rather than an
assumed one (``tests/selftest_mx_host.py``), and the split is not gratuitous: the C runs on the
device and the Python has to run in the reference pipeline, which cannot call it.

A :class:`~kernels.spec.HostStage` names an op from :data:`OPS` instead of carrying a closure. That
is what makes it emittable: a closure can be *run* but not *compiled*, so a kernel built from
closures can only ever execute on the host between ELFs — which is the thing
``merlin_glue_port_plan.md`` D4 abolishes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def softmax(x: np.ndarray, *, scale: float = 1.0, causal: bool = False) -> np.ndarray:
    """Row softmax of ``x[M][N]`` in fp32. Twin of ``mx_softmax_rows``.

    ``causal`` masks ``j > m`` (requires N == M). The max subtraction is taken over exactly the
    elements that survive the mask — masking after the max would change the result.
    """
    z = np.asarray(x, dtype=np.float32) * np.float32(scale)
    if causal:
        m, n = z.shape
        if n != m:
            raise ValueError(f"causal softmax needs a square tile, got {z.shape}")
        keep = np.tril(np.ones((m, n), dtype=bool))
        z = np.where(keep, z, -np.inf)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    if causal:
        e = np.where(np.isfinite(z), e, 0.0)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def rmsnorm(x: np.ndarray, *, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Twin of ``mx_rmsnorm``."""
    x = np.asarray(x, dtype=np.float32)
    inv = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + np.float32(eps))
    return (x * inv * np.asarray(weight, dtype=np.float32)).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    """Twin of ``mx_silu``."""
    x = np.asarray(x, dtype=np.float32)
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """``silu(gate) * up``. Twin of ``mx_swiglu``. TWO inputs, both computed values."""
    return (silu(gate) * np.asarray(up, dtype=np.float32)).astype(np.float32)


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residual add. Twin of a plain elementwise loop; no mx_host entry point needed."""
    return (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)).astype(np.float32)


def rope(x: np.ndarray, *, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotary embedding over the last axis, split-half. Twin of ``mx_rope``.

    Written to mirror the C's indexing exactly — ``out[m][i] = x*cos[m][i] + rot*sin[m][i]`` with
    ``rot = -x[m][i+half]`` on the low half and ``+x[m][i-half]`` on the high half — rather than
    slicing ``cos[:, :half]`` and relying on the tables being duplicated across halves. They ARE
    duplicated in llama (checked on the capture), but depending on that would be a silent trap for
    any model where they are not.
    """
    x = np.asarray(x, dtype=np.float32)
    h = x.shape[-1] // 2
    rot = np.concatenate([-x[:, h:], x[:, :h]], axis=-1)
    return (x * np.asarray(cos, np.float32) + rot * np.asarray(sin, np.float32)).astype(np.float32)


def transpose(x: np.ndarray) -> np.ndarray:
    """Twin of ``mx_transpose_f32``.

    A host op because the MX loop path IGNORES the ``A_transpose``/``B_transpose`` bits:
    ``mx_loop_ws_spad`` does ``(void)rs1;`` and never reads them (they are honoured only by the
    stock int8 ``loop_ws``). So ``S = Q @ K^T`` needs a real byte transpose on the scalar core.
    """
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32).T)


@dataclass(frozen=True)
class HostOp:
    """One host op: its Python twin, the C function to call, and which params it takes."""

    name: str
    fn: Callable[..., np.ndarray]
    c_fn: str                       #: the mx_host.h entry point
    params: tuple[str, ...] = ()    #: parameter names this op accepts
    #: How many VALUES it consumes. SwiGLU takes two (gate and up); most take one.
    arity: int = 1
    #: Does the op change the tile's shape? ``transpose`` does; the rest are elementwise or per-row.
    reshapes: bool = False


OPS: dict[str, HostOp] = {
    "softmax": HostOp("softmax", softmax, "mx_softmax_rows", ("scale", "causal")),
    "rmsnorm": HostOp("rmsnorm", rmsnorm, "mx_rmsnorm", ("weight", "eps")),
    "silu": HostOp("silu", silu, "mx_silu"),
    "swiglu": HostOp("swiglu", swiglu, "mx_swiglu", arity=2),
    "add": HostOp("add", add, "", arity=2),
    "rope": HostOp("rope", rope, "mx_rope", ("cos", "sin")),
    "transpose": HostOp("transpose", transpose, "mx_transpose_f32", reshapes=True),
}


class HostOpError(RuntimeError):
    """An unknown host op, or one given parameters it does not take."""


def get(op: str, **params) -> HostOp:
    """Resolve an op name, checking its parameters. Fails closed on a typo'd param name.

    Silently ignoring an unknown parameter is how a `scale=` becomes a no-op and a kernel quietly
    computes the wrong thing, so an unexpected key is an error.
    """
    if op not in OPS:
        raise HostOpError(f"unknown host op {op!r}; known: {sorted(OPS)}")
    o = OPS[op]
    extra = sorted(set(params) - set(o.params))
    if extra:
        raise HostOpError(
            f"host op {op!r} takes {list(o.params)}, got unexpected {extra}")
    return o
