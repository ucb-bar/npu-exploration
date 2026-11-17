"""Direct E2M1/E3M2/E4M3FN fixed-K32 quantizer. NumPy only."""

import numpy as np


def grid(f):
    c = np.arange(8 if f == 4 else 32 if f == 6 else 127, dtype=np.uint8)
    if f == 4:
        v = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    else:
        mb, bias = (2, 3) if f == 6 else (3, 7)
        e = c.astype(int) >> mb
        q = c.astype(int) & ((1 << mb) - 1)
        v = np.where(
            e == 0,
            np.ldexp(q.astype(np.float32) / (1 << mb), 1 - bias),
            np.ldexp(1 + q.astype(np.float32) / (1 << mb), e - bias),
        )
    return v.astype(np.float32)


def quant(x, f):
    # x [K32blocks,32,N]. E8M0=2**floor(log2(block amax)).
    a = np.max(np.abs(x), axis=1)
    exp = np.frexp(a)[1] - 1
    exp = np.where(a == 0, 0, exp)
    exp = np.clip(exp, -127, 127)
    y = np.ldexp(x, -exp[:, None, :])
    v = grid(f)
    a = np.abs(y)
    hi = np.minimum(np.searchsorted(v, a), len(v) - 1)
    lo = np.maximum(hi - 1, 0)
    dl = a - v[lo]
    dh = v[hi] - a
    ix = np.where((dl < dh) | ((dl == dh) & ((lo & 1) == 0)), lo, hi)
    code = ix.astype(np.uint8) | (
        np.signbit(y).astype(np.uint8) << (3 if f == 4 else 5 if f == 6 else 7)
    )
    return code, (exp + 127).astype(np.uint8)
