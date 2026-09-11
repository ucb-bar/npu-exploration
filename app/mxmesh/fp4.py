"""The bit-exact FP4/FP6 mesh. 32x32 tiles (gemmini.cc:1519) and PROD_MANT_BITS=3, which is the whole difference from the FP8 model.

EXTRACTED VERBATIM from ``fp4_matmul_model.py`` by ``tools/extract_model.py`` — do not edit.
Entry points: tiled_matmul_hwlike, matrix_mx_requantize, tensor_to_custom_fp_codes
Closure: 39 definitions.
"""
from __future__ import annotations

import math
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

Tensor = torch.Tensor
QuantFn = Optional[Callable[[Tensor], Tensor]]


Tensor = torch.Tensor

INPUT_SPEC     = "fp4:e2m1"   # 1 sign + 2 exp (bias=1) + 1 mant = 4 bits

PROD_MANT_BITS = 3             # 1.m × 1.m → at most 3 fractional bits (cf. FP8: 7)

TILE_M         = 32            # M-dim of one hardware tile (cf. FP8: 16)

TILE_N         = 32            # N-dim of one hardware tile (cf. FP8: 16)

TILE_K         = 16            # K-dim of one hardware tile (same as FP8)

GROUP          = 32            # MX quantization group size along K

SCALE_SPEC     = "fpe8m0"

_FP_PRESETS = {
    "fp16":     dict(exp=5, man=10),
    "bf16":     dict(exp=8, man=7),
    "fp8:e4m3": dict(exp=4, man=3),
    "fp8:e5m2": dict(exp=5, man=2),
    "fp6:e3m2": dict(exp=3, man=2),
    "fp4:e2m1": dict(exp=2, man=1),
    "fpe8m0":   dict(exp=8, man=0),
}

def parse_fp_spec(spec: str) -> Tuple[int, int]:
    s = spec.strip().lower()
    if s in _FP_PRESETS:
        return _FP_PRESETS[s]["exp"], _FP_PRESETS[s]["man"]
    if s == "fp32":
        return 8, 23
    m = re.fullmatch(r"fp\d+:(e(\d+)m(\d+))", s)
    if m:
        return int(m.group(2)), int(m.group(3))
    raise ValueError(f"Unrecognized floating-point spec: {spec}")

def trunc_product_mantissa(x: Tensor, frac_bits: int) -> Tensor:
    x = x.to(torch.float32)
    is_finite = torch.isfinite(x)
    ax = x.abs()
    out = x.clone()
    nz = (ax != 0) & is_finite
    if not torch.any(nz):
        return out
    xn = x[nz]
    m, e = torch.frexp(xn)
    sign = torch.sign(m)
    m = m.abs()
    m2 = m * 2.0
    e2 = e - 1
    frac = m2 - 1.0
    scale = float(1 << frac_bits)
    frac_q = torch.floor(frac * scale) / scale
    m2_q = 1.0 + frac_q
    out_nz = (sign * m2_q) * torch.ldexp(torch.ones_like(m2_q), e2)
    out_nz = torch.where(out_nz.abs() < torch.finfo(torch.float32).tiny,
                         torch.zeros_like(out_nz), out_nz)
    out[nz] = out_nz
    return out

def q_bf16_rne(x: Tensor) -> Tensor:
    return x.to(torch.bfloat16).to(torch.float32)

def _fp_emax(exp: int, man: int) -> int:
    """Unbiased maximum exponent, accounting for formats with no Inf/NaN.

    Standard IEEE:  all-ones biased_exp reserved → emax = bias
    FP8 E4M3 (MX): biased_exp=15 mostly valid (only mant=7 is NaN) → emax = bias+1
    FP4 E2M1 (MX): no Inf/NaN at all, all-ones biased_exp is normal → emax = bias+2
    General no-inf/nan rule: emax = (1<<exp) - 1 - bias
    """
    bias = (1 << (exp - 1)) - 1
    is_mx_fp8  = (exp == 4 and man == 3)   # FP8 E4M3: one NaN code, otherwise valid
    is_no_special = (exp == 2 and man == 1) # FP4 E2M1: no Inf/NaN at all
    if is_mx_fp8:
        return bias + 1
    if is_no_special:
        return (1 << exp) - 1 - bias       # = 3 - 1 = 2 for E2M1
    return bias

def float_quantize_trunc(x: Tensor, exp: int, man: int) -> Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    out = torch.zeros_like(x)
    if torch.all(x == 0):
        return out
    sign = torch.sign(x)
    ax = x.abs()
    nz_mask = ax > 0
    ax_nz = ax[nz_mask]
    if ax_nz.numel() == 0:
        return out
    log2_ax = torch.log2(ax_nz)
    E = torch.floor(log2_ax)
    bias = (1 << (exp - 1)) - 1
    emin = 1 - bias
    is_mx_fp8 = (exp == 4 and man == 3)
    emax = _fp_emax(exp, man)
    # Subnormal range: 0 < ax < 2^emin (biased_exp=0, value = k * subnorm_delta)
    subnorm_delta = 2.0 ** (emin - man)   # ULP of subnormals = 2^(1-bias-man)
    subnorm_mask  = (E < emin)
    overflow_mask = (E > emax)
    normal_mask   = (~subnorm_mask) & (~overflow_mask)
    ax_q = torch.zeros_like(ax_nz)
    if subnorm_mask.any():
        # RTZ: truncate to nearest subnormal step toward zero
        k = torch.floor(ax_nz[subnorm_mask] / subnorm_delta)
        k = k.clamp(0, 2**man - 1)
        ax_q[subnorm_mask] = k * subnorm_delta
    if overflow_mask.any():
        E_max     = float(emax)
        base_max  = torch.pow(torch.tensor(2.0, dtype=ax_nz.dtype, device=ax_nz.device), E_max)
        delta_max = base_max / (2 ** man)
        max_mant  = (2**man - 2) if is_mx_fp8 else (2**man - 1)
        ax_q[overflow_mask] = base_max + max_mant * delta_max
    if normal_mask.any():
        E_norm = E[normal_mask]
        x_norm = ax_nz[normal_mask]
        base  = torch.pow(torch.tensor(2.0, dtype=ax_nz.dtype, device=ax_nz.device), E_norm)
        delta = base / (2 ** man)
        t = (x_norm - base) / delta
        k = torch.floor(torch.clamp(t, 0, 2**man - 1 - 1e-7))
        ax_q[normal_mask] = base + k * delta
    out[nz_mask] = sign[nz_mask] * ax_q
    return out

def make_fp_quantizer(spec: str, rounding: str = "nearest"):
    s = spec.strip().lower()
    if s in ("", "none", "identity", "fp32"):
        return None
    e, m = parse_fp_spec(spec)
    rounding = rounding.lower()
    if rounding in ("nearest", "nearest_even", "stochastic"):
        from qtorch.quant import float_quantize
        mode = "nearest" if rounding in ("nearest", "nearest_even") else "stochastic"
        return lambda x: float_quantize(x, exp=e, man=m, rounding=mode)
    if rounding in ("zero", "toward_zero", "trunc", "rtz"):
        return lambda x: float_quantize_trunc(x, e, m)
    raise ValueError(f"Unsupported rounding mode: {rounding}")

def tensor_to_custom_fp_codes(t: Tensor, spec: str) -> Tuple[List[List[int]], int]:
    e_bits, m_bits = parse_fp_spec(spec)
    total_bits = 1 + e_bits + m_bits
    bias = (1 << (e_bits - 1)) - 1
    emin = 1 - bias
    is_mx_fp8 = (e_bits == 4 and m_bits == 3)
    emax = _fp_emax(e_bits, m_bits)
    subnorm_delta = 2.0 ** (emin - m_bits)  # ULP of subnormals
    arr = t.detach().cpu()
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    if arr.ndim != 2:
        raise ValueError("Expected 1D or 2D tensor for hex dump.")
    rows, cols = arr.shape
    out_codes: List[List[int]] = []
    for r in range(rows):
        row_codes: List[int] = []
        for c in range(cols):
            v = float(arr[r, c].item())
            if v == 0.0 or not math.isfinite(v):
                code = 0
            else:
                s = 1 if v < 0 else 0
                av = abs(v)
                E = math.floor(math.log2(av))
                if E < emin:
                    # Subnormal: biased_exp=0, mant = round(av / subnorm_delta)
                    mant = int(round(av / subnorm_delta))
                    mant = max(0, min(mant, 2**m_bits - 1))
                    # mant==0 is ±0; canonicalize to +0 (code=0), hw treats both as zero
                    code = 0 if mant == 0 else (s << (e_bits + m_bits)) | mant
                else:
                    if E > emax:
                        E_used = emax
                        mant = (2**m_bits - 2) if is_mx_fp8 else (2**m_bits - 1)
                    else:
                        E_used = E
                        base  = 2.0 ** E_used
                        delta = base / (2 ** m_bits)
                        tpos  = (av - base) / delta
                        mant  = int(round(tpos))
                        if mant >= 2**m_bits:
                            E_used += 1
                            mant = 0
                            if E_used > emax:
                                E_used = emax
                                mant = (2**m_bits - 2) if is_mx_fp8 else (2**m_bits - 1)
                        else:
                            max_mant = (2**m_bits - 2) if (is_mx_fp8 and E_used == emax) else (2**m_bits - 1)
                            mant = max(0, min(mant, max_mant))
                    exp_bits_val = int(E_used + bias)
                    code = ((s & 0x1) << (e_bits + m_bits)) | \
                           ((exp_bits_val & ((1 << e_bits) - 1)) << m_bits) | \
                           (mant & ((1 << m_bits) - 1))
            row_codes.append(code)
        out_codes.append(row_codes)
    return out_codes, total_bits

def codes_to_hex_rows(codes: List[List[int]], total_bits: int) -> List[List[str]]:
    width = (total_bits + 3) // 4
    return [[f"{code:0{width}x}" for code in row] for row in codes]

def prod_quant(x: Tensor) -> Tensor:
    return trunc_product_mantissa(x, frac_bits=PROD_MANT_BITS)

def _rne_e8(x: Tensor, man_bits: int) -> Tensor:
    """RNE-round float32 mantissa to man_bits bits, preserving float32 exponent range."""
    if man_bits >= 23:
        return x
    drop = 23 - man_bits
    xi   = x.view(torch.int32)
    mag  = xi & 0x7FFFFFFF
    sign = xi ^ mag
    keep_mask = ~((1 << drop) - 1)
    round_bit = ((mag >> (drop - 1)) & 1).bool()
    sticky    = (mag & ((1 << (drop - 1)) - 1)).ne(0)
    lsb       = ((mag >> drop) & 1).bool()
    round_up  = round_bit & (sticky | lsb)
    mag_r = (mag & keep_mask) + round_up.int() * (1 << drop)
    return (sign | mag_r).view(torch.float32)

def _round_div_pow2_rne_int(n: int, shift: int) -> int:
    if shift <= 0:
        return n << (-shift)
    q = n >> shift
    rem = n & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if rem > half or (rem == half and (q & 1)):
        q += 1
    return q

def _encode_exact_scalar_to_fields(x: float, exp_bits: int, man_bits: int) -> Tuple[str, int, int, int]:
    sign = 1 if math.copysign(1.0, x) < 0 else 0
    if math.isnan(x):
        return "nan", sign, 0, 0
    if math.isinf(x):
        return "inf", sign, 0, 0
    if x == 0.0:
        return "zero", sign, 0, 0
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    ax = abs(x)
    m, e = math.frexp(ax)
    E = e - 1
    if E >= emin:
        total_sig = int(round(math.ldexp(ax, man_bits - E)))
        if total_sig == (1 << (man_bits + 1)):
            total_sig >>= 1
            E += 1
        return "finite", sign, E + bias, total_sig - (1 << man_bits)
    frac = int(round(math.ldexp(ax, man_bits - emin)))
    if frac >= (1 << man_bits):
        return "finite", sign, 1, 0
    return "finite", sign, 0, frac

def _fields_to_dyadic(sign: int, exp_field: int, frac: int, exp_bits: int, man_bits: int) -> Tuple[int, int, int]:
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    if exp_field == 0:
        return sign, frac, emin - man_bits
    return sign, (1 << man_bits) + frac, (exp_field - bias) - man_bits

def _round_dyadic_to_scalar(sign: bool, num: int, exp2: int, exp_bits: int, man_bits: int) -> float:
    if num == 0:
        return 0.0
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    emax = bias
    p = num.bit_length() - 1
    E = exp2 + p
    if E < emin:
        shift = (emin - man_bits) - exp2
        sub_sig = _round_div_pow2_rne_int(num, shift)
        if sub_sig == 0:
            return 0.0
        if sub_sig >= (1 << man_bits):
            E = emin
            total_sig = sub_sig
            if total_sig >= (1 << (man_bits + 1)):
                total_sig >>= 1
                E += 1
            if E > emax:
                return float("-inf") if sign else float("inf")
            val = math.ldexp(float(total_sig), E - man_bits)
            return -val if sign else val
        val = math.ldexp(float(sub_sig), emin - man_bits)
        return -val if sign else val
    total_sig = _round_div_pow2_rne_int(num, p - man_bits)
    if total_sig >= (1 << (man_bits + 1)):
        total_sig >>= 1
        E += 1
    if E > emax:
        return float("-inf") if sign else float("inf")
    val = math.ldexp(float(total_sig), E - man_bits)
    return -val if sign else val

def _scalar_to_dyadic(x: float) -> Tuple[bool, int, int]:
    ax = abs(x)
    num, den = ax.as_integer_ratio()
    exp2 = -(den.bit_length() - 1)
    return (math.copysign(1.0, x) < 0), num, exp2

def fp_quantize_rne_scalar(x: float, exp_bits: int, man_bits: int) -> float:
    if math.isnan(x):
        return float("nan")
    if math.isinf(x):
        return x
    if x == 0.0:
        return x
    sign, num, exp2 = _scalar_to_dyadic(x)
    return _round_dyadic_to_scalar(sign, num, exp2, exp_bits, man_bits)

def fp_quantize_rne(x: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    """Unified FP quantizer with round-to-nearest-even."""
    x = x.float()
    if exp_bits == 8 and man_bits == 7:
        return x.to(torch.bfloat16).to(torch.float32)
    if exp_bits == 8:
        return _rne_e8(x, man_bits)
    flat = x.detach().cpu().reshape(-1).tolist()
    out_flat = [fp_quantize_rne_scalar(float(v), exp_bits, man_bits) for v in flat]
    return torch.tensor(out_flat, dtype=torch.float32, device=x.device).view_as(x)

def fp_add_exact_scalar(x: float, y: float, exp_bits: int, man_bits: int) -> float:
    if math.isnan(x) or math.isnan(y):
        return float("nan")
    sign_x = 1 if math.copysign(1.0, x) < 0 else 0
    sign_y = 1 if math.copysign(1.0, y) < 0 else 0
    if math.isinf(x) or math.isinf(y):
        if math.isinf(x) and math.isinf(y) and sign_x != sign_y:
            return float("nan")
        return x if math.isinf(x) else y
    kind_x, sign_x, exp_x, frac_x = _encode_exact_scalar_to_fields(x, exp_bits, man_bits)
    kind_y, sign_y, exp_y, frac_y = _encode_exact_scalar_to_fields(y, exp_bits, man_bits)
    if kind_x == "zero":
        return y
    if kind_y == "zero":
        return x
    sign_x, sig_x, e_x = _fields_to_dyadic(sign_x, exp_x, frac_x, exp_bits, man_bits)
    sign_y, sig_y, e_y = _fields_to_dyadic(sign_y, exp_y, frac_y, exp_bits, man_bits)
    e_min = min(e_x, e_y)
    lhs = sig_x << (e_x - e_min)
    rhs = sig_y << (e_y - e_min)
    total = (-lhs if sign_x else lhs) + (-rhs if sign_y else rhs)
    if total == 0:
        return 0.0
    return _round_dyadic_to_scalar(total < 0, abs(total), e_min, exp_bits, man_bits)

def fp_add_exact(x: Tensor, y: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    assert x.shape == y.shape
    x_flat = x.detach().cpu().reshape(-1).tolist()
    y_flat = y.detach().cpu().reshape(-1).tolist()
    out_flat = [fp_add_exact_scalar(float(a), float(b), exp_bits, man_bits)
                for a, b in zip(x_flat, y_flat)]
    return torch.tensor(out_flat, dtype=torch.float32, device=x.device).view_as(x)

def mx_product_saturate(x: Tensor, exp_bits: int, man_bits: int) -> Tensor:
    """Clamp product overflow to match MxPEOutToRaw saturation."""
    x = x.float()
    bias = (1 << (exp_bits - 1)) - 1
    is_mx_fp8 = (exp_bits == 4 and man_bits == 3)
    emax = bias + 1 if is_mx_fp8 else bias
    scale = float(2 ** man_bits)
    max_mant = (2 ** man_bits - 2) if is_mx_fp8 else (2 ** man_bits - 1)
    max_normal = (2.0 ** emax) * (1.0 + max_mant / scale)
    sat_man = 2 ** man_bits - 2
    sat_val = (1.0 + sat_man / scale) * (2.0 ** (bias + 1))
    sign = torch.sign(x)
    ax = x.abs()
    return sign * torch.where(ax > max_normal, torch.full_like(ax, sat_val), ax)

def mx_product_quantize_trunc(x: Tensor, exp_bits: int, frac_bits: int) -> Tensor:
    """Match the MxFPMul product path before the accumulator resize."""
    x = x.float()
    out = x.clone()
    finite = torch.isfinite(x)
    ax = x.abs()
    nz = finite & (ax != 0)
    if not nz.any():
        return mx_product_saturate(out, exp_bits, frac_bits)
    ax_nz = ax[nz]
    m, e = torch.frexp(ax_nz)
    E = e - 1
    scale = float(1 << frac_bits)
    frac = 2.0 * m - 1.0
    frac_q = torch.floor(frac * scale) / scale
    val_nz = (1.0 + frac_q) * torch.ldexp(torch.ones_like(frac_q), E)
    out[nz] = torch.sign(x[nz]) * val_nz
    out[finite & (ax == 0)] = x[finite & (ax == 0)]
    return mx_product_saturate(out, exp_bits, frac_bits)

def matmul_outer_quantized_hwlike(
    A_in: Tensor,
    B_in: Tensor,
    prod_precision_list: Optional[List[Tuple[int, int]]] = None,
    acc_precision_list: Optional[List[Tuple[int, int]]] = None,
) -> Tensor:
    M, K = A_in.shape
    K2, N = B_in.shape
    assert K == K2
    if prod_precision_list is not None:
        assert len(prod_precision_list) == TILE_K
    if acc_precision_list is not None:
        assert len(acc_precision_list) == TILE_K
    C = torch.zeros((M, N), dtype=torch.float32, device=A_in.device)
    for k in range(K):
        k_idx = k % TILE_K
        outer = torch.outer(A_in[:, k], B_in[k, :])

        if prod_precision_list is not None:
            exp_p, man_p = prod_precision_list[k_idx]
            outer = mx_product_quantize_trunc(outer, exp_p, man_p)
        else:
            outer = prod_quant(outer)
            outer = q_bf16_rne(outer)

        if acc_precision_list is not None:
            exp_a, man_a = acc_precision_list[k_idx]
            C = fp_add_exact(
                fp_quantize_rne(C, exp_a, man_a),
                fp_quantize_rne(outer, exp_a, man_a),
                exp_a, man_a)
        else:
            C = q_bf16_rne(q_bf16_rne(C) + q_bf16_rne(outer))
    return C

def compute_tile_scale_matrix(A_scales_row, B_scales_col, m0, n0, k0, TM, TN):
    g  = k0 // GROUP
    sA = A_scales_row[m0:m0+TM, g].to(torch.float32)
    sB = B_scales_col[g, n0:n0+TN].to(torch.float32)
    S  = torch.outer(sA, sB)
    return make_fp_quantizer(SCALE_SPEC, "nearest")(S)

def bf16_accum_add(x: Tensor, y: Tensor) -> Tensor:
    return fp_add_exact(q_bf16_rne(x), q_bf16_rne(y), 8, 7)

def print_matrix(name: str, mat: Tensor, spec: str = "bf16"):
    arr = mat.detach().cpu()
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    print(f"\n  {name} (decimal):")
    for r in range(arr.shape[0]):
        vals = [f"{arr[r,c].item():12.6f}" for c in range(arr.shape[1])]
        print("    " + " ".join(vals))
    codes, bits = tensor_to_custom_fp_codes(arr, spec)
    hex_rows = codes_to_hex_rows(codes, bits)
    print(f"  {name} (hex, {spec}):")
    for row in hex_rows:
        print("    " + " ".join(row))

def tiled_matmul_hwlike(
    A_in: Tensor,
    B_in: Tensor,
    A_scales_row: Tensor,
    B_scales_col: Tensor,
    verbose: bool = True,
    prod_precision_list: Optional[List[Tuple[int, int]]] = None,
    acc_precision_list: Optional[List[Tuple[int, int]]] = None,
) -> Tensor:
    M, K = A_in.shape
    K2, N = B_in.shape
    assert K == K2

    TM, TN, TK = TILE_M, TILE_N, TILE_K
    Gk = (K + GROUP - 1) // GROUP
    assert A_scales_row.shape == (M, Gk)
    assert B_scales_col.shape == (Gk, N)

    C_out = torch.zeros((M, N), dtype=torch.float32, device=A_in.device)
    tile_count = 0

    for k0 in range(0, K, TK):
        for n0 in range(0, N, TN):
            for m0 in range(0, M, TM):
                tile_count += 1
                A_tile = A_in[m0:m0+TM, k0:k0+TK]
                B_tile = B_in[k0:k0+TK, n0:n0+TN]

                if verbose:
                    print(f"\n{'='*60}")
                    print(f"TILE {tile_count}: m0={m0}, n0={n0}, k0={k0} (group={k0//GROUP})")
                    print(f"{'='*60}")
                    print_matrix("A_tile", A_tile, INPUT_SPEC)
                    print_matrix("B_tile", B_tile, INPUT_SPEC)

                C_tile = matmul_outer_quantized_hwlike(A_tile, B_tile,
                    prod_precision_list=prod_precision_list,
                    acc_precision_list=acc_precision_list)

                if verbose:
                    print_matrix("C_tile (pre-scale, bf16)", C_tile, "bf16")

                S_tile = compute_tile_scale_matrix(A_scales_row, B_scales_col, m0, n0, k0, TM, TN)

                if verbose:
                    print_matrix("S_tile (scale factors)", S_tile, "bf16")

                C_tile_scaled = q_bf16_rne(C_tile * S_tile)

                if verbose:
                    print_matrix("C_tile_scaled (post-scale, bf16)", C_tile_scaled, "bf16")

                C_out[m0:m0+TM, n0:n0+TN] = bf16_accum_add(
                    C_out[m0:m0+TM, n0:n0+TN], C_tile_scaled
                )

                if verbose:
                    print_matrix("C_accumulated (running sum, bf16)",
                                 C_out[m0:m0+TM, n0:n0+TN], "bf16")

    if verbose:
        print(f"\n{'='*60}")
        print("FINAL OUTPUT C_out (bf16)")
        print(f"{'='*60}")
        print_matrix("C_out", C_out, "bf16")

    return C_out

def _e3m1_to_e2m1(x: Tensor) -> Tensor:
    """Deterministic E3M1 → E2M1 float mapping, matching E3M1Tofp4 hardware."""
    sign = x.sign()
    ax   = x.abs()

    mapToZero    = (ax == 0.0) | (ax == 0.125) | (ax == 0.25)
    mapToSubnorm = (ax == 0.375) | (ax == 0.5)
    mapToMinNorm = (ax == 0.75)
    mapToMax     = (ax >= 8.0) | ~torch.isfinite(ax)  # exp>5 or Inf/NaN

    out = x.clone()
    out = torch.where(mapToZero,    torch.zeros_like(x), out)
    out = torch.where(mapToMax,     sign * 6.0,          out)  # checked before suborm/minNorm
    out = torch.where(mapToSubnorm, sign * 0.5,          out)
    out = torch.where(mapToMinNorm, sign * 1.0,          out)
    return out

def _bf16_to_e3m1_rne(x: Tensor) -> Tensor:
    """Round BF16 tensor to E3M1 (1+3+1, bias=3) using RNE.

    Bit-accurate match to hardfloat RoundAnyRawFNToRecFN(8,8,3,2,options=0)
    with round_near_even / tininess_afterRounding.

    Works directly on BF16 bit patterns, handling all boundary cases:
      e ∈ [-2,3]  → E3M1 normal: keep 1 mantissa bit, RNE on remaining 6 bits
      e = -3      → BF16 in [0.125, 0.25):  M≥64 → 0.25, M<64 → 0.125
      e = -4      → BF16 in [0.0625, 0.125): M>0  → 0.125, M=0 → 0
      e ≤ -5      → 0
      BF16 subnormal (E=0) → 0
      E=255       → E3M1 Inf / NaN
    """
    bf16 = x.to(torch.bfloat16)
    bits = bf16.view(torch.int16).to(torch.int32) & 0xFFFF

    sign_bit = (bits >> 15) & 1
    E = (bits >> 7) & 0xFF   # 8-bit biased BF16 exponent
    M = bits & 0x7F           # 7-bit fractional BF16 mantissa

    sign_f = torch.where(sign_bit == 0, torch.ones_like(x), -torch.ones_like(x))
    e = E - 127  # unbiased exponent (valid for normal BF16, E in [1,254])

    # ── Normal BF16 → E3M1 normal (general formula, valid for e ∈ [-2, 3]) ──
    # Full significand: S = {1, M[6], ..., M[0]} = 128 + M  (8-bit integer)
    # Round to 2-bit significand {1, q}:
    #   kept bit q = S[6] = M[6]
    #   round bit r = S[5] = M[5]
    #   sticky    s = S[4:0] = M[4:0] != 0
    # RNE: round_up = r & (s | q)
    S = 128 + M
    q = (S >> 6) & 1
    r = (S >> 5) & 1
    sticky = (S & 0x1F).bool()
    round_up = r.bool() & (sticky | q.bool())

    sig2 = q + round_up.to(torch.int32)   # 0 or 1; becomes 2 on carry
    carry = sig2 >= 2
    mant_out = torch.where(carry, torch.zeros_like(sig2), sig2)
    exp_out  = torch.where(carry, e + 1, e)            # unbiased E3M1 exp

    is_overflow = exp_out > 3
    safe_exp = exp_out.float().clamp(-10, 10)
    val_normal = sign_f * (1.0 + mant_out.float() * 0.5) * torch.pow(
        torch.full_like(x, 2.0), safe_exp)
    val_normal = torch.where(is_overflow, sign_f * float('inf'), val_normal)

    # ── Special boundary cases below E3M1 emin=-2 ────────────────────────────
    # e = -3: BF16 in [0.125, 0.25)
    #   midpoint 0.1875 = (128+64)/1024 → M=64
    #   M≥64: round to 0.25 (E3M1 normal emin, even code=2), M<64: 0.125 (subnorm)
    is_e_neg3 = (e == -3) & (E >= 1)
    val_e_neg3 = sign_f * torch.where(
        M >= 64, torch.full_like(x, 0.25), torch.full_like(x, 0.125))

    # e = -4: BF16 in [0.0625, 0.125)
    #   midpoint 0.0625 = 128/2048 → M=0; at tie, 0 (code=0 even) wins over 0.125 (code=1 odd)
    is_e_neg4 = (e == -4) & (E >= 1)
    val_e_neg4 = sign_f * torch.where(
        M > 0, torch.full_like(x, 0.125), torch.zeros_like(x))

    # ── Assemble ─────────────────────────────────────────────────────────────
    result = val_normal
    result = torch.where(is_e_neg3, val_e_neg3, result)
    result = torch.where(is_e_neg4, val_e_neg4, result)
    result = torch.where(((e <= -5) & (E >= 1)) | (E == 0),
                         torch.zeros_like(x), result)

    # BF16 Inf / NaN → E3M1 Inf / NaN
    is_nan = (E == 255) & (M != 0)
    is_inf = (E == 255) & (M == 0)
    result = torch.where(is_inf, sign_f * float('inf'), result)
    result = torch.where(is_nan, torch.full_like(x, float('nan')), result)

    return result

def hw_bf16_to_e2m1(x: Tensor) -> Tensor:
    """Hardware-accurate BF16 → E2M1 matching BF16ScaleRoundToTiny (fp4 path).

    1. Manual RNE round of BF16 bit pattern to E3M1 (1+3+1, bias=3).
    2. Apply deterministic E3M1Tofp4 map to obtain the E2M1 float value.
    """
    e3m1 = _bf16_to_e3m1_rne(x)
    return _e3m1_to_e2m1(e3m1)

def matrix_mx_requantize(matrix: Tensor, quant_spec: str = INPUT_SPEC):
    M, N = matrix.shape
    nblocks = N // GROUP
    e_bits, m_bits = parse_fp_spec(quant_spec)
    # Requant block-scale floor: p=0 for all formats -> output normalizes to [1,2) (chained-matmul
    # fix). Matches the RTL/spike log2_pmax=0. (Previously subtracted emax=_fp_emax(e_bits,m_bits).)
    log2_pmax = 0

    scale_q_fn = make_fp_quantizer(SCALE_SPEC, "nearest")
    C_quantized = torch.zeros_like(matrix)
    C_scales    = torch.zeros(M, nblocks)

    for bi in range(nblocks):
        block     = matrix[:, bi*GROUP:(bi+1)*GROUP]
        block_max = block.abs().amax(dim=1, keepdim=True)
        max_exp   = torch.floor(torch.log2(block_max.clamp(min=1e-45)))
        scale_exp = max_exp - log2_pmax
        scale     = torch.pow(2.0, scale_exp)
        scale     = scale_q_fn(scale)
        # Scale is a power-of-2 → division is an exact BF16 exponent shift.
        # Apply the hardware two-stage quantization: RNE→E3M1, then E3M1→E2M1.
        scaled = q_bf16_rne(block / scale)
        C_quantized[:, bi*GROUP:(bi+1)*GROUP] = hw_bf16_to_e2m1(scaled)
        C_scales[:, bi] = scale.squeeze(1)

    return C_quantized, C_scales
