"""Emit a whole kernel GRAPH as one driver — mesh matmuls and host ops in a single ELF.

``mxgemm_emit`` lowers a straight CHAIN and does it well: the intermediates never leave the
scratchpad. This module handles the kernels that are not a chain — attention, and anything with a
host op in the middle — and it is deliberately a SEPARATE emitter rather than a generalization of
that one, so the chain path stays byte-identical and keeps its bit-exact gates.

The structure is transcribed from ``bareMetalC/llama_attention.c``, which is this exact shape in one
ELF: a handful of reusable C helpers, then a linear sequence of calls, one per graph step. Every
edge passes through host fp32 memory — drained as bf16, converted, re-quantized before its next use.
For attention that is what the hardware requires, since every seam but ``P@V -> O@Wo`` carries a
host op and those values must reach the scalar core regardless.

What it buys is D4: one ELF, one run, and numpy never sitting between two stages. It does not buy
speed; the scalar glue dominates (``llama_layer_hw_plan.md`` §8.3).
"""
from __future__ import annotations

from typing import Any, Sequence

from app import mxformats as _fmt

from .mxgemm_emit import (BF16_PER_WORD, DEFAULT_GEOMETRY, MxEmitError, _c_array_2d)


def _helpers(dim: int, spad_rows: int) -> list[str]:
    """The reusable C the sequence calls. One copy, not inlined per step."""
    return [
        f"#define DIM {dim}",
        f"#define SPAD_ROWS {spad_rows}",
        "",
        "/* MVIN A: tile (i,k) -> a_spad + (i*tiles_K + k)*DIM. Row stride is K BYTES, which for a",
        "   4-bit format is still K -- packing halves the row COUNT, not the row length. */",
        "static void mvin_A(const uint8_t *A, int M, int K, int a_row_bytes, uint32_t a_spad) {",
        "  gemmini_config_ld(a_row_bytes * sizeof(uint8_t));",
        "  for (int i = 0; i < M / DIM; i++)",
        "    for (int k = 0; k < K / DIM; k++)",
        "      gemmini_extended_mvin((void *)(A + (size_t)i * DIM * a_row_bytes + (size_t)k * DIM),",
        "                            a_spad + (i * (K / DIM) + k) * DIM, DIM, DIM);",
        "}",
        "",
        "/* MVIN B: tile (k,j) -> b_spad + (k*tiles_J + j)*DIM. Slot uses tiles_J, NOT tiles_K --",
        "   gemmini.cc's B_t = B_sp + (k_outer*TJ + j)*DIM. The two coincide only when N == K.",
        "   `n0`/`b_row_bytes` select a COLUMN SLICE of a wider B, for N-chunking. */",
        "static void mvin_B(const uint8_t *B, int K, int n0, int N, int b_row_bytes,",
        "                   uint32_t b_spad) {",
        "  gemmini_config_ld(b_row_bytes * sizeof(uint8_t));",
        "  for (int k = 0; k < K / DIM; k++)",
        "    for (int j = 0; j < N / DIM; j++)",
        "      gemmini_extended_mvin(",
        "          (void *)(B + (size_t)k * DIM * b_row_bytes + (size_t)(n0 + j * DIM)),",
        "          b_spad + (k * (N / DIM) + j) * DIM, DIM, DIM);",
        "}",
        "",
        "/* Copy a drained [M][N] chunk into columns [n0, n0+N) of a wider [M][N_full] buffer.",
        "   Done on the host rather than by a strided mvout: a strided de-tiling mvout makes the",
        "   writer DMA emit whole 64-byte cache lines and zero-fill the gaps on RTL. */",
        "static void paste_cols(uint16_t *dst, int N_full, const uint16_t *src, int M, int n0,",
        "                       int N) {",
        "  for (int m = 0; m < M; m++)",
        "    for (int j = 0; j < N; j++) dst[(size_t)m * N_full + n0 + j] = src[(size_t)m * N + j];",
        "}",
        "",
        "/* Drain a BF16 result: flat contiguous spad->DRAM mvout, the sequence every reference test",
        "   uses (identical instruction stream on Spike and RTL). 2 bytes/elem -> M*N*2/DIM rows. */",
        "static void mvout_bf16(uint16_t *dst, uint32_t spad, int M, int N) {",
        "  gemmini_fence();",
        "  gemmini_config_st(DIM * sizeof(uint8_t));",
        "  uint8_t *b = (uint8_t *)dst;",
        "  for (int r = 0; r < M * N * 2 / DIM; r += DIM)",
        "    gemmini_extended_mvout(b + (size_t)r * DIM, spad + r, DIM, DIM);",
        "  gemmini_fence();",
        "}",
        "",
        "/* One mesh matmul, BF16 out. The SPAD_AB command is what marks the following LOOP_WS as",
        "   the MX variant; flag 0x38 keeps the accumulator->spad store the drain reads back. */",
        "static void mesh_matmul(int M, int K, int N, int tile_m, int tile_n,",
        "                        uint32_t a_spad, uint32_t b_arg, uint32_t c_spad,",
        "                        uint32_t scale_dram) {",
        "  const int I = M / tile_m, J = N / tile_n, Kt = K / DIM;",
        "  gemmini_config_st(DIM * sizeof(uint8_t));",
        "  gemmini_mxquant_config_mvout(scale_dram, I, J, Kt, 0, 0, 1);",
        "  gemmini_loop_ws_spad(I, J, Kt, 0, 0, 0, a_spad, b_arg, 0, c_spad,",
        "                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);",
        "  gemmini_fence();",
        "}",
    ]


#: RMSNorm over an fp32 tile. `mx_host.h`'s `mx_rmsnorm` takes bf16 inputs, which is right for the
#: reference kernel's residual stream but not for a graph where every edge is already fp32. Same
#: arithmetic, same order of operations, so `app.mxhost.rmsnorm` remains its twin.
_ROPE_F32 = """\
/* RoPE with fp32 cos/sin tables. mx_host.h's mx_rope takes them as uint32 bit patterns and puns
   through a union; the emitter bakes real floats, so this variant takes them directly. Same
   arithmetic and same indexing, so app.mxhost.rope stays its twin. */
static void mx_rope_f32(const uint16_t *x_bf16, const float *cosv, const float *sinv,
                        int M, int H, float *out) {
  const int half = H / 2;
  for (int m = 0; m < M; m++)
    for (int i = 0; i < H; i++) {
      size_t o = (size_t) m * H + i;
      float x = mx_bf16_to_f32(x_bf16[o]);
      float rot = (i < half) ? -mx_bf16_to_f32(x_bf16[o + half])
                             :  mx_bf16_to_f32(x_bf16[o - half]);
      out[o] = x * cosv[o] + rot * sinv[o];
    }
}"""

_RMSNORM_F32 = """\
static void mx_rmsnorm_f32(const float *x, const float *w, int M, int D, float eps, float *out) {
  for (int m = 0; m < M; m++) {
    const float *row = x + (size_t) m * D;
    float *dst = out + (size_t) m * D;
    float acc = 0.0f;
    for (int d = 0; d < D; d++) acc += row[d] * row[d];
    const float inv = 1.0f / sqrtf(acc / (float) D + eps);
    for (int d = 0; d < D; d++) dst[d] = row[d] * inv * w[d];
  }
}"""


def _cfloat(v: float) -> str:
    """A valid C float literal. `%.9g` alone yields `0` for 0.0, and `0f` is an integer with an
    invalid suffix -- which the compiler reports 4000 lines into a generated file."""
    t = f"{float(v):.9g}"
    if not any(c in t for c in ".eEni"):
        t += ".0"
    return t + "f"


def _bf16_to_f32(name: str, m: int, n: int) -> list[str]:
    return [f"  for (long i = 0; i < {m} * {n}; i++)",
            f"    {name}_f32[i] = mx_bf16_to_f32(((const uint16_t *){name}_bf16)[i]);"]


def generate_graph_driver(cb: dict[str, Any], *, dtype: str = "fp8_e4m3") -> str:
    """Emit the one-ELF driver for a graph-shaped kernel.

    ``cb["graph"]`` carries the step list (see :mod:`app.mxgraph`) and ``cb["graph_operands"]`` the
    quantized leaves. The graph is a side channel because ``merlin_iface`` v0.1 cannot express it —
    three live values, computed B operands, and a softmax it has no op for.
    """
    graph = cb.get("graph")
    ops = cb.get("graph_operands")
    if not graph or ops is None:
        raise MxEmitError("command buffer carries no 'graph'/'graph_operands' side channel")

    f = _fmt.get(dtype, where="graph emission")
    geom = dict(DEFAULT_GEOMETRY)
    geom.update({k: v for k, v in (cb.get("params") or {}).items() if k in DEFAULT_GEOMETRY})
    dim, spad_rows = geom["dim"], geom["bank_num"] * geom["bank_rows"]
    if f.lut:
        raise MxEmitError(
            f"graph kernels are emitted for direct formats only; {dtype} is codebook-indexed and "
            "its per-group books would have to be rebuilt for every computed operand on device")

    steps, shapes, uses = graph["steps"], graph["shapes"], graph["uses"]
    consts = cb.get("graph_consts") or {}
    leaves = set(graph["leaves"])
    result = graph["result"]

    head = ["/* Generated by mxgraph_emit.py for target mx_gemmini_rocket — do not edit. */",
            f"/* GRAPH of {len(steps)} step(s) fused into one program   operands:{dtype} */"]
    for st in steps:
        head.append(f"/*   {st['kind']:4s} {st['name']:>4s}"
                    + (f"  {st['m']}x{st['k']}x{st['n']}  {st['lhs']} @ {st['rhs']}"
                       f"{'.T' if st['rhs_transposed'] else ''}" if st["kind"] == "mesh"
                       else f"  {st['op']}({', '.join(st['srcs'])})  "
                            f"[{st['m']}x{st['n']}]") + " */")

    # --- baked leaves ---------------------------------------------------------------------
    data: list[str] = []
    for key, bundle in sorted(ops.items()):
        name, how = key.split(":")
        c = bundle["codes"]
        rows, cols = len(c), len(c[0])
        tag = f"{name}_{how.replace('.', '')}"
        data.append(_c_array_2d("uint8_t", f"{tag}_codes", c, f"[{rows}][{cols}]"))
        s = bundle["scales"]
        data.append(_c_array_2d("uint8_t", f"{tag}_scales", s, f"[{len(s)}][{len(s[0])}]"))

    # fp32 constants the host ops read: RMSNorm's weight, a residual's input. Flattened whatever
    # their rank -- the emitted C indexes them linearly, as the ops do.
    import numpy as _np

    for cname, arr in sorted(consts.items()):
        flat = _np.ascontiguousarray(arr, dtype=_np.float32).ravel()
        body = ", ".join(_cfloat(v) for v in flat.tolist())
        data.append(f"static const float {cname}_f32[{flat.size}] = {{{body}}};")

    # --- buffers, one set per computed value ------------------------------------------------
    produced = [st["out"] for st in steps]
    buf: list[str] = []
    for v in produced:
        r, c = shapes[v]
        buf.append(f"  static uint16_t {v}_bf16[{r} * {c}];" if any(
            st["kind"] == "mesh" and st["out"] == v for st in steps) else "")
        buf.append(f"  static float {v}_f32[{r} * {c}];")
        for how in sorted(uses.get(v, [])):
            tag = f"{v}_{how.replace('.', '')}"
            rr, cc = (c, r) if how == "b.T" else (r, c)
            buf.append(f"  static uint8_t {tag}_codes[{rr} * {cc}];")
            buf.append(f"  static uint8_t {tag}_scales[{(cc if how == 'a' else rr) // 32} * "
                       f"{rr if how == 'a' else cc}];")
        if "b.T" in uses.get(v, ()):
            buf.append(f"  static float {v}_T_f32[{c} * {r}];")
    buf = [b for b in buf if b]

    # --- the sequence ----------------------------------------------------------------------
    body: list[str] = []
    a_spad, spad_dest = 0, geom["spad_dest"]
    for st in steps:
        if st["kind"] == "host":
            body += _emit_host_step(st, uses)
            continue
        body += _emit_mesh_step({**st, "_uses": uses}, shapes, leaves, dim,
                                a_spad, spad_dest, spad_rows, f)
    body += ["", "  c1 = read_cycles();"]

    rm, rn = shapes[result]
    report = [
        "  /* merlin console protocol — parsed by runtime.backends.base.parse_console. */",
        f'  printf("OUT Y0 {rm} {rn}");',
        f"  for (long i = 0; i < {rm} * {rn}; i++)",
        f'    printf(" %u", (unsigned)((const uint16_t *){result}_bf16)[i]);',
        '  printf("\\n");',
        '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));',
        '  printf("METRIC cycle_window_mx_gemmini_region 1\\n");',
        '  printf("DONE\\n");',
    ]

    return "\n".join([
        *head,
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <string.h>",
        '#include "include/gemmini_testutils.h"',
        '#include "mx_host.h"',
        "",
        *_helpers(dim, spad_rows),
        "",
        _RMSNORM_F32,
        "",
        _ROPE_F32,
        "",
        *data,
        "",
        "int main(void) {",
        "  uint64_t c0, c1;",
        "  static uint32_t scale_factors[512];",
        *buf,
        "  memset(scale_factors, 0, sizeof(scale_factors));",
        "",
        "  mx_host_init();",
        "  gemmini_flush(0);",
        "",
        "  /* One accelerator region spanning EVERY step, mesh and host alike. Nothing returns to",
        "     the host between steps -- that is the whole point of fusing a graph. */",
        "  c0 = read_cycles();",
        "",
        *body,
        "",
        *report,
        "  return 0;",
        "}",
    ]) + "\n"


def _emit_host_step(st: dict, uses: dict) -> list[str]:
    """A host op, plus the re-quantization that hands its result back to the mesh."""
    from app import mxhost

    o = mxhost.get(st["op"], **st["params"])
    m, n, out = st["m"], st["n"], st["out"]
    srcs = list(st["srcs"])
    src = srcs[0]
    lines = [f"  /* ---- host: {out} = {st['op']}({', '.join(srcs)}) ---- */"]

    # `_src(v)` names the fp32 buffer for a value. A mesh output arrives as bf16 and its `_f32`
    # twin is filled by _emit_uses; a host output is already fp32; a LEAF is baked as fp32.
    def f32(v: str) -> str:
        return f"{v}_f32"

    if st["op"] == "softmax":
        scale = float(st["params"].get("scale", 1.0))
        causal = int(bool(st["params"].get("causal", False)))
        lines.append(f"  mx_softmax_rows({src}_bf16, {m}, {n}, {scale!r}f, {causal}, {out}_f32);")
    elif st["op"] == "rmsnorm":
        eps = float(st["params"].get("eps", 1e-5))
        w = st["const_names"]["weight"]
        lines.append(f"  mx_rmsnorm_f32({f32(src)}, {w}_f32, {m}, {n}, {eps!r}f, {out}_f32);")
    elif st["op"] == "rope":
        c, sn = st["const_names"]["cos"], st["const_names"]["sin"]
        lines.append(f"  mx_rope_f32({src}_bf16, {c}_f32, {sn}_f32, {m}, {n}, {out}_f32);")
    elif st["op"] == "swiglu":
        g_, u_ = srcs
        lines += [f"  for (long i = 0; i < {m} * {n}; i++)",
                  f"    {out}_f32[i] = mx_silu({f32(g_)}[i]) * {f32(u_)}[i];"]
    elif st["op"] == "add":
        a_, b_ = srcs
        lines += [f"  for (long i = 0; i < {m} * {n}; i++)",
                  f"    {out}_f32[i] = {f32(a_)}[i] + {f32(b_)}[i];"]
    else:
        raise MxEmitError(
            f"host op {st['op']!r} has a Python twin but no emitter here yet. Add it beside "
            f"softmax; the C entry point is {o.c_fn or '(none -- emit it inline)'}.")

    return lines + _emit_uses(out, m, n, uses, from_bf16=False) + [""]


def _emit_uses(name: str, m: int, n: int, uses: dict, *, from_bf16: bool) -> list[str]:
    """Hand a produced value back to the mesh, on whichever side(s) it is consumed.

    THE host seam, and unavoidable rather than lazy: the value is in host memory because either a
    host op produced it, or it is a mesh output some later matmul needs on the B side (where
    residency cannot help -- residency puts a result in the operand-A layout).

    A value used on BOTH sides is quantized twice, differently: A blocks along K by rows, B by
    columns. Getting that wrong is silent, so the side is part of every buffer's name.
    """
    how_set = sorted(uses.get(name, []))
    if not how_set:
        return []
    lines = []
    if from_bf16:
        lines += [f"  for (long i = 0; i < {m} * {n}; i++)",
                  f"    {name}_f32[i] = mx_bf16_to_f32({name}_bf16[i]);"]
    for how in how_set:
        tag = f"{name}_{how.replace('.', '')}"
        if how == "a":
            lines.append(f"  mx_quantize_rows({name}_f32, {m}, {n}, {tag}_codes, {tag}_scales);")
        elif how == "b":
            lines.append(f"  mx_quantize_cols({name}_f32, {m}, {n}, {tag}_codes, {tag}_scales);")
        else:
            # The MX loop path IGNORES B_transpose (mx_loop_ws_spad does `(void)rs1;`), so K^T is a
            # real byte transpose on the scalar core -- not a config bit.
            lines += [f"  mx_transpose_f32({name}_f32, {m}, {n}, {name}_T_f32);",
                      f"  mx_quantize_cols({name}_T_f32, {n}, {m}, {tag}_codes, {tag}_scales);"]
    return lines


def _emit_mesh_step(st: dict, shapes: dict, leaves: set, dim: int,
                    a_spad: int, spad_dest: int, spad_rows: int, f) -> list[str]:
    """One matmul: move both operands in, load both scale streams, compute, drain."""
    m, k, n = st["m"], st["k"], st["n"]
    lhs, rhs, out = st["lhs"], st["rhs"], st["out"]
    how_b = "bT" if st["rhs_transposed"] else "b"
    a_tag, b_tag = f"{lhs}_a", f"{rhs}_{how_b}"
    b_spad = spad_rows - (k // dim) * (n // dim) * dim
    lines = [
        f"  /* ---- mesh: {out} = {lhs} @ {rhs}{'.T' if st['rhs_transposed'] else ''} "
        f"({m}x{k}x{n}) ---- */",
        "  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0,",
        f"                              false, {f.fmt_code}, {f.fmt_code}, "
        f"{_fmt.BF16_FMT_CODE}, {int(f.lut)});",
    ]
    # A computed operand's codes live in a runtime buffer; a leaf's are baked const. Same names,
    # different storage, so the emitted calls are identical -- which is why the buffers above are
    # named with the same scheme as the baked arrays.
    # Scratchpad budget, in rows of DIM bytes: the A tiles, the B tiles and the BF16 output all
    # live there at once. `H @ Wd` at [32,64]x[64,2048] needs 16512 of 16384 -- 128 over -- so N is
    # split until it fits. The plan's section 3 table is this arithmetic.
    nc = n
    while nc >= f.tile_n:
        rows = (m * k) // dim + (k * nc) // dim + (m * nc * 2) // dim
        if rows <= spad_rows:
            break
        nc //= 2
    if nc < f.tile_n or n % nc:
        raise MxEmitError(
            f"{out}: {m}x{k}x{n} does not fit the {spad_rows}-row scratchpad even chunked "
            f"(smallest tried {nc}); A={m*k//dim} B={k*n//dim} out={m*n*2//dim} rows")
    chunks = n // nc
    lines += [
        f"  gemmini_mx_load_scales((uint64_t)&{a_tag}_scales, sizeof({a_tag}_scales), 0);",
        f"  mvin_A((const uint8_t *){a_tag}_codes, {m}, {k}, {k}, {a_spad});",
    ]
    if chunks > 1:
        b_spad = spad_rows - (k // dim) * (nc // dim) * dim
        lines += [
            f"  /* N-chunked x{chunks}: {m}x{k}x{n} needs "
            f"{(m*k)//dim + (k*n)//dim + (m*n*2)//dim} rows of {spad_rows}; each chunk needs "
            f"{(m*k)//dim + (k*nc)//dim + (m*nc*2)//dim}. */",
            f"  {{ static uint16_t chunk[{m} * {nc}];",
            f"    for (int n0 = 0; n0 < {n}; n0 += {nc}) {{",
            f"      gemmini_mx_load_scales((uint64_t)&{b_tag}_scales[0][n0], "
            f"{nc} * {k // 32}, 1);",
            "      gemmini_fence();",
            f"      mvin_B((const uint8_t *){b_tag}_codes, {k}, n0, {nc}, {n}, {b_spad});",
            f"      mesh_matmul({m}, {k}, {nc}, {f.tile_m}, {f.tile_n}, {a_spad}, {spad_rows}, "
            f"{spad_dest}, (uint64_t)scale_factors);",
            f"      mvout_bf16(chunk, {spad_dest}, {m}, {nc});",
            f"      paste_cols({out}_bf16, {n}, chunk, {m}, n0, {nc});",
            "    } }",
        ]
    else:
        lines += [
            f"  gemmini_mx_load_scales((uint64_t)&{b_tag}_scales, sizeof({b_tag}_scales), 1);",
            "  gemmini_fence();",
            f"  mvin_B((const uint8_t *){b_tag}_codes, {k}, 0, {n}, {n}, {b_spad});",
            f"  mesh_matmul({m}, {k}, {n}, {f.tile_m}, {f.tile_n}, {a_spad}, {spad_rows}, "
            f"{spad_dest}, (uint64_t)scale_factors);",
            f"  mvout_bf16({out}_bf16, {spad_dest}, {m}, {n});",
        ]
    return lines + _emit_uses(out, m, n, st["_uses"], from_bf16=True) + [""]
