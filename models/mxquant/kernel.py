"""The mxquant model, bit path: the bits MX-Gemmini must produce for one kernel, computed from the recipe on mxq.

VERDICT is ``spike bits == mxquant bits`` with no tolerance anywhere in it. The model walks the same
graph the device runs: every mesh stage goes through ``mxq.matmul.systolic`` with the recipe's
Arithmetic, schedule and window (``config/scheme.py``); host stages run their own fp32 function, as
they do on the Rocket core.

Operands come from mxq's block quantizer (``mxq.block.mxgemmini.quantize``, round-to-nearest-even,
block max floored at 2^-23), which is what the compiler's wire encoder calls too, so the model
multiplies the values the ELF carries. Measured 2026-09-28 (``tests/selftest_mxquant.py`` [5], [6]):
on the direct formats this equals the wire round trip element for element, and for an fp8_e4m3
chain ``quantize(bf16(C))`` IS the device requantizer (``gemmini.cc:1303-1378``), 0 differing on
10^6 values including ties, subnormals, zero and sub-2^-23 blocks.

The fp4_e2m1 chain requantizer rounds bf16 -> E3M1 -> E2M1 in two steps; that is mxq's
``quantize(via=(3, 1))`` (mxq 5fe2690), measured identical to the hardware team's model
(``rtl_exact/mxmesh/fp4.py`` ``matrix_mx_requantize``) on 3.4 M codes of finite blocks with max >= 2^-126.
Its scale floor is E8M0's smallest scale, 2^-126, not the fp8 requantizer's 2^-23 (``gemmini.cc``): the two
device models floor differently, a spike-vs-RTL question for Nicolas, invisible on every fixture so far.

One thing mxq does not have stays on the hardware team's code, in :func:`_device_operands` and
:func:`_device_requant` and nowhere else: the four CODEBOOK formats (fp8_e4m3_quad, fp8_e5m2, fp6_e3m2,
fp6_e2m3), whose wire carries 4-bit indices into a per-row-pair table built from the data
(``compiler/codebook.py``).

``edges`` (built by the lowering, ``compiler/lower.py``) says how each intermediate reached the mesh:
``{"via": "host"}`` -- through host memory, re-quantized there -- or ``{"via": "requant", "books"}`` --
the hardware requantizer wrote it and the next stage read it in place (fused chains only). An edge
the requantizer model cannot reproduce raises :class:`Unavailable`; the pipeline then grades on the
fp32 tier and says so. A confident wrong reference would be worse than none.

The informational "as shipped" number is MXQuant's published simulator on this recipe's ladder:
operands from ``mxq.block.mxquant`` (MXQuant's own codes, from the floats), arithmetic
``mxq.matmul.MXQUANT``. It is reported as the model-vs-silicon gap and never grades a run.
"""
from __future__ import annotations

import numpy as np
import torch

import models
from config import scheme as _scheme
from config.recipe import KERNEL_ROUNDING
from models.mxquant.block import BLOCK, SCALE_FLOOR

TIER = "mxquant_recipe_exact"
#: How a value reached the mesh. THE LOWERING DECIDES THIS (compiler/lower.py writes "host" / "requant"
#: into the edge map); these names are only read here.
VIA_HOST = "host"
VIA_REQUANT = "requant"
#: The fp4 requantizer's two roundings and scale floor (module docstring); every other direct format is one rounding.
_REQUANT_VIA = {"fp4_e2m1": dict(via=(3, 1), scale_floor=2.0 ** -126)}


class Unavailable(RuntimeError):
    """mxq is missing, or this edge cannot be reproduced: degrade the tier, never the numbers."""


def available() -> tuple[bool, str]:
    if not models.paths():
        return False, models.mxq_missing()
    try:
        import mxq  # noqa: F401
    except Exception as exc:                       # pragma: no cover
        return False, f"mxq import failed: {exc}"
    return True, f"mxq {models.mxq_commit()}"


def run(spec, recipe, *, dtype: str = "fp8_e4m3", edges: dict | None = None, shipped: bool = True,
        lut=None) -> dict:
    """The reference for one KernelSpec on the machine ``recipe`` (a hardware recipe) describes, with operands
    in ``dtype`` (the run recipe's ``operand_fmt``). Rounding and scale floor are the chip's: the kernel path
    refuses a run recipe that asks for others (``config.recipe.check``).

    Returns ``{"y", "stages", "shipped_y", "tier", "model"}``: ``y`` the final output (fp32 values of
    bf16 bits), ``stages`` every intermediate by name, ``shipped_y`` the as-shipped output or None,
    ``model`` what was run, for the results record. A LUT format needs ``lut``, the run's codebook
    settings (``config.recipe.lut_settings``): the wire operands are built with them.
    """
    ok, why = available()
    if not ok:
        raise Unavailable(why)
    try:
        fmt = _scheme.mxq_format(dtype)
    except _scheme.RecipeError as exc:
        raise Unavailable(str(exc)) from exc
    arith, sched, window = _scheme.datapath(recipe)
    if recipe.block != BLOCK:
        raise Unavailable(f"{recipe.name}: block = {recipe.block}, but the wire operands are quantized "
                          f"in groups of {BLOCK}; this model cannot follow")
    codebook = _scheme.is_codebook(dtype)
    if codebook and lut is None:
        raise Unavailable(f"{dtype} is LUT-indexed: pass lut= (config.recipe.lut_settings of the two recipes)")
    y, stages = _walk(spec, dtype, edges, _mesh_hw(recipe, arith, sched, window, dtype, fmt, lut), lut)
    shipped_y = None
    if shipped:
        s_arith, _, _ = _scheme.shipped_datapath(recipe)
        shipped_y, _ = _walk(spec, dtype, None, _mesh_shipped(recipe, s_arith, sched, window, fmt), lut)
    return {
        "y": y, "stages": stages, "shipped_y": shipped_y, "tier": TIER,
        "model": {
            "source": "mxq", "commit": models.mxq_commit(),
            "arith": arith.name, "schedule": [list(s) for s in sched], "window": window,
            "block": recipe.block,
            "operand_quantizer": (f"mxq.block.mxgemmini {KERNEL_ROUNDING} floor=2^-23"
                                  + (" + compiler/codebook codebooks (wire operands)" if codebook else "")),
            "chain_requantizer": ("compiler/operands device model" if codebook
                                  else "mxq.block.mxgemmini on the bf16 output" + (" via E3M1" if dtype in _REQUANT_VIA else "")),
            "as_shipped": f"mxq block.mxquant + {s_arith.name}" if shipped else None,
            "recipe": recipe.name, "build_id": recipe.build_id(), "dtype": dtype,
        },
    }


def compare(hw: np.ndarray, ref: np.ndarray) -> dict:
    """Hardware vs this model. Bit-identity is the headline; the rest is for the log."""
    hw = np.asarray(hw, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if hw.shape != ref.shape:
        return {"shape_mismatch": [list(hw.shape), list(ref.shape)], "identical": False}
    same = hw == ref                                   # NaN != NaN: the same rule grade/metrics.bit_exact_diff grades by
    diff = np.abs(hw - ref)
    denom = float(np.linalg.norm(ref))
    return {
        "identical": bool(same.all()),
        "n_identical": int(same.sum()),
        "total": int(same.size),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "rel_fro": float(np.linalg.norm(hw - ref) / denom) if denom else float("inf"),
    }


def line(metrics: dict) -> str:
    """The terminal block for a run that had this model: VERDICT when spike ran, MXQUANT otherwise."""
    corr = metrics.get("correctness_vs_mxquant")
    gap = (metrics.get("delta_vs_mxquant_as_shipped") or {}).get("rel_fro")
    fp32 = metrics["accuracy_vs_fp32_reference"]["rel_fro"]
    info = metrics.get("mxquant") or {}
    if corr is not None:
        ok = "PASS" if metrics["pass"] else "FAIL"
        out = (f"\nVERDICT  {ok}  hardware {'==' if corr['bit_exact'] else '!='} mxquant"
               f"  ({corr['total_elements'] - corr['n_mismatch']}/{corr['total_elements']}"
               f" identical, max|d| {corr['max_abs_diff']:g})")
        if gap is not None:
            out += f"\n         MXQuant as shipped is {gap:.2%} from the hardware -- the model-vs-silicon gap"
        out += f"\n         fp32 {fp32:.4%} (context), cycles {metrics.get('total_cycles')}"
        return out
    # no hardware ran: the model alone, against fp32 and as-shipped, NO VERDICT
    out = (f"\nMXQUANT  {info.get('recipe', '?')}/{info.get('dtype', '?')}   fp32 {fp32:.4%} (context)")
    if gap is not None:
        out += f"   as-shipped gap {gap:.2%}"
    out += "   (spike not run -- NO VERDICT)"
    return out


# --- internals --------------------------------------------------------------------------------------

def _walk(spec, dtype: str, edges: dict | None, mesh, lut):
    """Run the whole KernelSpec: host stages in fp32, mesh stages through ``mesh(A, W, a_px)``."""
    from kernels.spec import INPUT

    vals: dict[str, np.ndarray] = {INPUT: spec.x.numpy().astype(np.float32)}

    def operand(ref: str) -> np.ndarray:
        base, tr = (ref[:-2], True) if ref.endswith(".T") else (ref, False)
        v = vals[base]
        return np.ascontiguousarray(v.T) if tr else v

    prev = INPUT
    edge_map: dict = edges or {}
    for st in spec.stages:
        if not st.on_mesh:                           # host stage: fp32, as on the Rocket core
            vals[st.name] = np.asarray(st.run(*[operand(r) for r in (st.srcs or (prev,))]),
                                       dtype=np.float32)
        else:
            a = operand(st.lhs or prev)
            b = (st.weight.numpy().astype(np.float32) if st.weight is not None
                 else operand(st.rhs))
            lhs_name = (st.lhs or prev).removesuffix(".T")
            edge = edge_map.get(lhs_name, {})
            a_px = None
            if edge.get("via") == VIA_REQUANT:
                a_px = _requant(vals[lhs_name], dtype, edge.get("books"), lut)
            vals[st.name] = mesh(a, b, a_px)
        prev = st.name
    return vals[spec.stages[-1].name], {k: v for k, v in vals.items() if k != INPUT}


def _quantize(V: np.ndarray, fmt: str, axis: int, *, scale_floor: float = SCALE_FLOOR, via=None):
    """mxq's block quantizer in the hardware's convention: RNE, block max floored at 2^-23 unless the caller
    says otherwise (the fp4 requantizer: 2^-126 and ``via=(3, 1)``)."""
    from mxq import block
    return block.mxgemmini.quantize(torch.from_numpy(np.ascontiguousarray(V, np.float32)), fmt, axis=axis,
                                    block_size=BLOCK, rounding_mode=KERNEL_ROUNDING, scale_floor=scale_floor, via=via)


def _operands(A: np.ndarray, W: np.ndarray, fmt: str):
    """``(PA [K][M], XA [K/32][M], PB [K][N], XB [K/32][N])``: both block along K, from mxq."""
    PA, XA = _quantize(A.T, fmt, axis=0)
    PB, XB = _quantize(W, fmt, axis=0)
    return PA, XA, PB, XB


def _requant(C: np.ndarray, dtype: str, books, lut) -> tuple[torch.Tensor, torch.Tensor]:
    """The A operand a chained stage receives: the device requantizer's output, ``(P [N][M], X [N/32][M])``.

    The requantizer reads the accumulator out of SMEM, which holds bf16, blocks each row along N and
    writes one E8M0 scale per block; mxq's quantizer on that bf16 value is the same thing for
    fp8_e4m3 (measured, see the module docstring). The formats mxq cannot follow go to the device
    model.
    """
    if _scheme.is_codebook(dtype):
        return _device_requant(C, dtype, books, lut)
    return _requant_mxq(C, dtype)


def _requant_mxq(C: np.ndarray, dtype: str) -> tuple[torch.Tensor, torch.Tensor]:
    if C.shape[1] % BLOCK:                       # the device blocks its output in 32s along N; mxq would pad
        raise Unavailable(f"{dtype} chained: the requantizer blocks its output in {BLOCK}s along N, and this "
                          f"stage's N is {C.shape[1]}. A partial block is not what the device does with a full tile.")
    C_bf16 = torch.from_numpy(np.ascontiguousarray(C, np.float32)).to(torch.bfloat16).to(torch.float32).numpy()
    P, X = _quantize(C_bf16, _scheme.mxq_format(dtype), axis=1, **_REQUANT_VIA.get(dtype, {}))
    return P.t().contiguous(), X.t().contiguous()


def _mesh_hw(recipe, arith, sched, window: int, dtype: str, fmt: str, lut):
    """``A[M][K] @ W[K][N]`` on the operands the device is given, through mxq's systolic column."""
    from mxq import matmul
    codebook = _scheme.is_codebook(dtype)

    def mesh(A: np.ndarray, W: np.ndarray, a_px) -> np.ndarray:
        if codebook:
            PA, XA, PB, XB = _device_operands(A, W, dtype, a_px, lut)
        else:
            PA, XA, PB, XB = _operands(A, W, fmt)
            if a_px is not None:
                PA, XA = a_px                        # a chained operand: the requantizer already made it
        Y = matmul.systolic(PA.float(), XA.float(), PB.float(), XB.float(), arith, sched,
                            size=window, block_size=recipe.block)
        return Y.numpy().astype(np.float32)
    return mesh


def _mesh_shipped(recipe, arith, sched, window: int, fmt: str):
    """MXQuant as published: its own operand codes from the floats, its own arithmetic."""
    from mxq import block, matmul

    def mesh(A: np.ndarray, W: np.ndarray, a_px) -> np.ndarray:
        PA, XA = block.mxquant.quantize(torch.from_numpy(np.ascontiguousarray(A.T, np.float32)),
                                        fmt, axis=0, block_size=recipe.block)
        PB, XB = block.mxquant.quantize(torch.from_numpy(np.ascontiguousarray(W, np.float32)),
                                        fmt, axis=0, block_size=recipe.block)
        Y = matmul.systolic(PA, XA, PB, XB, arith, sched, size=window, block_size=recipe.block)
        return Y.numpy().astype(np.float32)
    return mesh


# --- what mxq does not have: the codebook formats, on the hardware team's model in compiler/operands.py ---

def _device_operands(A: np.ndarray, W: np.ndarray, dtype: str, a_px, lut):
    """Codebook formats: the wire operands the compiler emits (indices + per-row-pair tables), decoded."""
    from compiler.operands import quantize_operand, wire_to_px
    bc, bsc, bl = quantize_operand(np.ascontiguousarray(W, np.float32), side="b", dtype=dtype, lut=lut)
    PB, XB = wire_to_px(bc, bsc, side="b", dtype=dtype, books=bl, lut=lut)
    if a_px is not None:
        PA, XA = a_px
    else:
        ac, asc, al = quantize_operand(np.ascontiguousarray(A, np.float32), side="a", dtype=dtype, lut=lut)
        PA, XA = wire_to_px(ac, asc, side="a", dtype=dtype, books=al, lut=lut)
    return PA, XA, PB, XB


def _device_requant(C: np.ndarray, dtype: str, books, lut):
    """The device requantizer transcribed from gemmini.cc / the extracted mesh model (compiler/operands.py)."""
    from compiler.operands import NotModelled, requantize_chained
    try:
        return requantize_chained(C, dtype=dtype, books=books, lut=lut)
    except NotModelled as exc:
        raise Unavailable(str(exc)) from exc
