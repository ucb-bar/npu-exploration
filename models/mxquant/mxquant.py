"""The mxquant model: the bits MX-Gemmini must produce for one kernel, computed from the recipe on mxq.

VERDICT is ``spike bits == mxquant bits`` with no tolerance anywhere in it. The model walks the same
graph the device runs: mesh stages go through ``mxq.matmul.systolic`` with the recipe's Arithmetic,
schedule and window (``config/scheme.py``), on exactly the wire operands the compiler emitted
(``app.mxq_golden.quantize_operand`` -> ``wire_to_px``), so codebook formats and hardware-requantized
chain intermediates are reproduced, not re-derived. Host stages run their own fp32 function, as they
do on the Rocket core.

``edges`` (built by the lowering that ran, ``grade/pipeline.py``) says how each intermediate reached
the mesh: ``{"via": "host"}`` -- it went through host memory and was re-quantized there -- or
``{"via": "requant", "books": ...}`` -- the hardware requantizer wrote it and the next stage read it
in place, which only the fused chain does. An edge the requantizer model cannot reproduce raises
:class:`Unavailable`; the pipeline then grades on the fp32 tier and says so. A confident wrong
reference would be worse than none.

The informational "as shipped" number is MXQuant's published simulator on this recipe's ladder:
operands from ``mxq.block.mxquant`` (MXQuant's own codes, from the floats), arithmetic
``mxq.matmul.MXQUANT``. It is reported as the model-vs-silicon gap and never grades a run.

This replaces ``grade/mxquant_ref.py``, which patched MXQuant's simulator at runtime and could not
see the recipe; that file stays untouched until the second PR, as the legacy implementation the
equivalence test (``tests/selftest_mxquant.py``) compares against.
"""
from __future__ import annotations

import numpy as np
import torch

import models
from config import scheme as _scheme
from models.mxquant.block import BLOCK

TIER = "mxquant_recipe_exact"
#: How a value reached the mesh. THE LOWERING DECIDES THIS (grade/pipeline.py writes "host" / "requant"
#: into the edge map); these names are only read here.
VIA_HOST = "host"
VIA_REQUANT = "requant"


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


def run(spec, recipe, *, dtype: str = "fp8_e4m3", edges: dict | None = None, shipped: bool = True) -> dict:
    """The reference for one KernelSpec on the machine ``recipe`` describes.

    Returns ``{"y", "stages", "shipped_y", "tier", "model"}``: ``y`` the final output (fp32 values of
    bf16 bits), ``stages`` every intermediate by name, ``shipped_y`` the as-shipped output or None,
    ``model`` what was run, for the results record.
    """
    ok, why = available()
    if not ok:
        raise Unavailable(why)
    arith, sched, window = _scheme.datapath(recipe)
    if recipe.block != BLOCK:
        raise Unavailable(f"{recipe.name}: software.block = {recipe.block}, but the wire operands are "
                          f"quantized in groups of {BLOCK} (app/mxq_golden.py); this model cannot follow")
    y, stages = _walk(spec, dtype, edges, _mesh_hw(recipe, arith, sched, window, dtype))
    shipped_y = None
    if shipped:
        s_arith, _, _ = _scheme.shipped_datapath(recipe)
        shipped_y, _ = _walk(spec, dtype, None, _mesh_shipped(recipe, s_arith, sched, window, dtype))
    return {
        "y": y, "stages": stages, "shipped_y": shipped_y, "tier": TIER,
        "model": {
            "source": "mxq", "commit": models.mxq_commit(),
            "arith": arith.name, "schedule": [list(s) for s in sched], "window": window,
            "block": recipe.block,
            "operand_quantizer": f"mxq.block.mxgemmini {_scheme.ROUNDING} floor=2^-23 (wire operands)",
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

def _walk(spec, dtype: str, edges: dict | None, mesh):
    """Run the whole KernelSpec: host stages in fp32, mesh stages through ``mesh(A, W, a_px)``.

    The same walk ``grade/mxquant_ref.simulate`` does; only the matmul differs.
    """
    from app.mxq_golden import NotModelled, requantize_chained
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
                try:
                    a_px = requantize_chained(vals[lhs_name], dtype=dtype, books=edge.get("books"))
                except NotModelled as exc:
                    raise Unavailable(str(exc)) from exc
            vals[st.name] = mesh(a, b, a_px)
        prev = st.name
    return vals[spec.stages[-1].name], {k: v for k, v in vals.items() if k != INPUT}


def _mesh_hw(recipe, arith, sched, window: int, dtype: str):
    """``A[M][K] @ W[K][N]`` on the wire operands the device was given, through mxq's systolic column."""
    from mxq import matmul
    from app.mxq_golden import quantize_operand, wire_to_px

    def mesh(A: np.ndarray, W: np.ndarray, a_px) -> np.ndarray:
        bc, bsc, bl = quantize_operand(np.ascontiguousarray(W, np.float32), side="b", dtype=dtype)
        if a_px is not None:
            PA, XA = a_px                            # a chained operand: the requantizer already made it
        else:
            ac, asc, al = quantize_operand(np.ascontiguousarray(A, np.float32), side="a", dtype=dtype)
            PA, XA = wire_to_px(ac, asc, side="a", dtype=dtype, books=al)
        PB, XB = wire_to_px(bc, bsc, side="b", dtype=dtype, books=bl)
        Y = matmul.systolic(PA.float(), XA.float(), PB.float(), XB.float(), arith, sched,
                            window=window, block_size=recipe.block)
        return Y.numpy().astype(np.float32)
    return mesh


def _mesh_shipped(recipe, arith, sched, window: int, dtype: str):
    """MXQuant as published: its own operand codes from the floats, its own arithmetic."""
    from mxq import block, matmul
    from app import mxformats
    fmt = mxformats.get(dtype, where="mxquant as-shipped", proven_only=False).mxq

    def mesh(A: np.ndarray, W: np.ndarray, a_px) -> np.ndarray:
        PA, XA = block.mxquant.quantize(torch.from_numpy(np.ascontiguousarray(A.T, np.float32)),
                                        fmt, axis=0, block_size=recipe.block)
        PB, XB = block.mxquant.quantize(torch.from_numpy(np.ascontiguousarray(W, np.float32)),
                                        fmt, axis=0, block_size=recipe.block)
        Y = matmul.systolic(PA, XA, PB, XB, arith, sched, window=window, block_size=recipe.block)
        return Y.numpy().astype(np.float32)
    return mesh
