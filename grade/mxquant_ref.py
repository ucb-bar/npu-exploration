"""Run a kernel through MXQuant — the model the quantization work is actually done in.

This is what the hardware is graded against (``merlin_glue_port_plan.md`` D7). Two configurations,
answering two different questions:

* ``rtl_exact=True`` — MXQuant with ``rtl_exact/rtl_datapath.install()``, under which its simulated
  matmul is **bit-identical** to the datapath. Disagreement is a bug in the RTL, in spike, or in
  our codegen. This is the pass/fail gate, and it has no tolerance.
* ``rtl_exact=False`` — MXQuant as a researcher would configure it. It will NOT agree; the gap is
  the interesting number, because it says how far an MXQuant accuracy result sits from what the
  chip computes.

Nothing here reimplements MXQuant or the datapath. ``rtl_exact/`` owns the three behaviours that
differ and imports its arithmetic primitives from the hardware golden model; this module only walks
a :class:`kernels.spec.KernelSpec` and calls ``MXLinearSim`` once per mesh stage.

**Chained stages go beyond what rtl_exact was verified on**, and that is deliberate rather than
overlooked. It was proven on a llama MLP whose every matmul takes host-quantized operands; in a
fused chain, stage *i+1*'s A operand is the device's own requantizer output. The two should still
agree — the device's cross-tile accumulate is bf16 and so is MXQuant's under ``rtl_exact``, and FP8
requant is byte-identical to MXQuant's quantizer (``chain_seam_hw_notes.md`` §9.4) — but "should"
is why ``linear`` is gated before ``mlp2``, and ``mlp2`` before the depth sweep.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RTL_EXACT = REPO / "rtl_exact"
MXQ = REPO / "MXQuant"

#: MXQuant's prod/acc bundle: the module carrying ``MXLinearSim``. It lives on a branch rather than
#: in the checked-out tree, so it is extracted once and cached instead of per run.
PRODACC_FILES = ("eval_complete.py", "mx_quantization.py", "lut_quantization.py")
BRANCH = "origin/chloe-branch-all"
CACHE = REPO / "out" / "mxquant_prodacc"


class MxQuantUnavailable(RuntimeError):
    """MXQuant, its prod/acc bundle, or the rtl_exact config could not be loaded."""


def _prodacc_dir() -> Path:
    """The directory holding ``eval_complete.py``, extracting it from the branch if needed."""
    for cand in (MXQ / "prodacc_bundle", MXQ / "FP4_complete_integration_e2e", CACHE):
        if (cand / "eval_complete.py").exists():
            return cand
    if not MXQ.is_dir():
        raise MxQuantUnavailable(f"MXQuant checkout not found at {MXQ}")
    CACHE.mkdir(parents=True, exist_ok=True)
    for f in PRODACC_FILES:
        blob = subprocess.run(["git", "show", f"{BRANCH}:prodacc_bundle/{f}"],
                              cwd=MXQ, capture_output=True, text=True)
        if blob.returncode != 0:
            raise MxQuantUnavailable(
                f"cannot read prodacc_bundle/{f} from {BRANCH} in {MXQ}:\n{blob.stderr}")
        (CACHE / f).write_text(blob.stdout)
    return CACHE


_LOADED: dict = {}


def _load():
    """Import MXQuant's simulator and our RTL config, once per process."""
    if _LOADED:
        return _LOADED
    for p in (str(RTL_EXACT), str(_prodacc_dir())):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import eval_complete as EC          # noqa: N806
        import rtl_datapath
    except ImportError as exc:
        raise MxQuantUnavailable(f"cannot import MXQuant's simulator: {exc}") from exc
    _LOADED.update(EC=EC, rtl=rtl_datapath, cfg=rtl_datapath.load_config(),
                   pristine=EC.MXLinearSim._simulate_atw)
    return _LOADED


def available() -> tuple[bool, str]:
    """``(ok, reason)`` — whether this grader can run at all."""
    try:
        env = _load()
    except MxQuantUnavailable as exc:
        return False, str(exc)
    return True, f"MXQuant {env['cfg'].mx_fmt}, window {env['cfg'].window}"


@dataclass
class MxQuantRun:
    """One kernel simulated in MXQuant."""
    y: np.ndarray                                   #: final output, fp32
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    config: str = "rtl_exact"

    @property
    def rtl_exact(self) -> bool:
        return self.config == "rtl_exact"


def _matmul(env, A: np.ndarray, W: np.ndarray, dtype: str = "fp8_e4m3",
            a_px=None) -> np.ndarray:
    """``A[M][K] @ W[K][N]`` through ``MXLinearSim``, on the operands the DEVICE was given.

    ``MXLinearSim`` wraps an ``nn.Linear`` (weight ``[out][in]``, so W is transposed in), and would
    normally block-quantize both operands itself. We hand it the wire operands instead — decoded
    straight back from the codes and scales our compiler emitted — via the `_wire_operands` hook
    `rtl_datapath.install` adds.

    That is what makes the CODEBOOK formats work. Their operands are 4-bit indices into a 16-entry
    table fitted to the data; MXQuant re-deriving it could not match (its k-means seeds randomly),
    and skipping the codebook step entirely measured 12-15% off. For the direct formats it changes
    nothing — verified bit-identical either way.
    """
    import torch
    from torch import nn

    from app.mxq_golden import quantize_operand, wire_to_px

    EC, cfg = env["EC"], env["cfg"]
    f = _fmt().get(dtype, where="mxquant reference")
    K, N = W.shape
    layer = nn.Linear(K, N, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(W.T, dtype=np.float32)))
    sim = EC.MXLinearSim(layer, f.mxq, False, cfg.product[0], cfg.product[1],
                         cfg.acc_schedule, 0, 0, window=cfg.window)

    bc, bsc, bl = quantize_operand(np.ascontiguousarray(W, np.float32), side="b", dtype=dtype)
    if a_px is not None:
        PA, XA = a_px                    # a chained operand: the requantizer already made it
    else:
        ac, asc, al = quantize_operand(np.ascontiguousarray(A, np.float32), side="a", dtype=dtype)
        PA, XA = wire_to_px(ac, asc, side="a", dtype=dtype, books=al)
    PB, XB = wire_to_px(bc, bsc, side="b", dtype=dtype, books=bl)
    sim._wire_operands = (PA, XA, PB, XB)

    with torch.no_grad():
        out = sim(torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32)))
    return out.numpy().astype(np.float32)


def _fmt():
    from app import mxformats
    return mxformats


#: How a value reached the mesh as an operand. THE LOWERING DECIDES THIS, not the graph shape —
#: the same edge is produced differently depending on which emitter ran, and the reference has to
#: model whichever actually happened:
#:
#:   ``host``     the value passed through host memory and was re-quantized there
#:                (``mx_quantize_rows``, MXQuant's convention). Every edge of the GRAPH lowering,
#:                and every edge of the per-stage path.
#:   ``requant``  the value never left the device: the hardware requantizer wrote it, in the
#:                FORMAT's own ``out_requant`` convention. Every intermediate of the fused CHAIN.
#:
#: The two coincide for FP8 — its requantizer was migrated to MXQuant's convention — and diverge
#: for the nibble formats, which requantize with the hardware's own two-stage rounding. So
#: inferring this from graph shape would be right for FP8 and silently wrong for FP4/FP6. It is
#: declared instead.
VIA_HOST = "host"
VIA_REQUANT = "requant"


def simulate(spec, *, rtl_exact: bool = True, dtype: str = "fp8_e4m3",
             edges: dict | None = None) -> MxQuantRun:
    """Run a whole :class:`KernelSpec` in MXQuant and return its outputs.

    Walks the same graph the device does: mesh stages go to ``MXLinearSim``, host stages run their
    own fp32 function — which is what the Rocket core does too, so it is a faithful mirror rather
    than a simplification.

    ``edges`` maps a value name to how it reached the mesh (:data:`VIA_HOST` / :data:`VIA_REQUANT`)
    and, for a codebook format, the output book it was projected onto:
    ``{"T0": {"via": VIA_REQUANT, "books": <packed>}}``. Anything unnamed defaults to
    :data:`VIA_HOST`, which is what every lowering except the fused chain does. A NEW lowering must
    declare its own edges here; defaulting is safe rather than silent, because host re-quantization
    is the conservative case — it is what the value would look like if it had travelled.
    """
    from kernels.spec import INPUT

    env = _load()
    EC, rtl = env["EC"], env["rtl"]

    # install() is a global rebind with no inverse, so the pristine method is captured at load and
    # restored here. Without this the two configurations would be order-dependent -- once installed,
    # every later "as shipped" run in the process would silently be the RTL one.
    if rtl_exact:
        rtl.install(EC, env["cfg"])
    else:
        EC.MXLinearSim._simulate_atw = env["pristine"]
        EC.MXLinearSim._rtl_exact = False

    try:
        vals: dict[str, np.ndarray] = {INPUT: spec.x.numpy().astype(np.float32)}

        def operand(ref: str) -> np.ndarray:
            base, tr = (ref[:-2], True) if ref.endswith(".T") else (ref, False)
            v = vals[base]
            return np.ascontiguousarray(v.T) if tr else v

        prev = INPUT
        edge_map: dict = edges or {}
        for st in spec.stages:
            if not st.on_mesh:                       # host stage: fp32, as on the Rocket core
                vals[st.name] = np.asarray(st.run(*[operand(r) for r in (st.srcs or (prev,))]),
                                           dtype=np.float32)
            else:
                a = operand(st.lhs or prev)
                b = (st.weight.numpy().astype(np.float32) if st.weight is not None
                     else operand(st.rhs))
                # How the A operand was produced is a property of the LOWERING; ask, do not infer.
                lhs_name = (st.lhs or prev).removesuffix(".T")
                edge = edge_map.get(lhs_name, {})
                a_px = None
                if edge.get("via") == VIA_REQUANT:
                    from app.mxq_golden import NotModelled, requantize_chained
                    try:
                        a_px = requantize_chained(vals[lhs_name], dtype=dtype,
                                                  books=edge.get("books"))
                    except NotModelled as exc:
                        # Degrade the TIER, never the numbers: a golden we cannot produce must not
                        # be faked. The caller drops to the fp32 tier and says why.
                        raise MxQuantUnavailable(str(exc)) from exc
                vals[st.name] = _matmul(env, a, b, dtype, a_px=a_px)
            prev = st.name

        return MxQuantRun(y=vals[spec.stages[-1].name],
                          stages={k: v for k, v in vals.items() if k != INPUT},
                          config="rtl_exact" if rtl_exact else "shipped")
    finally:
        EC.MXLinearSim._simulate_atw = env["pristine"]
        EC.MXLinearSim._rtl_exact = False


def compare(hw: np.ndarray, ref: np.ndarray) -> dict:
    """Hardware against an MXQuant run. Bit-identity is the headline, not an error norm."""
    hw = np.asarray(hw, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if hw.shape != ref.shape:
        return {"shape_mismatch": [list(hw.shape), list(ref.shape)], "identical": False}
    same = int((hw == ref).sum())
    denom = float(np.linalg.norm(ref))
    return {
        "identical": bool(same == hw.size),
        "n_identical": same,
        "total": int(hw.size),
        "max_abs_diff": float(np.abs(hw - ref).max()),
        "rel_fro": (float(np.linalg.norm(hw - ref) / denom) if denom else 0.0),
    }
