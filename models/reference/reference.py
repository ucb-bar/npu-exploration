"""The reference model: what the kernel computes in fp32 with no quantization anywhere.

It is context, never the verdict: the distance from it is the cost of the MX format, and cannot
tell "MX is lossy" from "the hardware is wrong". It becomes the pass criterion only when the
mxquant model is unavailable (the ``fp32`` tier, a relative-Frobenius tolerance).
"""
from __future__ import annotations


def run(spec) -> dict:
    """``{"y": torch.Tensor}``: the whole graph in fp32, host stages included."""
    return {"y": spec.reference()}


def line(metrics: dict, *, tol: float | None = None) -> str:
    """The VERDICT line of a run graded on the fp32 tier (no mxquant available or selected)."""
    ok = {True: "PASS", False: "FAIL", None: "NO VERDICT"}[metrics["pass"]]
    acc = metrics["accuracy_vs_fp32_reference"]
    tol_txt = f", tol={tol:.2%}" if tol is not None else ""
    return (f"\nVERDICT  {ok}  (tier={metrics['tier']}, rel_fro={acc['rel_fro']:.4%}{tol_txt}, "
            f"cycles={metrics.get('total_cycles')})")
