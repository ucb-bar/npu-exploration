"""The three comparisons this framework makes.

  - hardware vs MXQuant/rtl_exact -> hardware CORRECTNESS. Bit-identical or it is
    a bug, in the RTL, in spike, or in our codegen. This is the verdict.
  - hardware vs MXQuant as shipped -> how far the model the quantization work is
    done in sits from the silicon. Reported, never a pass criterion.
  - hardware vs FP32 reference -> the cost of the format at all. Context only.

The fp32 comparison used to BE the verdict, keyed off a tolerance. It cannot
distinguish "MX is lossy" from "the RTL is wrong", and it is too coarse to settle
real questions: a 0.63-point move at chain depth 8 sat inside a 0.90-point
seed-to-seed band (merlin_glue_port_plan.md section 4.2). It stays as a labelled
context line so a weaker grade is never mistaken for a stronger one, and the
verdict keys off bit-identity whenever a reference is available.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def bit_exact_diff(a: Tensor, b: Tensor) -> dict:
    diff = (a - b).abs()
    n_mismatch = int((a != b).sum().item())
    return {
        "n_mismatch": n_mismatch,
        "total_elements": a.numel(),
        "max_abs_diff": float(diff.max().item()) if a.numel() else 0.0,
        "bit_exact": n_mismatch == 0,
    }


def accuracy_metrics(actual: Tensor, reference: Tensor) -> dict:
    diff = (actual - reference).detach()
    mse = float(torch.mean(diff.pow(2)).item())
    mae = float(torch.mean(diff.abs()).item())
    max_abs = float(torch.max(diff.abs()).item())
    fro_ref = float(torch.linalg.norm(reference).item())
    fro_diff = float(torch.linalg.norm(diff).item())
    rel_fro = (fro_diff / fro_ref) if fro_ref != 0.0 else float("inf")
    return {"mse": mse, "mae": mae, "max_abs": max_abs, "rel_fro": rel_fro}


def compare(hardware_output: Tensor, fp32_reference: Tensor,
            golden_model_output: Tensor | None = None,
            *, tol_rel_fro: float = 0.15,
            shipped_reference: Tensor | None = None) -> dict:
    """Grade one run.

    ``golden_model_output`` is MXQuant under ``rtl_exact`` — the datapath's own arithmetic. When it
    is present the verdict is BIT-IDENTITY, with no tolerance anywhere in it; ``tol_rel_fro`` then
    applies to nothing and is recorded only so the fp32 context line can be read.

    ``shipped_reference`` is MXQuant as a researcher would configure it. Purely reported: it is
    EXPECTED to differ, so it can never fail a run.
    """
    accuracy = accuracy_metrics(hardware_output, fp32_reference)
    finite = int(torch.isfinite(hardware_output).sum().item())
    out = {
        "tier": "mxquant_rtl_exact" if golden_model_output is not None else "fp32",
        "accuracy_vs_fp32_reference": accuracy,
        "finite": {"n_finite": finite, "total": hardware_output.numel(),
                   "all_finite": finite == hardware_output.numel()},
        "tol_rel_fro": tol_rel_fro,
    }
    if shipped_reference is not None:
        out["delta_vs_mxquant_as_shipped"] = {
            **accuracy_metrics(hardware_output, shipped_reference),
            **bit_exact_diff(hardware_output, shipped_reference),
        }
    if golden_model_output is not None:
        correctness = bit_exact_diff(hardware_output, golden_model_output)
        out["correctness_vs_golden_model"] = correctness
        out["pass"] = correctness["bit_exact"] and out["finite"]["all_finite"]
    else:
        out["correctness_vs_golden_model"] = None
        out["pass"] = out["finite"]["all_finite"] and accuracy["rel_fro"] <= tol_rel_fro
    return out
