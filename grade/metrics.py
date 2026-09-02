"""The two comparisons this framework makes.

  - hardware vs FP32 reference  -> accuracy cost of the low-precision format
  - hardware vs MX golden model -> hardware CORRECTNESS (should be ~bit-exact)

Phase 1 runs the first only, so ``golden_model_output`` is optional. When it is
absent the verdict keys off a tolerance on the fp32 comparison, which measures
format cost and CANNOT distinguish "MX is lossy" from "the RTL is wrong" — the
report labels the tier so a weaker grade is never mistaken for a stronger one.
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
            *, tol_rel_fro: float = 0.15) -> dict:
    """Grade one run. Tier is 'golden' when a golden is supplied, else 'fp32'."""
    accuracy = accuracy_metrics(hardware_output, fp32_reference)
    finite = int(torch.isfinite(hardware_output).sum().item())
    out = {
        "tier": "golden" if golden_model_output is not None else "fp32",
        "accuracy_vs_fp32_reference": accuracy,
        "finite": {"n_finite": finite, "total": hardware_output.numel(),
                   "all_finite": finite == hardware_output.numel()},
        "tol_rel_fro": tol_rel_fro,
    }
    if golden_model_output is not None:
        correctness = bit_exact_diff(hardware_output, golden_model_output)
        out["correctness_vs_golden_model"] = correctness
        out["pass"] = correctness["bit_exact"] and out["finite"]["all_finite"]
    else:
        out["correctness_vs_golden_model"] = None
        out["pass"] = out["finite"]["all_finite"] and accuracy["rel_fro"] <= tol_rel_fro
    return out
