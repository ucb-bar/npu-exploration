"""Run a KernelSpec on MX-Gemmini and grade it.

ONE entry path for every kernel shape we can lower: a single matmul is a chain of
length 1, so nothing special-cases it. Each stage becomes its own command buffer,
because the backend lowers exactly one matmul per buffer
(``mxgemm_emit._plan``); intermediates travel back through the host, which is the
same structure ``app/chain_2gemm/run_chain.py`` uses.

Phase 1 grades against FP32, which measures the ACCURACY COST OF THE MX FORMAT.
It does not prove the hardware correct — that needs the MX golden (phase 2), for
which ``metrics.compare`` already reserves an optional argument.

Every hardware step is a call into the bring-up branch's own modules
(``app/mxquant.py``, ``app/mxiface.py``, the ``mx_gemmini_rocket`` backend).
Nothing here reimplements quantization, lowering, or codegen.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The intermediate dtype between chained stages: the requantizer's FP8 write-back.
INTERMEDIATE_DTYPE = "f8E4M3FN"

#: W-side block-scale target for the `weight` seam, from app/chain_2gemm/run_chain.py.
#: Chosen there by measurement, not derivation -- see mxquant.CHAIN_EXP_SHIFT for why the
#: requantizer's full-range output and the mesh's 4-bit accumulator exponent do not compose.
WEIGHT_SEAM_TARGET_EXP = -4
RESCALE_SEAM_TARGET_EXP = 2


def _wire_paths(repo: Path) -> list[str]:
    """Put the app, backend and merlin packages on sys.path.

    Done here rather than pushed onto the user as PYTHONPATH exports, so a run is one
    command; an existing PYTHONPATH still wins because we only add what is missing.
    """
    added = []
    for p in (repo / "app",
              repo / "compiler" / "targets" / "mx_gemmini_rocket",
              repo / "merlin" / "merlin" / "python"):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)
            added.append(s)
    return added


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def run(spec, *, tol: float = 0.15, simulator: str = "spike", seam: str = "weight",
        build_only: bool = False, artifacts: bool = False,
        workdir: Path | None = None, results_dir: Path | None = None,
        repo: Path | None = None, telemetry=None) -> dict:
    """Build, run and grade one KernelSpec (1..N stages). Returns the run record."""
    from .metrics import compare
    from .report import make_run_id, write_report
    from .telemetry import Telemetry

    repo = repo or REPO
    tel = telemetry or Telemetry()
    workdir = workdir or repo / "out" / "build" / spec.name
    results_dir = results_dir or repo / "results"

    wired = _wire_paths(repo)
    import numpy as np
    import torch

    import backend as mx          # the OOT merlin target package
    import mxiface
    import mxquant as q
    from backend import mxgemm_emit

    tel.log("setup", f"repo={repo}  "
                     f"merlin={'yes' if (repo / 'merlin' / 'merlin').exists() else 'MISSING'}  "
                     f"wired={len(wired)} paths")

    # --- the kernel ---------------------------------------------------------------
    n_stages = len(spec.stages)
    tel.log("kernel", f"{spec.describe()}   ({n_stages} stage"
                      f"{'s' if n_stages != 1 else ''}, seam={seam if n_stages > 1 else 'n/a'})")

    errs = spec.validate(dim=mxgemm_emit.DEFAULT_GEOMETRY["dim"],
                         block=mxgemm_emit.BLOCK_SCALE_GROUP)
    if errs:
        for e in errs:
            tel.log("illegal", e)
        raise ValueError(f"{spec.name}: {len(errs)} shape violation(s); first: {errs[0]}")

    # --- (1) the FP32 reference the hardware is graded against ----------------------
    ref_fp32 = spec.reference()
    tel.log("reference", f"fp32 torch {tuple(ref_fp32.shape)}  "
                         f"range [{ref_fp32.min():.4g}, {ref_fp32.max():.4g}]")

    if not build_only and not mx.available(simulator):
        tel.log("toolchain", f"NOT AVAILABLE for {simulator} -- source scripts/env.sh")
        raise RuntimeError(f"toolchain unavailable for simulator={simulator!r}")
    tel.log("toolchain", f"gcc={mx.runner.gcc_path()}  spike={mx.runner.spike_path()}")

    # --- (2) stage loop -------------------------------------------------------------
    art_dir = workdir / "artifacts"
    stage_records: list[dict] = []
    saved: dict[str, "np.ndarray"] = {"x": spec.x.numpy()}
    a_codes = a_scales = None
    x_np = spec.x.numpy().astype(np.float32)

    for i, st in enumerate(spec.stages):
        last = (i == n_stages - 1)
        out_name = "Y0" if last else f"T{i}"
        lhs_name = "X" if i == 0 else f"T{i-1}"
        w_np = st.weight.numpy().astype(np.float32)          # [K][N]

        # Operands. Stage 0 quantizes the model input; later stages inherit the
        # previous stage's requantizer output, already in codes+scales form.
        if i == 0:
            ops = q.quantize_matmul_operands(x_np, w_np)
        else:
            w_exp = (RESCALE_SEAM_TARGET_EXP if seam == "rescale"
                     else WEIGHT_SEAM_TARGET_EXP)
            wc, ws = q.quantize_rows(np.ascontiguousarray(w_np.T), target_exp=w_exp)
            ops = {"a_codes": a_codes, "a_scales": a_scales,
                   "b_codes": np.ascontiguousarray(wc.T), "b_scales": ws}

        iface = mxiface.matmul_interface_mlir(
            spec.m, st.n, st.k,
            out_dtype="bf16" if last else INTERMEDIATE_DTYPE,
            lhs=lhs_name, weight=f"W{i}", out=out_name)
        cb = mxiface.to_command_buffer(iface, ops)
        stage_dir = workdir / f"stage{i}"

        if build_only:
            elf = mx.compile_command_buffer(cb, stage_dir)
            tel.log("compile", f"stage {i} ({st.name}) {spec.m}x{st.n}x{st.k} -> "
                               f"{'bf16' if last else 'fp8'}  {elf} ({elf.stat().st_size} B)")
            if not last:
                tel.log("done", "build-only: later stages need the intermediate, stopping")
            stage_records.append({"stage": i, "name": st.name, "elf": str(elf)})
            break

        res = mx.run_command_buffer(cb, workdir=stage_dir, simulator=simulator)
        cycles = res["metrics"].get("cycles")
        stage_records.append({"stage": i, "name": st.name,
                              "m": spec.m, "k": st.k, "n": st.n,
                              "out_dtype": "bf16" if last else INTERMEDIATE_DTYPE,
                              "metrics": res["metrics"], "elf": res["elf"]})

        if artifacts:
            art_dir.mkdir(parents=True, exist_ok=True)
            (art_dir / f"stage{i}.interface.mlir").write_text(iface, encoding="utf-8")
            src = stage_dir / "main.c"
            if src.exists():
                shutil.copy(src, art_dir / f"stage{i}.c")
            for key, val in ops.items():
                saved[f"s{i}_{key}"] = np.asarray(val)

        if last:
            bits = np.array(res["outputs"][out_name], dtype=np.uint16)
            hw = torch.from_numpy(q.bf16_bits_to_float(bits).copy())
            if artifacts:
                saved["Y0_bits"] = bits
            tel.log("stage", f"{i} ({st.name}) {spec.m}x{st.n}x{st.k} -> bf16   "
                             f"cycles {cycles}")
        else:
            codes = np.array(res["outputs"][out_name], dtype=np.uint8)
            scales = np.array(res["outputs"][f"{out_name}_scales"], dtype=np.uint8)
            if seam == "rescale":
                a_codes, a_scales = q.rescale_for_next_gemm(codes, scales)
            else:
                a_codes, a_scales = codes, np.ascontiguousarray(scales.T)
            if artifacts:
                saved[f"{out_name}_codes"], saved[f"{out_name}_scales"] = codes, scales
            peak = float(np.abs(q.fp8_e4m3_decode(a_codes)).max())
            tel.log("stage", f"{i} ({st.name}) {spec.m}x{st.n}x{st.k} -> fp8   "
                             f"cycles {cycles}  peak code {peak:.4g}  "
                             f"E8M0 {scales.min()}..{scales.max()}")

    if build_only:
        return {"metrics": None, "run_dir": None, "stages": stage_records}

    # --- (3) grade ------------------------------------------------------------------
    tel.log("decode", f"bf16 bit patterns -> float32 {tuple(hw.shape)}")
    metrics = compare(hw, ref_fp32, None, tol_rel_fro=tol)
    metrics["stages"] = stage_records
    metrics["total_cycles"] = sum((s.get("metrics") or {}).get("cycles", 0)
                                  for s in stage_records)
    acc = metrics["accuracy_vs_fp32_reference"]
    tel.log("grade", f"rel_fro={acc['rel_fro']:.4%}  mae={acc['mae']:.4g}  "
                     f"max_abs={acc['max_abs']:.4g}  "
                     f"finite {metrics['finite']['n_finite']}/{metrics['finite']['total']}  "
                     f"cycles {metrics['total_cycles']}")

    # --- (4) record -------------------------------------------------------------------
    shape_tag = "x".join([str(spec.m), str(spec.stages[0].k)] + [str(s.n) for s in spec.stages])
    run_id = make_run_id(spec.name, shape_tag)
    artifact_paths = {"workdir": str(workdir)}
    if artifacts:
        np.savez_compressed(art_dir / "operands.npz", **saved)
        artifact_paths["rtl_replay_bundle"] = str(art_dir)
        tel.log("artifacts", f"RTL replay bundle -> {art_dir} "
                             "(interface.mlir + .c per stage, operands.npz)")

    run_dir = write_report(
        results_dir=results_dir, run_id=run_id,
        run_config={
            "kernel": spec.name, "stages": n_stages, "seam": seam if n_stages > 1 else None,
            "m": spec.m, "dims": [spec.stages[0].k] + [s.n for s in spec.stages],
            "operand_format": "mxfp8_e4m3", "output_dtype": "bf16",
            "intermediate_dtype": INTERMEDIATE_DTYPE if n_stages > 1 else None,
            "block_scale_group": q.BLOCK, "target_code_exp": q.TARGET_CODE_EXP,
            "chain_exp_shift": getattr(q, "CHAIN_EXP_SHIFT", None),
            "geometry_defaults": mxgemm_emit.DEFAULT_GEOMETRY,
            "cb_params_override": None,
            "tol_rel_fro": tol,
        },
        provenance={
            "simulator": simulator, "oracle": mx.ORACLE.get(simulator),
            "repo_head": _git_head(repo), "merlin_head": _git_head(repo / "merlin"),
            "gcc": str(mx.runner.gcc_path()), "spike": str(mx.runner.spike_path()),
            "libgemmini": str(mx.runner.libgemmini_so()),
        },
        hardware_output=hw, fp32_reference=ref_fp32, golden_model_output=None,
        metrics=metrics, artifacts=artifact_paths, telemetry=tel)

    if artifacts:
        (art_dir / "spike_result.json").write_text(
            json.dumps({"run_id": run_id, "kernel": spec.name, "seam": seam,
                        "stages": stage_records, "metrics": metrics["accuracy_vs_fp32_reference"],
                        "oracle": mx.ORACLE.get(simulator),
                        "note": "replay on RTL: same ELFs, compare Y0_bits "
                                "(spike is derived_from_rtl=False)"}, indent=2),
            encoding="utf-8")

    return {"metrics": metrics, "run_dir": run_dir, "stages": stage_records}
