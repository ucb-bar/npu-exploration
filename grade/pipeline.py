"""Run a KernelSpec on MX-Gemmini and grade it.

ONE entry path for every kernel shape we can lower: a single matmul is a chain of
length 1, so nothing special-cases it. Two lowerings sit behind that entry:

* **fused** (``spec.is_chain``) — the whole chain becomes ONE command buffer, ONE
  ELF, ONE spike run. Intermediates never come back to the host: stage *i*'s
  requantized FP8 output IS stage *i+1*'s A operand, and what sits between them is
  the seam the backend emits (``mxgemm_emit._emit_seam``) — a scale-byte transpose,
  and on ``--seam rescale`` a code shift as well. Everything the seam has to do
  exists because of a hardware property; each one is catalogued in
  ``planning/chain_seam_hw_notes.md``.
* **per-stage** — one command buffer per matmul, intermediates carried through the
  host as float. Required by any graph that is not a straight chain (attention
  contracts two computed values and has a host softmax), and reachable for a chain
  via ``per_stage_elf=True`` for differential debugging.

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
SPEC_INPUT = "x"

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


def _fused_command_buffer(spec, seam: str) -> tuple[str, dict, list[dict]]:
    """Lower a WHOLE matmul chain to one interface module + one command buffer.

    Only stage 0 supplies an A operand. Every later stage's A is the previous stage's requantizer
    output, produced on device and consumed in place — it never travels through the host, so it
    never appears in the ``mx_operands`` side channel (the backend refuses one that does).

    The seam's compensation is chosen HERE, because it is a quantization decision:

    * ``weight`` — nothing is done to the codes; the next stage's WEIGHT is quantized low enough
      (``WEIGHT_SEAM_TARGET_EXP``) that the product of a full-range requant operand and it stays
      inside the mesh's 4-bit accumulator exponent. Free on device.
    * ``rescale`` — the codes are shifted on device instead, so the stage declares
      ``chain_code_shift`` and ships the 256-entry table that performs it. The backend will not
      invent that table: it is the element format's encode/decode, which ``app/mxquant.py`` owns.

    Both exist only because ``MxRequantizer.scala:35`` hardwires the output exponent to fill the
    element format; see ``chain_seam_hw_notes.md`` §1, which is also where the §1a caveat lives
    (``weight`` is measured-safe, not bounded-safe).
    """
    import numpy as np

    import mxiface
    import mxquant as q

    mnk = spec.stage_mnk()
    stages = spec.stages                      # is_chain => every stage is a mesh matmul
    n = len(stages)
    shift = q.CHAIN_EXP_SHIFT if seam == "rescale" else 0
    chained_w_exp = RESCALE_SEAM_TARGET_EXP if seam == "rescale" else WEIGHT_SEAM_TARGET_EXP

    ms: list = []
    bundles: list[dict] = []
    meta: list[dict] = []
    for i, st in enumerate(stages):
        m_, k_, n_ = mnk[st.name]
        last = i == n - 1
        out_name = "Y0" if last else f"T{i}"
        # Stage 0's A is host-quantized X, so its weight pairs with a TARGET_CODE_EXP operand.
        w_exp = q.TARGET_CODE_EXP if i == 0 else chained_w_exp
        w_kn = st.weight.numpy().astype(np.float32)
        # B is blocked along ITS K axis, so quantize [N][K] and transpose the codes back to [K][N] —
        # the layout the device reads. Same two calls quantize_matmul_operands makes.
        wc_nk, ws = q.quantize_rows(np.ascontiguousarray(w_kn.T), target_exp=w_exp)
        bundle: dict = {"b_codes": np.ascontiguousarray(wc_nk.T), "b_scales": ws}
        if i == 0:
            a_codes, a_scales = q.quantize_rows(spec.x.numpy().astype(np.float32),
                                                target_exp=q.TARGET_CODE_EXP)
            bundle |= {"a_codes": a_codes, "a_scales": a_scales}
        if shift and not last:
            bundle["chain_code_lut"] = q.code_shift_lut(shift)
        bundles.append(bundle)

        ms.append(mxiface.MatmulStage(
            m=m_, k=k_, n=n_, weight=f"W{i}", out=out_name,
            lhs="X" if i == 0 else f"T{i - 1}",
            out_dtype="bf16" if last else INTERMEDIATE_DTYPE,
            chain_code_shift=0 if last else shift))
        meta.append({"stage": i, "name": st.name, "where": "mesh", "out": out_name,
                     "m": m_, "k": k_, "n": n_,
                     "out_dtype": "bf16" if last else INTERMEDIATE_DTYPE,
                     "weight_target_exp": w_exp, "fused": True})

    iface = mxiface.chain_interface_mlir(ms)
    return iface, mxiface.to_command_buffer(iface, bundles), meta


def run(spec, *, tol: float = 0.15, simulator: str = "spike", seam: str = "weight",
        per_stage_elf: bool = False,
        build_only: bool = False, artifacts: bool = False,
        workdir: Path | None = None, results_dir: Path | None = None,
        repo: Path | None = None, telemetry=None) -> dict:
    """Build, run and grade one KernelSpec (1..N stages). Returns the run record.

    ``per_stage_elf`` forces a chain onto the one-ELF-per-matmul path it would otherwise skip. The
    two are meant to agree bit-for-bit -- fusing changes how many programs run, not what is computed
    -- so the difference between them is a debugging instrument, not a mode.
    """
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

    # --- (2) lower and run ------------------------------------------------------------
    art_dir = workdir / "artifacts"
    stage_records: list[dict] = []
    saved: dict[str, "np.ndarray"] = {"x": spec.x.numpy()}
    mnk = spec.stage_mnk()
    fused_cycles = None

    # A straight chain fuses into ONE program: the whole point is that an intermediate never comes
    # back to the host, so there is nothing here to carry between stages. Anything else -- a host
    # stage, or a B operand that is itself computed -- has no on-device operand path and runs one
    # command buffer per matmul, carrying values as float.
    fused = spec.is_chain and not per_stage_elf
    if fused:
        iface, cb, stage_records = _fused_command_buffer(spec, seam)
        chain_dir = workdir / "chain"
        tel.log("lower", f"{len(stage_records)} stage(s) -> ONE command buffer "
                         f"({len(cb['commands'])} commands, {len(cb['tensors'])} leaf tensors)"
                         + (f"   seam={seam}" if len(stage_records) > 1 else ""))

        if build_only:
            elf = mx.compile_command_buffer(cb, chain_dir)
            tel.log("compile", f"fused chain -> {elf} ({elf.stat().st_size} B)")
            for rec in stage_records:
                rec["elf"] = str(elf)
            return {"metrics": None, "run_dir": None, "stages": stage_records}

        res = mx.run_command_buffer(cb, workdir=chain_dir, simulator=simulator)
        raw = res["metrics"]
        fused_cycles = raw.get("cycles")

        for i, rec in enumerate(stage_records):
            rec["elf"] = res["elf"]
            rec["metrics"] = {k: raw[k] for k in (f"cycles_stage{i}", f"seam_cycles_stage{i}")
                              if k in raw}
            # A one-stage chain emits the single-matmul driver, which reports only `cycles` -- there
            # are no stages to break down and no seam to measure.
            cyc = rec["metrics"].get(f"cycles_stage{i}", fused_cycles if len(stage_records) == 1
                                     else None)
            if rec["out_dtype"] == "bf16":
                tel.log("stage", f"{i} ({rec['name']}) {rec['m']}x{rec['n']}x{rec['k']} -> bf16"
                                 f"   cycles {cyc}")
                continue
            # A non-final stage's product stayed on device; it still REPORTS, so fusing costs none
            # of the per-stage telemetry the host-carried path produced.
            codes = np.array(res["outputs"][rec["out"]], dtype=np.uint8)
            scales = np.array(res["outputs"][f"{rec['out']}_scales"], dtype=np.uint8)
            peak = float(np.abs(q.fp8_e4m3_decode(codes)).max())
            rec["peak_code"] = peak
            tel.log("stage", f"{i} ({rec['name']}) {rec['m']}x{rec['n']}x{rec['k']} -> fp8   "
                             f"cycles {cyc}  "
                             f"peak code {peak:.4g}  E8M0 {scales.min()}..{scales.max()}")
            seam_cyc = raw.get(f"seam_cycles_stage{i + 1}")
            tel.log("seam", f"{i} -> {i + 1}  cycles {seam_cyc}  "
                            + ("scale transpose + code shift (hw notes 1,2)" if seam == "rescale"
                               else "scale transpose only (hw notes 2)"))
            if artifacts:
                saved[f"{rec['out']}_codes"], saved[f"{rec['out']}_scales"] = codes, scales

        bits = np.array(res["outputs"]["Y0"], dtype=np.uint16)
        hw = torch.from_numpy(q.bf16_bits_to_float(bits).copy())
        if artifacts:
            art_dir.mkdir(parents=True, exist_ok=True)
            (art_dir / "chain.interface.mlir").write_text(iface, encoding="utf-8")
            src = chain_dir / "main.c"
            if src.exists():
                shutil.copy(src, art_dir / "chain.c")
            saved["Y0_bits"] = bits
            for i, bundle in enumerate(cb["mx_operands"]):
                for key, val in bundle.items():
                    saved[f"s{i}_{key}"] = np.asarray(val)

    if not fused:
        a_codes = a_scales = None
        x_np = spec.x.numpy().astype(np.float32)

        # Reached either by a graph that cannot fuse, or by per_stage_elf on one that can. A chain
        # run this way still keeps its intermediates in the requantizer's codes+scales form -- it is
        # the host that carries them between runs, instead of the device carrying them between
        # stages -- so the `--seam` compensation still applies.
        requant_chain = spec.is_chain and len(spec.stages) > 1
        vals: dict[str, "np.ndarray"] = {SPEC_INPUT: x_np}
        carried: dict[str, tuple] = {}            # stage name -> (a_codes, a_scales) when requant-chained
        prev = SPEC_INPUT
        final = spec.stages[-1].name

        def operand(ref: str) -> "np.ndarray":
            base, tr = (ref[:-2], True) if ref.endswith(".T") else (ref, False)
            v = vals[base]
            return np.ascontiguousarray(v.T) if tr else v

        for i, st in enumerate(spec.stages):
            last = (st.name == final)

            if not st.on_mesh:                    # ---- host stage: the mesh cannot do it ----
                vals[st.name] = np.asarray(st.fn(operand(st.src or prev)), dtype=np.float32)
                tel.log("host", f"{i} ({st.name}) {vals[st.name].shape}"
                                + (f"  {st.note}" if st.note else ""))
                stage_records.append({"stage": i, "name": st.name, "where": "host",
                                      "note": st.note})
                prev = st.name
                continue

            m_, k_, n_ = mnk[st.name]
            out_name = "Y0" if last else f"T{i}"
            lhs_ref = st.lhs or prev
            lhs_name = "X" if lhs_ref == SPEC_INPUT else f"A{i}"

            # Operands. In a requant chain, a later stage inherits the previous stage's requantizer
            # output already in codes+scales form; otherwise both operands are quantized from float.
            if requant_chain and i > 0:
                a_codes, a_scales = carried[prev]
                w_exp = (RESCALE_SEAM_TARGET_EXP if seam == "rescale" else WEIGHT_SEAM_TARGET_EXP)
                w_np = st.weight.numpy().astype(np.float32)
                wc, ws = q.quantize_rows(np.ascontiguousarray(w_np.T), target_exp=w_exp)
                ops = {"a_codes": a_codes, "a_scales": a_scales,
                       "b_codes": np.ascontiguousarray(wc.T), "b_scales": ws}
            else:
                b_np = (st.weight.numpy().astype(np.float32) if st.weight is not None
                        else operand(st.rhs))
                ops = q.quantize_matmul_operands(operand(lhs_ref), b_np)

            # A mesh stage commits through the requantizer only when a later stage will consume it in
            # that form; everything else reads back as bf16.
            emit_fp8 = requant_chain and not last
            iface = mxiface.matmul_interface_mlir(
                m_, n_, k_, out_dtype=INTERMEDIATE_DTYPE if emit_fp8 else "bf16",
                lhs=lhs_name, weight=f"W{i}", out=out_name)
            cb = mxiface.to_command_buffer(iface, ops)
            stage_dir = workdir / f"stage{i}"

            if build_only:
                elf = mx.compile_command_buffer(cb, stage_dir)
                tel.log("compile", f"stage {i} ({st.name}) {m_}x{n_}x{k_} -> "
                                   f"{'fp8' if emit_fp8 else 'bf16'}  {elf} ({elf.stat().st_size} B)")
                if not last:
                    tel.log("done", "build-only: later stages need the intermediate, stopping")
                stage_records.append({"stage": i, "name": st.name, "elf": str(elf)})
                break

            res = mx.run_command_buffer(cb, workdir=stage_dir, simulator=simulator)
            cycles = res["metrics"].get("cycles")
            stage_records.append({"stage": i, "name": st.name, "where": "mesh",
                                  "m": m_, "k": k_, "n": n_,
                                  "out_dtype": INTERMEDIATE_DTYPE if emit_fp8 else "bf16",
                                  "metrics": res["metrics"], "elf": res["elf"]})

            if artifacts:
                art_dir.mkdir(parents=True, exist_ok=True)
                (art_dir / f"stage{i}.interface.mlir").write_text(iface, encoding="utf-8")
                src = stage_dir / "main.c"
                if src.exists():
                    shutil.copy(src, art_dir / f"stage{i}.c")
                for key, val in ops.items():
                    saved[f"s{i}_{key}"] = np.asarray(val)

            if emit_fp8:
                codes = np.array(res["outputs"][out_name], dtype=np.uint8)
                scales = np.array(res["outputs"][f"{out_name}_scales"], dtype=np.uint8)
                if seam == "rescale":
                    carried[st.name] = q.rescale_for_next_gemm(codes, scales)
                else:
                    carried[st.name] = (codes, np.ascontiguousarray(scales.T))
                if artifacts:
                    saved[f"{out_name}_codes"], saved[f"{out_name}_scales"] = codes, scales
                peak = float(np.abs(q.fp8_e4m3_decode(carried[st.name][0])).max())
                tel.log("stage", f"{i} ({st.name}) {m_}x{n_}x{k_} -> fp8   cycles {cycles}  "
                                 f"peak code {peak:.4g}  E8M0 {scales.min()}..{scales.max()}")
            else:
                bits = np.array(res["outputs"][out_name], dtype=np.uint16)
                vals[st.name] = q.bf16_bits_to_float(bits).copy()
                if last:
                    hw = torch.from_numpy(vals[st.name])
                    if artifacts:
                        saved["Y0_bits"] = bits
                tel.log("stage", f"{i} ({st.name}) {m_}x{n_}x{k_} -> bf16   cycles {cycles}")
            prev = st.name

    if build_only:
        return {"metrics": None, "run_dir": None, "stages": stage_records}

    # --- (3) grade ------------------------------------------------------------------
    tel.log("decode", f"bf16 bit patterns -> float32 {tuple(hw.shape)}")
    metrics = compare(hw, ref_fp32, None, tol_rel_fro=tol)
    metrics["stages"] = stage_records
    # Fused: ONE measured window spanning every stage AND every seam, which is the honest number --
    # summing the per-stage windows would silently drop the seams. Per-stage: the sum of the runs.
    metrics["total_cycles"] = (fused_cycles if fused else
                               sum((s.get("metrics") or {}).get("cycles", 0)
                                   for s in stage_records))
    if fused:
        metrics["seam_cycles"] = sum(v for s in stage_records
                                     for k, v in (s.get("metrics") or {}).items()
                                     if k.startswith("seam_cycles_stage"))
    acc = metrics["accuracy_vs_fp32_reference"]
    tel.log("grade", f"rel_fro={acc['rel_fro']:.4%}  mae={acc['mae']:.4g}  "
                     f"max_abs={acc['max_abs']:.4g}  "
                     f"finite {metrics['finite']['n_finite']}/{metrics['finite']['total']}  "
                     f"cycles {metrics['total_cycles']}")

    # --- (4) record -------------------------------------------------------------------
    _mesh = spec.mesh_stages
    shape_tag = "x".join([str(spec.m), str(mnk[_mesh[0].name][1])]
                         + [str(mnk[s.name][2]) for s in _mesh])
    run_id = make_run_id(spec.name, shape_tag)
    artifact_paths = {"workdir": str(workdir)}
    if artifacts:
        np.savez_compressed(art_dir / "operands.npz", **saved)
        artifact_paths["rtl_replay_bundle"] = str(art_dir)
        tel.log("artifacts", f"RTL replay bundle -> {art_dir} (interface.mlir + .c"
                             f"{' for the fused chain' if fused else ' per stage'}, operands.npz)")

    run_dir = write_report(
        results_dir=results_dir, run_id=run_id,
        run_config={
            "kernel": spec.name, "stages": n_stages, "seam": seam if n_stages > 1 else None,
            "lowering": "fused" if fused else "per_stage",
            "elfs": 1 if fused else len([s for s in stage_records if s.get("elf")]),
            "m": spec.m,
            "dims": [mnk[_mesh[0].name][1]] + [mnk[s.name][2] for s in _mesh],
            "mesh_stages": len(_mesh), "host_stages": len(spec.stages) - len(_mesh),
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
