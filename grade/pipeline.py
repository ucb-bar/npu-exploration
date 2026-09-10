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

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Default intermediate dtype between chained stages (the requantizer's FP8 write-back).
#: A recipe's ``software.intermediate_dtype`` overrides it; this is the fallback
#: config/recipe.py applies when a recipe leaves the field out.
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


_GEOMETRY_CHECKED: set[str] = set()


def _assert_geometry(console: str, recipe, simulator: str, tel) -> None:
    """libgemmini announces its mesh size on reset; hold the recipe to it.

    ``gemmini.cc:70-71`` prints ``Gemmini extension configured with: dim = N`` every
    run. Without this check a recipe naming a mesh the loaded model does not implement
    would run anyway, the golden would honour the recipe, the device would not, and
    the report would blame the hardware for a mismatch we caused.
    """
    key = f"{simulator}:{recipe.build_id()}"
    if key in _GEOMETRY_CHECKED:
        return
    m = re.search(r"dim\s*=\s*(\d+)", console)
    if not m:
        tel.log("geometry", f"WARNING: {simulator} printed no dim banner; recipe dim="
                            f"{recipe.dim} is UNVERIFIED against the running model")
        return
    got = int(m.group(1))
    if got != recipe.dim:
        raise RuntimeError(
            f"geometry mismatch: recipe {recipe.name!r} says dim={recipe.dim} but the "
            f"loaded {simulator} model reports dim={got}. Build the model for this "
            f"recipe (config/build_spike.py) instead of running against another one")
    _GEOMETRY_CHECKED.add(key)
    tel.log("geometry", f"{simulator} reports dim={got}, matches recipe")


def _libgemmini_fingerprint(mx) -> dict:
    """Identify the model that actually ran, not just where it lives.

    A path is not an identity: we have already been burned once by a libgemmini.so
    that was stale relative to its own sources (the Makefile lists only gemmini.cc as
    a prerequisite) and failed as an unhandled trap rather than an error.
    """
    try:
        so = Path(mx.runner.libgemmini_so())
        h = hashlib.sha256(so.read_bytes()).hexdigest()[:16]
        return {"path": str(so), "sha256": h,
                "mtime": datetime.fromtimestamp(so.stat().st_mtime).isoformat(timespec="seconds")}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run(spec, *, recipe=None, tol: float = 0.15, simulator: str = "spike",
        seam: str | None = None,
        build_only: bool = False, artifacts: bool = False,
        workdir: Path | None = None, results_dir: Path | None = None,
        repo: Path | None = None, telemetry=None) -> dict:
    """Build, run and grade one KernelSpec (1..N stages) on one hardware recipe.

    ``recipe`` says WHICH MX-Gemmini this is -- mesh size, product and accumulator
    widths, block-scale group, operand format. It drives all three models of that
    machine (the software model, spike, Verilator), so there is one definition per
    run and no way to grade against a machine we did not build.
    """
    from config.recipe import load as load_recipe
    from .metrics import compare
    from .report import make_run_id, write_report
    from .telemetry import Telemetry

    repo = repo or REPO
    if recipe is None:
        recipe = load_recipe("baseline")
    if simulator not in recipe.supported_backends:
        raise RuntimeError(f"recipe {recipe.name!r} does not declare support for "
                           f"{simulator!r} (supported: {', '.join(recipe.supported_backends)})")
    seam = seam or recipe.seam
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

    tel.log("recipe", f"{recipe.describe()}   build_id={recipe.build_id()}  "
                      f"src={recipe.path.name}")
    for line in recipe.ladder_lines():
        tel.log("ladder", line)
    if recipe.dim != mxgemm_emit.DEFAULT_GEOMETRY["dim"]:
        raise RuntimeError(
            f"recipe dim={recipe.dim} but the backend plans for "
            f"{mxgemm_emit.DEFAULT_GEOMETRY['dim']} (mxgemm_emit.DEFAULT_GEOMETRY); "
            "pass it through cb['params'] before running a different mesh size")

    errs = spec.validate(dim=recipe.dim, block=recipe.block)
    if errs:
        for e in errs:
            tel.log("illegal", e)
        raise ValueError(f"{spec.name}: {len(errs)} shape violation(s); first: {errs[0]}")

    # --- (1) the FP32 reference the hardware is graded against ----------------------
    ref_fp32 = spec.reference()
    tel.log("reference", f"fp32 torch {tuple(ref_fp32.shape)}  "
                         f"range [{ref_fp32.min():.4g}, {ref_fp32.max():.4g}]")

    # Point the oracle at the model built for THIS recipe. A recipe whose hardware
    # matches the stock build resolves to None and uses the shipped model as-is.
    if not build_only and simulator == "spike":
        from config.build_spike import BuildError, resolve as resolve_build
        try:
            so = resolve_build(recipe)
        except BuildError as exc:
            raise RuntimeError(f"cannot build the spike model for recipe "
                               f"{recipe.name!r}: {exc}") from exc
        if so is not None:
            os.environ["MX_LIBGEMMINI"] = str(so)
            tel.log("build", f"recipe model {recipe.build_id()} -> {so}")
        else:
            os.environ.pop("MX_LIBGEMMINI", None)
            tel.log("build", f"recipe {recipe.name!r} matches the stock build; "
                             "using the shipped libgemmini.so")

    if not build_only and not mx.available(simulator):
        tel.log("toolchain", f"NOT AVAILABLE for {simulator} -- source scripts/env.sh")
        raise RuntimeError(f"toolchain unavailable for simulator={simulator!r}")
    tel.log("toolchain", f"gcc={mx.runner.gcc_path()}  spike={mx.runner.spike_path()}")

    # --- (2) stage loop -------------------------------------------------------------
    art_dir = workdir / "artifacts"
    stage_records: list[dict] = []
    golden_out = None
    saved: dict[str, "np.ndarray"] = {"x": spec.x.numpy()}
    a_codes = a_scales = None
    x_np = spec.x.numpy().astype(np.float32)

    # A straight matmul chain keeps intermediates in the requantizer's codes+scales form (the
    # `--seam` question). Any graph with a host stage or a computed B operand carries values as
    # float instead: they must come back to the host anyway, so bf16 out + host re-quantize is both
    # simpler and avoids the requant range seam entirely.
    requant_chain = spec.is_chain and len(spec.stages) > 1
    mnk = spec.stage_mnk()
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
            ops = q.quantize_matmul_operands(operand(lhs_ref), b_np,
                                             target_exp=recipe.target_code_exp)

        # A mesh stage commits through the requantizer only when a later stage will consume it in
        # that form; everything else reads back as bf16.
        emit_fp8 = requant_chain and not last
        iface = mxiface.matmul_interface_mlir(
            m_, n_, k_, operand_fmt=recipe.operand_fmt,
            out_dtype=recipe.intermediate_dtype if emit_fp8 else recipe.out_dtype,
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
        _assert_geometry(res.get("console", ""), recipe, simulator, tel)
        stage_records.append({"stage": i, "name": st.name, "where": "mesh",
                              "m": m_, "k": k_, "n": n_,
                              "out_dtype": (recipe.intermediate_dtype if emit_fp8
                                            else recipe.out_dtype),
                              "metrics": res["metrics"], "elf": res["elf"]})

        if artifacts:
            art_dir.mkdir(parents=True, exist_ok=True)
            (art_dir / f"stage{i}.interface.mlir").write_text(iface, encoding="utf-8")
            src = stage_dir / "main.c"
            if src.exists():
                shutil.copy(src, art_dir / f"stage{i}.c")
            for key, val in ops.items():
                saved[f"s{i}_{key}"] = np.asarray(val)

        # The software-model slot: when a datapath model lands (built on the
        # fp8_matmul_model lineage), it grades THIS stage's real operands here and
        # sets golden_out, flipping metrics["tier"] from "fp32" to "golden".
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
    metrics = compare(hw, ref_fp32, golden_out, tol_rel_fro=tol)
    metrics["stages"] = stage_records
    metrics["total_cycles"] = sum((s.get("metrics") or {}).get("cycles", 0)
                                  for s in stage_records)
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
        tel.log("artifacts", f"RTL replay bundle -> {art_dir} "
                             "(interface.mlir + .c per stage, operands.npz)")

    run_dir = write_report(
        results_dir=results_dir, run_id=run_id,
        run_config={
            "kernel": spec.name, "stages": n_stages, "seam": seam if n_stages > 1 else None,
            "m": spec.m,
            "dims": [mnk[_mesh[0].name][1]] + [mnk[s.name][2] for s in _mesh],
            "mesh_stages": len(_mesh), "host_stages": len(spec.stages) - len(_mesh),
            "recipe": {"name": recipe.name, "build_id": recipe.build_id(),
                       "path": str(recipe.path), "hardware": recipe.hardware(),
                       "ladder": recipe.ladder(),
                       "prod": [recipe.prod_e, recipe.prod_m],
                       "acc_e": list(recipe.acc_e), "acc_m": list(recipe.acc_m)},
            "operand_format": recipe.operand_fmt, "output_dtype": recipe.out_dtype,
            "intermediate_dtype": recipe.intermediate_dtype if n_stages > 1 else None,
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
            "libgemmini": _libgemmini_fingerprint(mx),
            "gemmini_head": _git_head(Path(mx.runner.chipyard_root()) / "generators/gemmini"),
            "chipyard_head": _git_head(Path(mx.runner.chipyard_root())),
        },
        hardware_output=hw, fp32_reference=ref_fp32, golden_model_output=golden_out,
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
