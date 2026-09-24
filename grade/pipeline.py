"""Run a KernelSpec on MX-Gemmini and grade it.

ONE entry path for every kernel shape we can lower: a single matmul is a chain of
length 1, so nothing special-cases it. Two lowerings sit behind that entry:

* **fused** (``spec.is_chain``) — the whole chain becomes ONE command buffer, ONE
  ELF, ONE spike run. Intermediates never come back to the host: stage *i*'s
  requantized FP8 output IS stage *i+1*'s A operand, and what sits between them is
  the seam the backend emits (``mxgemm_emit._emit_seam``) — today a scale-byte
  transpose, which exists only because the requantizer writes ``[row][block]`` and
  the A-side scale memory reads ``[group][row]``. Step 3 of
  ``planning/merlin_glue_port_plan.md`` removes even that, by having the hardware
  write the scales resident and transposed.
* **per-stage** — one command buffer per matmul, intermediates carried through the
  host as float. Required by any graph that is not a straight chain (attention
  contracts two computed values and has a host softmax), and reachable for a chain
  via ``per_stage_elf=True`` for differential debugging.

The grade is against **MXQuant**, not fp32: under ``rtl_exact`` MXQuant's simulated
matmul is bit-identical to the datapath, so the verdict is bit-identity with no
tolerance in it, and a second run of MXQuant *as shipped* reports how far the model
the quantization work is done in sits from the silicon. The fp32 comparison stays as
a labelled context line — it measures the cost of the format, not correctness.

Every hardware step is a call into this repo's own modules (``app/mxq_golden.py``
for quantization — MXQuant, never transcribed — ``app/mxiface.py`` for lowering,
and the ``mx_gemmini_rocket`` backend for codegen). Nothing here reimplements any
of the three.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Default operand format. Every other format is selected with --dtype and gated by
#: app/mxformats.py, which refuses one this repo has not proven end to end.
DEFAULT_DTYPE = "fp8_e4m3"
SPEC_INPUT = "x"

# THE SEAM CONSTANTS ARE GONE, deliberately (merlin_glue_port_plan.md D3, Step 1).
#
# `WEIGHT_SEAM_TARGET_EXP = -6` and `RESCALE_SEAM_TARGET_EXP = 0` used to sit here. Both were
# derived from an accumulator bound that assumed the requantizer normalizes each output block to
# the top of the element format (448 for e4m3), which is what `MxRequantizer.scala` did when
# `log2_pmax_floor` was 8:
#
#      16 * 448 * |B|max <= 256   ->   |B|max <= 0.0357   ->   target_exp <= -6
#
# That premise is false now. FP8 was migrated to MXQuant's e2e convention in both spike and the RTL
# (chain_seam_hw_notes.md section 8): the scale exponent is `floor(log2 amax)` with no `- emax`
# term, so a block max lands in [1, 2) and `16 * |A|max * |B|max = 64` sits FOUR TIMES UNDER the
# bound with no compensation at all. Quantizing the next weight 224x down was not protecting
# anything -- it was throwing away precision to solve a problem the hardware no longer has.
#
# Every operand, on every stage, is now quantized exactly once, by MXQuant, at its natural scale.


def _wire_paths(repo: Path) -> list[str]:
    """Put the app, backend and merlin packages on sys.path.

    Done here rather than pushed onto the user as PYTHONPATH exports, so a run is one
    command; an existing PYTHONPATH still wins because we only add what is missing.
    """
    added = []
    for p in (repo,
              repo / "app",
              repo / "compiler" / "targets" / "mx_gemmini_rocket",
              repo / "merlin" / "merlin" / "python"):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)
            added.append(s)
    import models  # noqa: F401  -- the one place the mxq submodule (microscaling-quant/) joins sys.path
    return added


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _fused_command_buffer(spec, dtype: str = DEFAULT_DTYPE, *,
                          allow_lossy_chain: bool = False,
                          tel=None) -> tuple[str, dict, list[dict], dict]:
    """Lower a WHOLE matmul chain to one interface module + one command buffer.

    Only stage 0 supplies an A operand. Every later stage's A is the previous stage's requantizer
    output, produced on device and consumed in place — it never travels through the host, so it
    never appears in the ``mx_operands`` side channel (the backend refuses one that does).

    **There is no seam compensation any more**, and that is the point: every operand here is
    quantized exactly once, by MXQuant, at its natural block scale. The two seams that used to be
    chosen at this call site both existed to survive a requantizer that filled the element format's
    range; it no longer does (see the note where their constants used to live).
    """
    import numpy as np

    import mxiface
    from app import mxformats, mxlut, mxq_golden
    from app.mxq_golden import quantize_operand

    f = mxformats.get(dtype, where="kernel lowering")
    intermediate = f.mlir or f.name
    mnk = spec.stage_mnk()

    # A chained CODEBOOK format needs one thing a direct format does not: the table stage i's
    # requantizer writes indices into (its C book) IS the table stage i+1 reads them with (its A
    # book). Nothing on the host ever sees that intermediate -- it is produced on device -- so the
    # table has to be chosen up front, from an ESTIMATE of the output.
    #
    # An fp32 matmul is a good enough estimate, and this is not a compromise: the reference
    # generator does not estimate at all. `lut_mapping_demo.py:489` builds every C book with
    # `make_lut`, which samples torch.randn and keeps the first 16 distinct values in-format. That
    # works because the requantizer divides by the block scale BEFORE projecting, so the values a C
    # book must span are already normalized (MXQuant puts a block max in [1,2)). Estimating from
    # fp32 keeps that property and additionally shapes the 16 signposts to this kernel's data.
    #
    # Getting it WRONG costs accuracy, not correctness -- the hardware rounds to the nearest entry
    # of whatever table it is given, and the other side decodes with the same one.
    from .telemetry import Telemetry
    tel = tel or Telemetry()
    est = spec.x.numpy().astype(np.float32) if f.lut and len(spec.stages) > 1 else None
    prev_c_book = None
    if est is not None:
        # The refusal lives HERE, not in the backend. The emitter drives a codebook chain correctly
        # -- tests/selftest_formats.py proves it bit-exact against the reference's own tables. What
        # cannot be done for some formats is CHOOSING the table: see mxformats.chain_refusal.
        why = mxformats.chain_refusal(f)
        if why and not allow_lossy_chain:
            raise ValueError(
                f"cannot chain {dtype}: {why}\n"
                "        Pass allow_lossy_chain=True (--allow-lossy-chain) to run it anyway. The "
                "REFERENCE has the same behaviour -- its own FP6 chain measures 58% against exact "
                "arithmetic on its own operands -- so this is worth running deliberately, just not "
                "by accident.")
        if why:
            tel.log("warning", f"LOSSY CHAIN accepted for {dtype}: {why.splitlines()[0]}")
    stages = spec.stages                      # is_chain => every stage is a mesh matmul
    n = len(stages)

    ms: list = []
    bundles: list[dict] = []
    meta: list[dict] = []
    for i, st in enumerate(stages):
        m_, k_, n_ = mnk[st.name]
        last = i == n - 1
        out_name = "Y0" if last else f"T{i}"
        # side="b" blocks along B's own K axis and returns [K/32][N] scales -- the layout the
        # device indexes as b_off = group * N + col. No transpose here: the operand entry point
        # answers that question, and it is verified against the shipped baremetal headers
        # (tests/selftest_quantizer.py).
        b_codes, b_scales, b_lut = quantize_operand(
            st.weight.numpy().astype(np.float32), side="b", dtype=dtype)
        bundle: dict = {"b_codes": b_codes, "b_scales": b_scales}
        if i == 0:
            a_codes, a_scales, a_lut = quantize_operand(
                spec.x.numpy().astype(np.float32), side="a", dtype=dtype)
            bundle |= {"a_codes": a_codes, "a_scales": a_scales}
        if b_lut is not None:
            # A book: stage 0 quantizes X itself; a chained stage inherits the previous stage's C.
            a_book = a_lut if i == 0 else prev_c_book
            # C book: only meaningful when a later stage will read this output. Estimate it.
            if last:
                c_book = a_book                     # unused by a bf16 commit; the load is still made
            else:
                est = est @ st.weight.numpy().astype(np.float32)
                # pmax_shift is ESSENTIAL, not a detail. The requantizer divides by
                # 2**(floor(log2 amax) - out_pmax), so its normalized output spans
                # [2**out_pmax, 2**(out_pmax+1)) -- [16,32) for E3M2, not the [1,2) MXQuant's own
                # convention produces. A codebook built without the shift spans +-2 while the
                # hardware feeds it +-32, and every value saturates onto the top entry.
                P = mxq_golden.normalized(est, fmt=f.mxq, axis="row", pmax_shift=f.out_pmax)
                c_book = mxlut.pack_codebooks(
                    mxlut.build_codebooks(P, axis="row", fmt=f), fmt=f)
            bundle |= {"a_lut": a_book, "b_lut": b_lut, "c_lut": c_book}
            prev_c_book = c_book
        bundles.append(bundle)

        ms.append(mxiface.MatmulStage(
            m=m_, k=k_, n=n_, weight=f"W{i}", out=out_name,
            lhs="X" if i == 0 else f"T{i - 1}",
            out_dtype="bf16" if last else intermediate))
        meta.append({"stage": i, "name": st.name, "where": "mesh", "out": out_name,
                     "m": m_, "k": k_, "n": n_,
                     "out_dtype": "bf16" if last else intermediate,
                     # The OUTPUT codebook, when there is one. The MXQuant reference needs it to
                     # reproduce a chained intermediate: the requantizer projects onto this table
                     # and the next stage reads it back with the same one.
                     "fused": True})

    # Edge provenance, returned SEPARATELY from the stage records — those are serialized to
    # metrics.json and a codebook is a numpy array. Every intermediate of a fused chain is written
    # by the HARDWARE requantizer and never reaches the host, which is what the reference must
    # model; `books` is the output codebook it was projected onto, when the format has one.
    edges = {r["name"]: {"via": "requant", "books": b.get("c_lut")}
             for r, b in zip(meta, bundles)}
    iface = mxiface.chain_interface_mlir(ms, operand_fmt=dtype)
    return iface, mxiface.to_command_buffer(iface, bundles), meta, edges


def _graph_command_buffer(spec, dtype: str = DEFAULT_DTYPE) -> tuple[dict, list[dict], dict]:
    """Lower a NON-CHAIN kernel to one command buffer carrying its graph on a side channel.

    ``merlin_iface`` v0.1 cannot express attention (three live values, computed B operands, a
    softmax it has no op for), so the graph travels alongside — the same decision, for the same
    reason, as ``mx_operands`` carrying raw codes the tensor table cannot hold. Labelled as ours.
    """
    from dataclasses import asdict

    from app import mxgraph

    g = mxgraph.from_spec(spec)
    ops = mxgraph.operand_bundles(g, dtype=dtype)
    cb = {
        "commands": [],                       # nothing merlin lowers: the graph IS the program
        "tensors": {},
        "graph": {
            "steps": [{**asdict(st), "kind": st.kind} for st in g.steps],
            "shapes": {k: list(v) for k, v in g.shapes.items()},
            "uses": {k: sorted(v) for k, v in g.uses.items()},
            "leaves": sorted(g.leaves),
            "result": g.result,
        },
        "graph_operands": {k: {kk: vv for kk, vv in v.items() if vv is not None}
                           for k, v in ops.items()},
        "graph_consts": {k: v for k, v in g.consts.items()},
    }
    # Every edge of the graph lowering is drained to bf16 and re-quantized ON THE HOST
    # (mxgraph_emit._emit_uses), so the reference must model host re-quantization -- NOT the
    # hardware requantizer, which only the fused chain path uses.
    meta = [{"stage": i, "name": st.name, "where": st.kind,
             **({"m": st.m, "k": st.k, "n": st.n, "out_dtype": "bf16"}
                if st.kind == "mesh" else {"note": st.op}),
             "fused": True}
            for i, st in enumerate(g.steps)]
    # Every edge here is drained to bf16 and re-quantized ON THE HOST (mxgraph_emit._emit_uses) --
    # NOT by the hardware requantizer, which only the fused chain uses.
    edges = {st.name: {"via": "host"} for st in g.steps if st.kind == "mesh"}
    return cb, meta, edges


def _edges_without_running(spec, dtype: str, *, graphed: bool, fused: bool,
                           allow_lossy_chain: bool, tel) -> tuple[list[dict], dict]:
    """The stage records and edge map the lowering WOULD produce, without building or running.

    Used when the spike model is not selected: the mxquant model must still be told how each
    intermediate reaches the mesh (host re-quantization vs the hardware requantizer), and that is
    decided by the lowering, not by the graph shape.
    """
    if graphed:
        cb, meta, edges = _graph_command_buffer(spec, dtype)
        return meta, edges
    if fused:
        iface, cb, meta, edges = _fused_command_buffer(spec, dtype, allow_lossy_chain=allow_lossy_chain,
                                                       tel=tel)
        return meta, edges
    mnk = spec.stage_mnk()
    meta = [{"stage": i, "name": st.name, "where": "mesh",
             "m": mnk[st.name][0], "k": mnk[st.name][1], "n": mnk[st.name][2], "out_dtype": "bf16"}
            if st.on_mesh else {"stage": i, "name": st.name, "where": "host", "note": st.note}
            for i, st in enumerate(spec.stages)]
    return meta, {st.name: {"via": "host"} for st in spec.stages if st.on_mesh}


_GEOMETRY_CHECKED: set[str] = set()


def _assert_geometry(console: str, recipe, simulator: str, tel) -> None:
    """libgemmini announces its mesh size on reset; hold the recipe to it.

    ``gemmini.cc:70-71`` prints ``Gemmini extension configured with: dim = N`` every
    run. Without this check a recipe naming a mesh the loaded model does not implement
    would run anyway, the mxquant model would honour the recipe, the device would not, and
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
            f"recipe (models/spike/build_spike.py) instead of running against another one")
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
        dtype: str = DEFAULT_DTYPE, seam: str | None = None,
        allow_lossy_chain: bool = False,
        per_stage_elf: bool = False,
        build_only: bool = False, artifacts: bool = False,
        workdir: Path | None = None, results_dir: Path | None = None,
        repo: Path | None = None, telemetry=None,
        models: tuple[str, ...] | None = None, legacy_mxquant: bool = False,
        accuracy_args: dict | None = None) -> dict:
    """Build, run and grade one KernelSpec (1..N stages). Returns the run record.

    ``models`` names which models run on the recipe's machine (see ``models.NAMES``; default
    ``models.DEFAULT``): the fp32 ``reference`` always; ``spike`` builds and runs the ELF; ``mxquant``
    computes the bits it must match; ``ppa``, ``perf`` and ``accuracy`` are fail-soft extras. Without
    ``spike`` there is no VERDICT: the record carries the model's numbers and ``pass = None``.
    ``legacy_mxquant`` grades against grade/mxquant_ref.py (MXQuant's simulator, patched; recipe-blind)
    for the equivalence test; it disappears in the next PR.

    ``per_stage_elf`` forces a chain onto the one-ELF-per-matmul path it would otherwise skip. The
    two are meant to agree bit-for-bit -- fusing changes how many programs run, not what is computed
    -- so the difference between them is a debugging instrument, not a mode.
    """
    from .metrics import compare, mxquant_only
    from .report import make_run_id, write_report
    import models as models_pkg                      # also puts the mxq submodule on sys.path
    from models import mxquant as mxquant_model
    from models import reference as reference_model
    models = tuple(models_pkg.DEFAULT if models is None else models)
    for _m in models:
        if _m not in models_pkg.NAMES:
            raise ValueError(f"unknown model {_m!r}; choose from {', '.join(models_pkg.NAMES)}")
    if build_only and "spike" not in models:
        raise ValueError("--build-only builds the ELF, which is the spike model: add spike to --models")
    legacy = None
    if legacy_mxquant:
        from . import mxquant_ref as legacy          # MXQuant-patching implementation, until PR 2
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
    from app import mxwire as w
    from app.mxq_golden import quantize_operand
    from backend import mxgemm_emit

    tel.log("setup", f"repo={repo}  "
                     f"merlin={'yes' if (repo / 'merlin' / 'merlin').exists() else 'MISSING'}  "
                     f"wired={len(wired)} paths")

    # --- the machine ---------------------------------------------------------------
    # A recipe says WHICH MX-Gemmini this is: mesh size, the per-column product and accumulator
    # precisions, the block-scale group. It drives the software model, spike and the Chisel from
    # one artifact, so a run cannot be graded against a machine that was never built.
    from config.recipe import load as load_recipe
    if recipe is None:
        recipe = load_recipe("baseline")
    if simulator not in recipe.supported_backends:
        raise RuntimeError(f"recipe {recipe.name!r} does not declare support for "
                           f"{simulator!r} (supported: {', '.join(recipe.supported_backends)})")
    seam = seam or recipe.seam
    tel.log("recipe", f"{recipe.describe()}   build_id={recipe.build_id()}  "
                      f"src={recipe.path.name}")
    for line in recipe.ladder_lines():
        tel.log("ladder", line)
    if recipe.dim != mxgemm_emit.DEFAULT_GEOMETRY["dim"]:
        raise RuntimeError(
            f"recipe dim={recipe.dim} but the backend plans for "
            f"{mxgemm_emit.DEFAULT_GEOMETRY['dim']} (mxgemm_emit.DEFAULT_GEOMETRY); "
            "pass it through cb['params'] before running a different mesh size")

    # --- the kernel ---------------------------------------------------------------
    n_stages = len(spec.stages)
    tel.log("kernel", f"{spec.describe()}   ({n_stages} stage"
                      f"{'s' if n_stages != 1 else ''})")

    errs = spec.validate(dim=recipe.dim, block=recipe.block)
    if errs:
        for e in errs:
            tel.log("illegal", e)
        raise ValueError(f"{spec.name}: {len(errs)} shape violation(s); first: {errs[0]}")

    # --- (1) the FP32 reference the hardware is graded against ----------------------
    ref_fp32 = reference_model.run(spec)["y"]
    tel.log("reference", f"fp32 torch {tuple(ref_fp32.shape)}  "
                         f"range [{ref_fp32.min():.4g}, {ref_fp32.max():.4g}]")

    if "spike" in models:
        if not build_only and not mx.available(simulator):
            tel.log("toolchain", f"NOT AVAILABLE for {simulator} -- source scripts/env.sh")
            raise RuntimeError(f"toolchain unavailable for simulator={simulator!r}")
        # Point the oracle at the model built for THIS recipe. A recipe that matches the stock
        # machine reuses the shipped libgemmini.so; anything else gets its own build, cached.
        if not build_only:
            from models.spike.build_spike import BuildError, resolve as resolve_build
            try:
                so = resolve_build(recipe)
            except BuildError as exc:
                raise RuntimeError(f"no spike model for recipe {recipe.name!r}: {exc}") from exc
            if so is not None:
                os.environ["MX_LIBGEMMINI"] = str(so)
                tel.log("build", f"recipe model {recipe.build_id()} -> {so}")
            else:
                tel.log("build", f"recipe {recipe.name!r} matches the stock build; "
                                 "using the shipped libgemmini.so")
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
    # A non-chain kernel whose every host stage is emittable becomes ONE ELF too, via the graph
    # emitter. That is D4 for attention: no numpy between two stages, ever.
    graphed = (not spec.is_chain) and not per_stage_elf and all(
        st.on_mesh or st.emittable for st in spec.stages)
    fused = spec.is_chain and not per_stage_elf
    from app import mxformats as _mxf
    _f = _mxf.get(dtype, where="kernel lowering")
    INTERMEDIATE_DTYPE = _f.mlir or _f.name
    hw = None

    if "spike" not in models:
        # No ELF, no run. The mxquant model still needs to know how each intermediate WOULD have
        # reached the mesh, which is a property of the lowering, so ask the lowering without running.
        stage_records, edges = _edges_without_running(
            spec, dtype, graphed=graphed, fused=fused, allow_lossy_chain=allow_lossy_chain, tel=tel)
        tel.log("spike", "not selected -- nothing built or run; there will be NO VERDICT")
    elif graphed:
        cb, stage_records, edges = _graph_command_buffer(spec, dtype)
        gdir = workdir / "graph"
        n_mesh = sum(r["where"] == "mesh" for r in stage_records)
        n_host = len(stage_records) - n_mesh
        tel.log("lower", f"{len(stage_records)} step(s) -> ONE command buffer via the GRAPH path "
                         f"({n_mesh} mesh, {n_host} host, {len(cb['graph_operands'])} baked operands)")
        if build_only:
            elf = mx.compile_command_buffer(cb, gdir)
            tel.log("compile", f"fused graph -> {elf} ({elf.stat().st_size} B)")
            for rec in stage_records:
                rec["elf"] = str(elf)
            return {"metrics": None, "run_dir": None, "stages": stage_records}
        res = mx.run_command_buffer(cb, workdir=gdir, simulator=simulator)
        _assert_geometry(res.get("console", ""), recipe, simulator, tel)
        raw = res["metrics"]
        fused_cycles = raw.get("cycles")
        for rec in stage_records:
            rec["elf"] = res["elf"]
            where = "mesh" if rec["where"] == "mesh" else "host"
            tel.log(where, f"{rec['stage']} ({rec['name']})"
                           + (f" {rec['m']}x{rec['n']}x{rec['k']} -> bf16" if where == "mesh"
                              else f"  {rec.get('note','')}"))
        bits = np.array(res["outputs"]["Y0"], dtype=np.uint16)
        hw = torch.from_numpy(w.bf16_bits_to_float(bits).copy()).reshape(
            *[int(v) for v in cb["graph"]["shapes"][cb["graph"]["result"]]])
    elif fused:
        iface, cb, stage_records, edges = _fused_command_buffer(
            spec, dtype, allow_lossy_chain=allow_lossy_chain, tel=tel)
        chain_dir = workdir / "chain"
        tel.log("lower", f"{len(stage_records)} stage(s) -> ONE command buffer "
                         f"({len(cb['commands'])} commands, {len(cb['tensors'])} leaf tensors)"
                         )

        if build_only:
            elf = mx.compile_command_buffer(cb, chain_dir)
            tel.log("compile", f"fused chain -> {elf} ({elf.stat().st_size} B)")
            for rec in stage_records:
                rec["elf"] = str(elf)
            return {"metrics": None, "run_dir": None, "stages": stage_records}

        res = mx.run_command_buffer(cb, workdir=chain_dir, simulator=simulator)
        _assert_geometry(res.get("console", ""), recipe, simulator, tel)
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
            peak = float(np.abs(w.fp8_e4m3_decode(codes)).max())
            rec["peak_code"] = peak
            tel.log("stage", f"{i} ({rec['name']}) {rec['m']}x{rec['n']}x{rec['k']} -> fp8   "
                             f"cycles {cyc}  "
                             f"peak code {peak:.4g}  E8M0 {scales.min()}..{scales.max()}")
            seam_cyc = raw.get(f"seam_cycles_stage{i + 1}")
            tel.log("seam", f"{i} -> {i + 1}  cycles {seam_cyc}  "
                            + ("scale transpose + code shift (hw notes 1,2)" if False
                               else "scale transpose only (hw notes 2)"))
            if artifacts:
                saved[f"{rec['out']}_codes"], saved[f"{rec['out']}_scales"] = codes, scales

        bits = np.array(res["outputs"]["Y0"], dtype=np.uint16)
        hw = torch.from_numpy(w.bf16_bits_to_float(bits).copy())
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

    if "spike" in models and not (fused or graphed):
        # The per-stage path carries every intermediate through the HOST as float, so its edges are
        # host-re-quantized like the graph path's.
        edges = {st.name: {"via": "host"} for st in spec.stages if st.on_mesh}

    if "spike" in models and not fused:
        a_codes = a_scales = None
        x_np = spec.x.numpy().astype(np.float32)

        # Reached either by a graph that cannot fuse, or by per_stage_elf on one that can. A chain
        # run this way still keeps its intermediates in the requantizer's codes+scales form -- it is
        # the host that carries them between runs, instead of the device carrying them between
        # stages. That host-mediated carry is what D4 of merlin_glue_port_plan.md abolishes; Step 6
        # removes this branch entirely once a HostStage can be emitted into the same ELF.
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
                vals[st.name] = np.asarray(st.run(*[operand(r) for r in (st.srcs or (prev,))]),
                                           dtype=np.float32)
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
                b_codes, b_scales, _bl = quantize_operand(
                    st.weight.numpy().astype(np.float32), side="b", dtype=dtype)
                ops = {"a_codes": a_codes, "a_scales": a_scales,
                       "b_codes": b_codes, "b_scales": b_scales}
            else:
                b_np = (st.weight.numpy().astype(np.float32) if st.weight is not None
                        else operand(st.rhs))
                a_c, a_s, _al = quantize_operand(operand(lhs_ref), side="a", dtype=dtype)
                b_c, b_s, _bl2 = quantize_operand(b_np, side="b", dtype=dtype)
                ops = {"a_codes": a_c, "a_scales": a_s, "b_codes": b_c, "b_scales": b_s}

            # A mesh stage commits through the requantizer only when a later stage will consume it in
            # that form; everything else reads back as bf16.
            emit_fp8 = requant_chain and not last
            iface = mxiface.matmul_interface_mlir(
                m_, n_, k_, out_dtype=INTERMEDIATE_DTYPE if emit_fp8 else "bf16",
                operand_fmt=dtype,
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
            _assert_geometry(res.get("console", ""), recipe, simulator, tel)
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
                # The requantizer writes [row][block]; the A-side scale memory indexes
                # [group][row] (a_off = group * M + row). A transpose of the SCALE BYTES is the
                # whole carry -- the codes are already in the layout the next mvin wants.
                carried[st.name] = (codes, np.ascontiguousarray(scales.T))
                if artifacts:
                    saved[f"{out_name}_codes"], saved[f"{out_name}_scales"] = codes, scales
                peak = float(np.abs(w.fp8_e4m3_decode(carried[st.name][0])).max())
                tel.log("stage", f"{i} ({st.name}) {m_}x{n_}x{k_} -> fp8   cycles {cycles}  "
                                 f"peak code {peak:.4g}  E8M0 {scales.min()}..{scales.max()}")
            else:
                bits = np.array(res["outputs"][out_name], dtype=np.uint16)
                vals[st.name] = w.bf16_bits_to_float(bits).copy()
                if last:
                    hw = torch.from_numpy(vals[st.name])
                    if artifacts:
                        saved["Y0_bits"] = bits
                tel.log("stage", f"{i} ({st.name}) {m_}x{n_}x{k_} -> bf16   cycles {cycles}")
            prev = st.name

    if build_only:
        return {"metrics": None, "run_dir": None, "stages": stage_records}

    # --- (3) grade ------------------------------------------------------------------
    if hw is not None:
        tel.log("decode", f"bf16 bit patterns -> float32 {tuple(hw.shape)}")

    # The reference is the mxquant model, not fp32 (merlin_glue_port_plan.md D7): the bits the
    # recipe's machine must produce, computed on mxq from the SPEC WE JUST RAN, in this process --
    # never from a saved artifact, which is how a stale results directory once produced a 150%
    # "divergence" that was two different seeds. The as-shipped run is MXQuant's published
    # simulator on the same ladder; it is reported as the model-vs-silicon gap, never graded.
    mxq_out = shipped = mxq_info = None
    tier = mxquant_model.TIER
    if "mxquant" in models:
        impl = legacy or mxquant_model
        ok, why = impl.available()
        not_modelled = (mxquant_model.Unavailable,) + ((legacy.MxQuantUnavailable,) if legacy else ())
        if not ok:
            tel.log("mxquant", f"UNAVAILABLE -- {why}; grading falls back to the fp32 tier")
        else:
            try:
                if legacy is not None:
                    g = legacy.simulate(spec, rtl_exact=True, dtype=dtype, edges=edges)
                    sh = legacy.simulate(spec, rtl_exact=False, dtype=dtype, edges=edges)
                    y_model, y_ship, tier = g.y, sh.y, "mxquant_rtl_exact"
                    mxq_info = {"source": "MXQuant prodacc via grade/mxquant_ref.py (legacy)",
                                "recipe_aware": False}
                else:
                    r = mxquant_model.run(spec, recipe, dtype=dtype, edges=edges)
                    y_model, y_ship, mxq_info, tier = r["y"], r["shipped_y"], r["model"], r["tier"]
            except not_modelled as exc:
                # An edge the model cannot reproduce degrades the TIER, never the numbers.
                # A confident wrong reference would be worse than none.
                tel.log("mxquant", f"NOT MODELLED -- {exc}")
                tel.log("mxquant", "grading falls back to the fp32 tier for this kernel")
            else:
                mxq_out, shipped = torch.from_numpy(y_model), torch.from_numpy(y_ship)
                if hw is not None:
                    c = mxquant_model.compare(hw.numpy(), y_model)
                    tel.log("mxquant", f"{tier}   {c['n_identical']}/{c['total']} identical  "
                                       f"max|d| {c['max_abs_diff']:.6g}"
                                       + ("" if c["identical"] else "   <-- MISMATCH, see below"))
                    cs = mxquant_model.compare(hw.numpy(), y_ship)
                    tel.log("mxquant", f"as-shipped  {cs['n_identical']}/{cs['total']} identical  "
                                       f"rel {cs['rel_fro']:.4%}  "
                                       f"(expected to differ -- this is the model-vs-silicon gap)")
                else:
                    tel.log("mxquant", f"{tier} computed: {mxq_info.get('arith')}, "
                                       f"window {mxq_info.get('window')} (no hardware to compare)")

    if hw is not None:
        metrics = compare(hw, ref_fp32, mxq_out, tol_rel_fro=tol, shipped_reference=shipped, tier=tier)
    elif mxq_out is not None:
        metrics = mxquant_only(mxq_out, ref_fp32, shipped_reference=shipped, tier=tier)
    else:
        raise RuntimeError("nothing to grade: neither spike ran nor the mxquant model was available "
                           "(--models must include spike or mxquant)")
    metrics["mxquant"] = mxq_info
    metrics["stages"] = stage_records
    if hw is not None:
        # Fused: ONE measured window spanning every stage AND every seam, which is the honest number
        # -- summing the per-stage windows would silently drop the seams. Per-stage: the sum of the
        # runs.
        metrics["total_cycles"] = (fused_cycles if fused else
                                   sum((s.get("metrics") or {}).get("cycles", 0)
                                       for s in stage_records))
        if fused:
            metrics["seam_cycles"] = sum(v for s in stage_records
                                         for k, v in (s.get("metrics") or {}).items()
                                         if k.startswith("seam_cycles_stage"))
    else:
        metrics["total_cycles"] = None
    # --- silicon cost: the PPA model. Analytical (no EDA tools, ~ms), consumes the
    # recipe alone -- independent of the spike run. Its absence is not a failure:
    # the grade stands without it, like the mxquant model.
    if "ppa" in models:
        try:
            from models.ppa.ppa import run_ppa
            ppa = metrics["ppa"] = run_ppa(recipe)
            tel.log("ppa", f"{ppa['area_um2']/1e3:.1f}k um2  {ppa['power_mw']:.1f} mW  "
                           f"{ppa['pj_per_op']:.2f} pJ/op  (post-syn model"
                           f"{'' if ppa['model']['calibrated'] else ', UNCALIBRATED dim'})")
        except Exception as exc:
            tel.log("ppa", f"UNAVAILABLE -- {exc}")

    # Predicted timeline of THIS kernel on that machine (RTL-calibrated; see
    # models/perf/perf.py for why it is not comparable to spike's functional count).
    if "perf" in models:
        try:
            from models.perf.perf import run_perf
            perf = metrics["perf"] = run_perf(recipe, metrics["stages"])
            perf["spike_functional_cycles"] = metrics.get("total_cycles")
            e = perf.get("energy")
            tel.log("perf", f"{perf['total_cycles_predicted']} cycles predicted "
                            f"({perf['total_us']:.1f} us, util {perf['utilization_pct_min']:.1f}%"
                            + (f", {e['uj_kernel']:.2f} uJ" if e else "") + ")  "
                            "[spike's count is a functional op counter, not a timeline]")
        except Exception as exc:
            tel.log("perf", f"UNAVAILABLE -- {exc}")

    # --- model-level accuracy: TinyLlama perplexity with the recipe's arithmetic in every linear
    # layer (models/accuracy). Minutes on a GPU, cached by build_id, so it runs only when named.
    if "accuracy" in models:
        try:
            from models import accuracy as accuracy_model
            acc = metrics["accuracy"] = accuracy_model.run(recipe, tel=tel, **(accuracy_args or {}))
            tel.log("accuracy", accuracy_model.line(acc).strip())
        except Exception as exc:
            tel.log("accuracy", f"UNAVAILABLE -- {exc}")

    acc = metrics["accuracy_vs_fp32_reference"]
    tel.log("grade", f"tier={metrics['tier']}  "
                     f"finite {metrics['finite']['n_finite']}/{metrics['finite']['total']}  "
                     f"cycles {metrics['total_cycles']}")
    tel.log("fp32", f"rel_fro={acc['rel_fro']:.4%}  mae={acc['mae']:.4g}  "
                    f"max_abs={acc['max_abs']:.4g}   (context only -- the cost of the format)")

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
            "kernel": spec.name, "stages": n_stages,
            "recipe": {"name": recipe.name, "build_id": recipe.build_id(),
                       "path": str(recipe.path), "hardware": recipe.hardware(),
                       "ladder": recipe.ladder(),
                       "prod": [recipe.prod_e, recipe.prod_m],
                       "acc_e": list(recipe.acc_e), "acc_m": list(recipe.acc_m)},
            # Three lowerings, and the record must name the one that ran: "fused" is the chain
            # emitter, "graph" the DAG emitter (also ONE ELF), "per_stage" the fallback.
            "lowering": "graph" if graphed else ("fused" if fused else "per_stage"),
            "elfs": 1 if (fused or graphed)
                    else len({s["elf"] for s in stage_records if s.get("elf")}),
            "m": spec.m,
            "dims": [mnk[_mesh[0].name][1]] + [mnk[s.name][2] for s in _mesh],
            "mesh_stages": len(_mesh), "host_stages": len(spec.stages) - len(_mesh),
            "operand_format": dtype, "output_dtype": "bf16",
            "intermediate_dtype": INTERMEDIATE_DTYPE if n_stages > 1 else None,
            "block_scale_group": w.BLOCK,
            "quantizer": f"mxq.block.mxgemmini@{models_pkg.mxq_commit()} rne floor=2^-23 "
                         "(wire encoding: app/mxq_golden.py)",
            "models": list(models),
            "geometry_defaults": mxgemm_emit.DEFAULT_GEOMETRY,
            "cb_params_override": None,
            "tol_rel_fro": tol,
        },
        provenance={
            "simulator": simulator, "oracle": mx.ORACLE.get(simulator),
            "repo_head": _git_head(repo), "merlin_head": _git_head(repo / "merlin"),
            "mxq_head": models_pkg.mxq_commit(),
            "gcc": str(mx.runner.gcc_path()) if hw is not None else None,
            "spike": str(mx.runner.spike_path()) if hw is not None else None,
            "libgemmini": str(mx.runner.libgemmini_so()) if hw is not None else None,
            # A path is not an identity: hash the .so that ran, so a stale build is visible in the
            # record rather than inferred later from a wrong answer.
            "libgemmini_id": _libgemmini_fingerprint(mx) if hw is not None else None,
        },
        hardware_output=hw, fp32_reference=ref_fp32, mxquant_output=mxq_out,
        metrics=metrics, artifacts=artifact_paths, telemetry=tel)

    if artifacts:
        (art_dir / "spike_result.json").write_text(
            json.dumps({"run_id": run_id, "kernel": spec.name,
                        "stages": stage_records, "metrics": metrics["accuracy_vs_fp32_reference"],
                        "oracle": mx.ORACLE.get(simulator),
                        "note": "replay on RTL: same ELFs, compare Y0_bits "
                                "(spike is derived_from_rtl=False)"}, indent=2),
            encoding="utf-8")

    return {"metrics": metrics, "run_dir": run_dir, "stages": stage_records}
