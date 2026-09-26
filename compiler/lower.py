"""The lowering: a KernelSpec and an operand format become ONE command buffer.

The command buffer is the dict the backend (compiler/targets/mx_gemmini_rocket/backend) emits C
from. A chain of matmuls is ``tensors`` (the leaves) plus four commands per matmul -- RES_PACK,
MATMUL_RESIDENT, COMMIT, EVICT -- with the wire bytes on ``mx_operands``; anything else is a graph
that rides on ``graph``/``graph_operands``/``graph_consts`` with an empty command list.

Three lowerings, chosen by :func:`lower`:

* ``fused``     -- a chain, one ELF; intermediates stay on device through the requantizer.
* ``graph``     -- a non-chain whose host stages the emitter knows, one ELF; every intermediate is
                   drained to bf16 and re-quantized on the host.
* ``per_stage`` -- one ELF per matmul, each fed by the previous run. It cannot be compiled ahead of
                   time; ``grade/pipeline.run`` still drives it, and :func:`lower` returns only its
                   stage records and edges.

``edges`` is what the mxquant model needs to reproduce the bits: per mesh stage, whether its input
arrived through host memory (``host``) or the hardware requantizer (``requant``, with the output
codebook for LUT formats).

Built directly, with no interface MLIR and no merlin: ``tests/selftest_lower.py`` holds
:func:`command_buffer` to the dicts merlin's parser produced (``tests/oracle/command_buffers.json``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Default operand format. Every other format is selected with --dtype and gated by
#: app/mxformats.py, which refuses one this repo has not proven end to end.
DEFAULT_DTYPE = "fp8_e4m3"


def wire_paths(repo: Path = REPO) -> list[str]:
    """Put the app and backend packages on sys.path (idempotent); ``import models`` adds mxq."""
    added = []
    for p in (repo, repo / "app", repo / "compiler" / "targets" / "mx_gemmini_rocket"):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)
            added.append(s)
    import models  # noqa: F401  -- the one place the mxq submodule (microscaling-quant/) joins sys.path
    return added


@dataclass(frozen=True)
class MatmulStage:
    """One matmul in a chain: ``lhs[m][k] @ weight[k][n] -> out[m][n]``.

    ``lhs`` is a NAME; for stage *i* > 0 it is the previous stage's ``out``, which is the whole of
    how a chain is expressed. ``out_dtype`` is ``"bf16"`` for a stage read back to the host, or an
    MX element type (``"f8E4M3FN"``) for one whose requantized output the next stage consumes.
    """

    m: int
    k: int
    n: int
    weight: str
    out: str
    lhs: str
    out_dtype: str = "bf16"


def command_buffer(stages: list[MatmulStage], bundles: list[dict] | None, *,
                   operand_fmt: str = DEFAULT_DTYPE) -> dict:
    """The chain command buffer: leaf tensors, four commands per matmul, the operand bytes.

    Only LEAF tensors appear in the table (weights, then inputs); an intermediate is named by its
    COMMIT and its shape is derived by the consumer. Element types use MLIR's builtin spellings
    (``f8E4M3FN``), which is what the emitter's format aliases are keyed on.

    ``bundles`` attaches one dict of wire bytes per stage, in order; a chained stage supplies only
    its ``b_*`` keys because its A operand is produced on device.
    """
    from app import mxformats
    elem = {f.name: (f.mlir or f.name) for f in mxformats.FORMATS.values()}
    if operand_fmt not in elem:
        raise ValueError(f"operand_fmt {operand_fmt!r} not in {sorted(elem)}")
    if not stages:
        raise ValueError("a chain needs at least one stage")
    et = elem[operand_fmt]

    produced = {st.out for st in stages}
    tensors: dict[str, dict] = {}
    for st in stages:
        tensors[st.weight] = {"shape": [st.k, st.n], "dtype": et, "role": "weight"}
    for st in stages:
        if st.lhs not in produced and st.lhs not in tensors:
            tensors[st.lhs] = {"shape": [st.m, st.k], "dtype": et, "role": "input"}

    commands: list[dict] = []
    for i, st in enumerate(stages):
        res = f"{st.weight}_res"
        commands += [
            {"opcode": "RES_PACK", "operands": {"src": st.weight, "dst": res},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": st.lhs, "rhs": res, "dst": f"acc{i}"}},
            {"opcode": "COMMIT", "operands": {"src": f"acc{i}", "dst": st.out},
             "attributes": {"epilogue": [], "output_dtype": st.out_dtype}},
            {"opcode": "EVICT", "operands": {"handle": res}},
        ]
    cb = {"abi_version": "0.1", "target": "mx_gemmini_rocket", "tensors": tensors,
          "commands": commands}
    if bundles is not None:
        import numpy as np
        cb["mx_operands"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in d.items() if v is not None} for d in bundles]
    return cb


@dataclass
class Lowering:
    kind: str                 #: "fused" | "graph" | "per_stage"
    stages: list[dict]        #: stage records (metrics.json "stages"; the perf model's input)
    edges: dict               #: {stage: {"via": "host"} | {"via": "requant", "books": ...}}
    cb: dict | None           #: the command buffer; None for per_stage


def lower(spec, dtype: str = DEFAULT_DTYPE, *, per_stage: bool = False,
          allow_lossy_chain: bool = False, warn=None) -> Lowering:
    """Lower ``spec`` in ``dtype``. ``warn(text)`` receives the one accepted-lossy-chain warning."""
    wire_paths()
    if spec.is_chain and not per_stage:
        cb, meta, edges = _fused(spec, dtype, allow_lossy_chain=allow_lossy_chain, warn=warn)
        return Lowering("fused", meta, edges, cb)
    if (not spec.is_chain and not per_stage
            and all(st.on_mesh or st.emittable for st in spec.stages)):
        cb, meta, edges = _graph(spec, dtype)
        return Lowering("graph", meta, edges, cb)
    meta, edges = _per_stage(spec)
    return Lowering("per_stage", meta, edges, None)


def _fused(spec, dtype: str, *, allow_lossy_chain: bool, warn) -> tuple[dict, list[dict], dict]:
    """Lower a WHOLE matmul chain to one command buffer.

    Only stage 0 supplies an A operand. Every later stage's A is the previous stage's requantizer
    output, produced on device and consumed in place — it never travels through the host, so it
    never appears in the ``mx_operands`` side channel (the backend refuses one that does).

    **There is no seam compensation any more**, and that is the point: every operand here is
    quantized exactly once, by MXQuant, at its natural block scale. The two seams that used to be
    chosen at this call site both existed to survive a requantizer that filled the element format's
    range; it no longer does (see the note where their constants used to live).
    """
    import numpy as np

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
        if why and warn:
            warn(f"LOSSY CHAIN accepted for {dtype}: {why.splitlines()[0]}")
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

        ms.append(MatmulStage(
            m=m_, k=k_, n=n_, weight=f"W{i}", out=out_name,
            lhs="X" if i == 0 else f"T{i - 1}",
            out_dtype="bf16" if last else intermediate))
        meta.append({"stage": i, "name": st.name, "where": "mesh", "out": out_name,
                     "m": m_, "k": k_, "n": n_,
                     "out_dtype": "bf16" if last else intermediate,
                     # The OUTPUT codebook, when there is one. The mxquant model needs it to
                     # reproduce a chained intermediate: the requantizer projects onto this table
                     # and the next stage reads it back with the same one.
                     "fused": True})

    # Edge provenance, returned SEPARATELY from the stage records — those are serialized to
    # metrics.json and a codebook is a numpy array. Every intermediate of a fused chain is written
    # by the HARDWARE requantizer and never reaches the host, which is what the reference must
    # model; `books` is the output codebook it was projected onto, when the format has one.
    edges = {r["name"]: {"via": "requant", "books": b.get("c_lut")}
             for r, b in zip(meta, bundles)}
    return command_buffer(ms, bundles, operand_fmt=dtype), meta, edges


def _graph(spec, dtype: str) -> tuple[dict, list[dict], dict]:
    """Lower a NON-CHAIN kernel to one command buffer carrying its graph on a side channel.

    The chain command buffer cannot express attention (three live values, computed B operands, a
    softmax it has no op for), so the graph travels alongside — the same decision, for the same
    reason, as ``mx_operands`` carrying raw codes the tensor table cannot hold. Labelled as ours.
    """
    from dataclasses import asdict

    from app import mxgraph

    g = mxgraph.from_spec(spec)
    ops = mxgraph.operand_bundles(g, dtype=dtype)
    cb = {
        "commands": [],                       # no chain: the graph IS the program
        "tensors": {},
        "graph": {
            "steps": [{**asdict(st), "kind": st.kind} for st in g.steps],
            "shapes": {k: list(v) for k, v in g.shapes.items()},
            "uses": {k: sorted(v) for k, v in g.uses.items()},
            "leaves": sorted(g.leaves),
            "result": g.result,
            "operand_fmt": dtype,
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


def _per_stage(spec) -> tuple[list[dict], dict]:
    """The records of the one-ELF-per-matmul path; every edge is host re-quantized."""
    mnk = spec.stage_mnk()
    meta = [{"stage": i, "name": st.name, "where": "mesh",
             "m": mnk[st.name][0], "k": mnk[st.name][1], "n": mnk[st.name][2], "out_dtype": "bf16"}
            if st.on_mesh else {"stage": i, "name": st.name, "where": "host", "note": st.note}
            for i, st in enumerate(spec.stages)]
    return meta, {st.name: {"via": "host"} for st in spec.stages if st.on_mesh}
