# models — one folder per model of the recipe's machine

A **hardware recipe** (`config/hardware/*.json`, `--hw`) defines one MX-Gemmini, and a **run recipe**
(`config/run/*.json`, `--run`) says how software drives it (see [`../config/`](../config/README.md)).
Each folder here is one model of that machine: it takes the recipes (and, where it applies, the
kernel), answers one question, and owns
the line `run_kernel.py` prints for it. `run_kernel.py --models` picks which ones run:
all five run by default; any comma list works.

| folder | question it answers | inputs | output | its line | cost |
|---|---|---|---|---|---|
| `reference/` | what does the kernel compute in plain float32? | kernel | `y` | `VERDICT` on the fp32 tier when no mxquant model is available | ms |
| `mxquant/` | which bits must the recipe's machine produce for this kernel? and, through `evaluate`, what does that arithmetic do to a language model? | kernel (or workload), recipe, operand format, edge map from the lowering | `y`, every intermediate, the as-shipped `y`; perplexity next to bf16 | `VERDICT PASS/FAIL hardware == mxquant` with spike, `MXQUANT … NO VERDICT` without | seconds |
| `spike/` | what does the functional model of that machine produce, and how many cycles does the ELF take on it? | recipe (`build_spike.py` patches and builds `libgemmini.so` per `build_id`; the same library holds the cycle model, run beside the bits with `GEMMINI_MODE=both`) | the run itself lives in `grade/pipeline.py` for now; `metrics.timing` is the cycle model's record | — | seconds to build, seconds to run |
| `ppa/` | what does the machine cost in silicon? | recipe | area, power, pJ/op | `PPA` | ms |
| `perf/` | how long does this kernel take on that machine, and what does it cost in energy? | recipe, stage shapes; after the run, spike's cycle model (`metrics.timing`) | measured cycles and utilisation on top (`merge_measured`), its own timeline under `estimate`, energy | `PERF` | ms |

`__init__.py` is the registry (`NAMES`, `DEFAULT`, `select`) and the one place the
`microscaling-quant/` submodule is put on `sys.path` (`paths()`), so every model imports `mxq` the
same way and a missing submodule fails soft with one message (`mxq_missing()`).

## The mxquant model

`mxquant/kernel.py` computes the bits the recipe's machine must produce. Operands are the bytes the
ELF carries (`compiler/operands.quantize_operand` → `wire_to_px`, codebooks included; a chained stage's
A operand comes from `requantize_chained`, the transcription of the device's requantizer). The
matmul is one `mxq.matmul.systolic` call on the recipe's arithmetic (`config/scheme.datapath`:
`MXGEMMINI(prod)`, the accumulator ladder, window = mesh dim). Before 2026-09-24 this model hardcoded
the tapeout ladder, so `wide_acc`, `flat_acc4` and `narrow_prod` could never pass VERDICT; now it
follows the recipe. The tier string is `mxquant_recipe_exact`.

**As shipped** is mxq's MXQuant mode: `block.mxquant` operand codes and the `MXQUANT` arithmetic on
the recipe's ladder. It is the published simulator's answer, reported as the model-vs-silicon gap
and never graded. Before 2026-09-24 the line came from a specific MXQuant bundle; the number changed
with the redefinition (linear/baseline: 51/4096 identical to the hardware instead of 53).

`mxquant/block.py` is MXQuant's block-quantizer API (`quantize_mx_block32`, `_broadcast_scales`,
`BLOCK`) computed by mxq with round-to-nearest-even and the block max floored at 2^-23, which is
what MXQuant's `_po2` and the hardware requantizer do. `compiler/operands.py` imports these three names
from here, so compiling a kernel no longer needs the MXQuant clone. `tests/selftest_block.py` holds
it bit-identical to MXQuant on 60 fixture cases (ties, subnormals, zero and sub-2^-23 blocks, ragged
shapes, all formats). It lives in this folder because the operand quantizer is this model's
numerics; compile shares it on purpose.

`grade/mxquant_ref.py` is the previous implementation (MXQuant's simulator patched at runtime).
`run_kernel.py --legacy-mxquant` grades with it; `tests/selftest_mxquant.py` proves the two
identical on every kernel × format (24 pairs). It goes away in the next PR.

## The perplexity path

`mxquant/workload.py` puts the recipe's Scheme (`config/scheme.scheme`, the same functions the bit
path grades spike against) into a language model's linear layers with `mxq.nn.patch` and measures
WikiText-2 perplexity the way MXQuant's published numbers were measured. `workloads.py` registers
what can run (`tinyllama`: 16 samples × 2048 tokens, seed 0, attention projections left in bf16,
`rules.py: mxquant_layers`). One subprocess per GPU (`_worker.py`, on mxq's `experiments/llm_ppl.py`,
`--gpus 0,1,2,3`), always a subprocess, so the caller never initialises CUDA. Results are cached
under `results/accuracy/<key>.json`; the key hashes everything the number depends on (model, samples,
seed, rules, hardware recipe `build_id`, operand format, rounding, scale floor, mxq commit, torch and
transformers versions). The bf16 baseline is measured once the same way. The run recipe's
`operand_fmt` picks the operand format. The four codebook (LUT) formats run through the chip's 16-entry
tables (`mxq.block.lut`, the same rule the compiler emits, `tests/selftest_codebook_mxq.py`): one per 2**G
tokens of A and per 2**G output channels of B, G and the fit from the run recipe's `lut`. A layer's output
is not requantized through a C table (no chained stage), and the record's `codebook` says so.
The run recipe's `reduce` picks how the codes are multiplied, with the same quantizers in all three cases:
`hardware` (default) is the recipe's array; `exact` is mxq's `fp64_accum`, the format's cost with a
perfect multiplier; `bf16_tiles` sums each 32-block in fp32 and folds it into the output with the
hardware's own bf16 step. Run all three and the recipe's cost splits into format, cross-block
rounding, and the ladder with the truncated product. `--rules linears_no_head` quantizes the
decoder's projections only (attention included, lm_head not). A non-default reducer is part of the
cache key and the record; the default leaves existing keys unchanged.
Which samples: `--nsamples 0` is the whole test split (165 samples at 2048), `--sequential` the first
N in order, the default the 16 seeded ones MXQuant used. The per-sample perplexity of TinyLlama at 2048
ranges from 4 to 18, so 16 samples carry about ±0.8 of sample choice: measured on the bf16 model, the
seeded 16 give 7.1989, the first 16 in order 7.9063, the whole split 8.0328. Differences between
recipes on the same samples are still meaningful; absolute numbers against the literature want the
whole split. `--hw none` measures the bf16 model alone.

```bash
.venv/bin/python -m models.mxquant --list
.venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --dry-run       # which layers get the Scheme
.venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --gpus 0,1,2,3   # run recipe "default"
.venv/bin/python -m models.mxquant --workload tinyllama --hw wide_acc --run fp4_e2m1 --gpus 0,1 --nsamples 4
.venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --run exact --gpus 0,1,2,3
.venv/bin/python -m models.mxquant --workload tinyllama --hw none --gpus 0,1,2,3 --nsamples 0        # bf16, whole split: 8.0328
```

`tests/selftest_workload.py` holds the two paths to the same bits: a `linear` kernel through
`mxquant.run` equals `MXLinear` on the same tensors, and an `mlp2` chain equals two `MXLinear` with
the bf16 accumulator between. Reproduction (2026-09-24, `tests/oracle/accuracy_baseline.json`): with
a run recipe with `rounding: ties_away` and `scale_floor: 1e-38` (mxq's defaults), and the product flush off
(`types.prodFloor: null`, as mxq was before it modelled the flush), the baseline recipe reproduces mxq's
recorded `hw_fp8` run exactly, 7.343833269506588, and the unpatched model reproduces MXQuant's bf16
number, 7.188464705866791, under the interpreter those were taken with (torch 2.9.1, transformers
4.57.3). Under the repo's `.venv` (torch 2.14.0, transformers 5.17.0) the same samples give 7.346034
and 7.198868 (with the flush on, as `baseline.json` has it: 7.346997): the bf16 model's own loss moves with the torch and transformers versions, so the
versions are part of the cache key and of every record. The standing number for the hardware's
rounding (rne, 2^-23 floor) in the `.venv` on mxq 5ee8bd4 was 7.365560971111733; the selftest checks
a cached measurement against the oracle whenever one exists for the running environment and mxq
commit. Recipes mxq cannot run (non-uniform product lists) are refused before any GPU work; without
a GPU the path reports why.

## What is not here yet

The lowering is `compiler/lower.py` and the operand encoder `compiler/operands.py`; the spike run
still lives in `grade/pipeline.py`.

## The cycle model inside spike

Since libgemmini 92fae92 the plug-in spike loads carries two models of the machine: `gemmini.cc`, the
bit-exact one every VERDICT is graded on, and `perf/` (Nico Rakela), an event-driven cycle model of the
mesh, reservation station, DMA, scratchpad and accumulator banks, L2, DRAM and the Rocket host, fed the
same RoCC command stream. The pipeline runs every spike run with `GEMMINI_MODE=both`, so one run gives
the bits and the time: the ELF's `rdcycle` reads return modelled cycles (`metrics.total_cycles`,
labelled by `metrics.cycles_source`; a record without that key holds spike's instruction counter, the
old meaning), and the model's summary and resolved parameters land in `metrics.timing` and as
`timing_summary.txt` / `timing_config.txt` next to `metrics.json`. The model knows time only, never
data, so `both` gives the same bits as `func`; `tests/selftest_timing.py` holds that on a matmul, a fused
chain and the attention graph.

Its geometry is a run-time parameter, not a build-time one: the recipe's `array.meshRows`,
`scratchpad.banks`, `scratchpad.rows` and `mx.scaleSize` are passed through `GEMMINI_PERF_SET`
(`runner.TIMING_PARAMS`) on the `mx_rocket` preset, read back from the dump and refused on a mismatch,
the way `_assert_geometry` holds the functional model to the recipe. Every other parameter is the
preset's (`perf/params/config.h`, each tagged measured / assumed / placeholder; the host side is still a
placeholder). `build_spike --list` marks a cached `.so` built before the model existed; the pipeline
refuses to record a timed run that left no summary.

| `metrics.timing` key | meaning |
|---|---|
| `cycles`, `stage_cycles` | the ELF's measured windows (modelled cycles), the kernel and per stage where the ELF reports them |
| `summary.mesh_busy_cycles`, `mesh_tiles` | cycles the array computed, and tile computes |
| `summary.host_stalled_cycles` | cycles the Rocket model stalled (RoCC queue, fences, L1 misses) |
| `summary.l2_hits/misses`, `bus_busy_cycles`, `dram_busy_cycles` | the memory system |
| `summary.load_bytes/gets`, `store_reads/puts`, `scale_bytes` | DMA traffic |
| `summary.ports` | busy cycles per scratchpad bank port (read, write) and accumulator bank |
| `summary.not_modelled`, `unparsed` | commands the model ignored; report lines this adapter does not know |
| `config.preset`, `params`, `overridden` | the parameters the model ran with, and which the recipe set |

A per-stage run (one ELF per matmul) records one summary per stage under `per_stage_summaries` and their
sum under `summary`. The number is not the one Amanda's perf model predicts for the same shapes (below):
hers is an analytical timeline of a GEMM stage calibrated on the workspace's kernels, this one times the
ELF our emitter wrote, host code included. Neither has yet been checked against RTL on these kernels.

### The `perf` record: measured on top, the timeline under `estimate`

`models/perf/perf.py: merge_measured` joins the two once spike has run. The top level of `metrics.perf`
is the measurement: `cycles` (the ELF's window, the same number as `metrics.total_cycles`), `us` at the
recipe's clock, `gops` (the kernel's ops over that window), `utilization_pct` (mesh busy cycles over the
window), and per stage `cycles` / `us` where the ELF reports a stage window (fused chains do; a graph
kernel reports one window). Amanda's timeline for the same shapes sits under `perf.estimate`
(`total_cycles_predicted`, `total_us`, `utilization_pct_min`; per stage `cycles_predicted`, `us`, `gops`,
`utilization_pct`, `phases`, `args`). What only her model gives stays at the top level: `energy`
(computed on her timeline and utilization, which `energy.basis_cycles` names), `pe_mode` /
`ops_per_pe_cycle`, `lut_loads` / `lut_tables`, `memory`. `perf.model` names both sources. Without a
spike run the measured fields are `None` and `cycles_source` says so; `python -m models.perf.perf` alone
prints the estimate. The `PERF` line reads `PERF 3766 cycles (spike cycle model) 7.5 us mesh util 27.2%
| estimate 8875 cycles, 4.78 uJ (18.4 pJ/op, on the estimate)`.

With MxGemmini-workspace at d5e82e7 or later, `perf_model --energy` gives no energy on this machine: its
own `compose_gemmini` call passes no `--system`, and the workspace's default (`rocket`) now wants the PDK
SRAM table. The record then carries `model.estimate.energy_note` with compose's reason, and the `PERF`
line has no energy. `models/ppa` is unaffected (it passes `--system radiance`).

## How ppa and perf are driven

Both are Amanda Shi's models (MxGemmini-workspace/ppa), called as she calls them. Each operand format gets the
workspace's own tokens (`models/ppa/ppa.py: FORMATS`, held equal to its `pair_modes.spec` by
`tests/selftest_ppa.py`). The hardware recipe decides which machine ppa prices: its LUT unit
(`mx.lut.projFormat`) names a machine the workspace measured whole, and the run's format sets only the stimulus.

| `mx.lut.projFormat` | machine (workspace `blocks.yaml`) | ppa settings |
|---|---|---|
| LutFP6E3M2 (baseline) | tapeout default: `mxgemmini` PE, `requantizer_new` (FP6 LUT) | `--fmtset mxgemmini --calib new --blocks-variant new` |
| LutFP8E4M3 | MxAll (`allMxFPConfig`): `mxgemmini-all` quad PE, `requantizer_all` (FP8 E4M3 LUT, all finders) | `--fmtset mxgemmini-all --calib all --blocks-variant all --lut fp8` |
| LutFP8E5M2, LutFP6E2M3, none | not measured on the current RTL | PpaError: callers record ppa as unavailable |

A format the workspace never ran on that machine's mesh (fp6_e2m3 on either; the quad arm on the tapeout PE)
is a PpaError too, with the missing point named. Nothing is extrapolated.

| operand format | ppa `--stim` | products | perf `--act/--wei` | LUT |
|---|---|---|---|---|
| fp8_e4m3 | fp8n | 1 | fp8 | no |
| fp8_e4m3_quad | fp8qn | 4 | fp8 (+ `--lut`: the quad arm) | yes |
| fp8_e5m2 | fp8e5m2 | 4 | fp8e5m2 | yes |
| fp6_e3m2 | fp6 | 4 | fp6 | yes |
| fp6_e2m3 | fp6e2m3q | 4 | fp6e2m3 | yes |
| fp4_e2m1 | fp4 | 4 | fp4 | no |

`fp8n` / `fp8qn` are the NaN-safe E4M3 kernels (mxgen reads E4M3 272..448 as NaN). With a LUT format, ppa adds
the format's products (`--products`), and perf runs
`--lut` with the chip's LUT layout: one LUT per 2**G rows of A, columns of W and rows of C (G = `run.lut.group`,
required), each load moving only the tables a stage needs, as our emitter issues them (the model does not
count the C LUT the emitter also loads for a bf16 output; the record notes it). perf's `--energy` gets the recipe's
ladder (`--acc-rows`); it has no `--prod`, so a non-e4m3 product is noted in the record, not priced there.
The kernel pipeline runs perf as measured (today's kernels, the validated configuration);
`python -m models.perf.perf --ideal [--tiles TM TN TK] [--dma-bw B] [--recipe-spad]` estimates a production GEMM.

Two more things each model reports, beside its totals and never added to them:

- `pe` (ppa) and `pe_mode` / `ops_per_pe_cycle` (perf, per stage): the PE mode the format runs in, its
  products per PE per cycle, and whether that mode was RTL-tested (`pair_modes.spec`, same format on both
  operands; mixed activation x weight pairs would need a second operand format in the run recipe, so they are
  not offered).
- `memory`: the SRAM macros the workspace's `memory_model.py` picks for the recipe's scratchpad (banks x rows x
  DIM bytes), the accumulator and the scale memory, with area, leakage, access energy and whether each meets the
  recipe clock (ppa); and perf's `--mem` read/write energy per stage (perf, on the memories perf itself uses:
  Radiance's as measured). Both need the SRAM compiler tables (`tech/sram_qrt/qrt_table.csv`), which are PDK
  data outside the workspace repo; without them `memory` is `{"available": false, "why": ...}` and nothing else
  changes. The memory energy is separate on purpose: ppa's power already contains the measured Scratchpad
  block, so adding the two would count the scratchpad twice. `tests/ppa_memfixture.py` runs both paths on an
  invented table to check the plumbing only.
