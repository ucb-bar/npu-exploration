# LUT integration: audit and plan

Status 2026-10-01. Part 1 is what exists today and what was measured. Part 2 is a plan; none of it is started.

Repos read: npu-exploration `main` (`3b7ace6`), microscaling-quant (mxq) `origin/main` `43c87b8` and branch
`luts` (`ce3b42c`), MXQuant (`chooper1/MXQuant`) branches `chloe-branch-all` and `lut_integration`, gemmini
`gemmini-mx-cleanup` at `0901baa` (the `toolchain/` checkout is `04d7502`; the remote was not fetched, it may be
newer), libgemmini `8bd6b0c`, gemmini-rocc-tests (chipyard tree), MxGemmini-workspace `04ce822`.

## Why the chip has LUTs

A LUT lets an operand element travel in **4 bits** while the chip still computes with **FP6 or FP8 values**.
Plain FP4 also uses 4 bits but always means the same 16 values (0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6). A LUT
holds 16 values chosen for one group of rows or columns, from all FP6/FP8 codes. After each 32-element block is
divided by its scale, a group uses few distinct codes, so 16 well-chosen ones cover most elements.

- Gain: operands take 4 bits in DRAM, scratchpad and transfers (⅓ less than FP6, ½ of FP8); arithmetic is
  unchanged, because each index is turned back into its FP6/FP8 code before the multiply.
- Cost: a table load before each kernel (fp6 128³ on perf: 13133 vs 12786 cycles, +2.7%, a kernel not limited
  by bandwidth); the LUT unit's area; choosing the tables; and accuracy wherever a group needs more than 16
  values. **That accuracy cost is what the perplexity path does not model today.**

## Words

- **LUT** (look-up table): 16 element values. Each operand element is sent as a 4-bit **index** into it.
- **group**, **G**: one LUT serves 2^G rows of A (activations), or 2^G columns of B (weights), or 2^G rows of C
  (a requantized output). G = 1 means one LUT per pair of rows or columns, across all of K.
- **table**: the chip keeps three: A, B and C. Each holds up to 64 LUTs at once.
- **projection format**: the element format a LUT's entries are written in (E3M2, E2M3, E4M3 or E5M2), fixed
  when the chip is built.
- **host pick**: software chooses each element's index and sends it.
- **finder**: the chip's own circuit that picks the nearest LUT entry for a requantized output. It compares in a
  fixed-point form with width masks, and ties go to the lower index.
- **finder-safe values**: the values a LUT may hold so the finder can tell every pair apart (in E3M2 the masks
  wrap: 20 lands on 4, and 16 lands on 0).
- **LUT format**: one of the four operand formats sent as indices: `fp8_e4m3_quad`, `fp8_e5m2`, `fp6_e3m2`,
  `fp6_e2m3`. `fp8_e4m3` and `fp4_e2m1` send codes directly.

---

# Part 1: audit

## 1.1 Summary

The chip's LUTs are modelled correctly in one place only: our compiler (`compiler/codebook.py`) on the kernel
path, where spike and the mxquant bit model agree bit for bit. Nothing models them on the perplexity path, and
neither research branch matches the chip. The recipes describe the LUT incompletely, and the one hardware flag
we rely on, `mx.enable_lut`, is not read by the RTL.

## 1.2 Every LUT implementation found

| where | what it is | domain | grouping | matches the chip? |
|---|---|---|---|---|
| npu `compiler/codebook.py` | chip LUT rule: k-means on MX codes, finder-safe snap, host pick, finder model | block-scaled codes (right) | 2^G rows of A / columns of B / rows of C | **yes**; bit-exact on spike |
| gemmini-rocc-tests `llama_operands.py` (`build_luts`), `lut_mapping_demo.py`, `lut_golden_model.py`, `lut_fp8_matmul_model.py` | the hardware team's generators; `codebook.py` was ported from `llama_operands.py` | codes | 2^G | yes (the source of ours) |
| libgemmini `gemmini.cc` | spike's LUT load, index decode, C-LUT finder | codes | 2^G | the functional model |
| MXQuant `lut_integration`, `microxcaling/mx/level2*.py` (Abhi Pomalapally) | "level-2 quantization" inside `_quantize_elemwise_core` | **per-element mantissas** | 32-element block, or channel | **no** |
| MXQuant `chloe-branch-all`, `prodacc_bundle/lut_quantization.py` + `eval_complete.apply_lut_to_mx_weight` | a verbatim copy of `level2_scratch.py`, weights only | `mx`: codes; `per_channel`: mantissas | per (output channel, block), or channel | **no** |
| npu `rtl_exact/rtl_datapath.py` | calls MXQuant's `apply_lut_to_mx_weight` when no wire operands are given; normally takes the compiler's operands instead | — | — | only through the compiler's operands |
| npu `grade/mxquant_ref.py` | legacy tier; same rule as above ("skipping the codebook step measured 12–15% off") | — | — | only through the compiler's operands |
| mxq `luts` (Alberto Garcia, PR #1, open) | `mxq/lut/` + `mxq/block/lut.py` | block-scaled codes (right) | block, or "channel" | **no**: see 1.4 |
| mxq `main` | nothing | — | — | — |
| MxGemmini-workspace | `compose_gemmini --lut fp6|fp8|fp8e5m2` (QuantLut projection), `perf_model --lut --lut-a/-w/-c --lut-full-set` | cost only | per block of A / W / C | cost model |
| Model2MLIR | no local clone found; not checked | | | |

## 1.3 MXQuant `lut_integration`

How it works:
1. Normal MX element quantization: scale, then `_safe_lshift(out, bits-2, private_exp)`, then round.
2. If `L2_CONFIG.apply_level2` is set, each group (a 32-element block for `mx`, a channel for `per_channel`) is
   clustered by k-means into N "signposts" (16 by default, 3 iterations, quantile or random start).
3. Centroids are snapped to a hardcoded E3M2-like list; each value becomes its nearest signpost.
4. `_safe_rshift` restores each element's own exponent.

Switched on by `eval_simquant.py --apply_level2 --L2_granularity mx|per_channel --L2_num_signposts N`.

Measured with the branch's own code (random weights, `[256, 8, 32]`, block scales spread over 2^-6..2^2):

| finding | evidence |
|---|---|
| **Not a 4-bit-index LUT.** The hook runs after each element is divided by its own power of two, so it clusters mantissas and every element keeps its own exponent. On the chip an index selects a whole value. | every value the hook sees is one of 17 levels, −8..8 |
| **Weights only, by a heuristic**: "weight" means `shape[0] != 8`, the batch size hardcoded. | `elemwise_ops.py` diff |
| **Config trap**: the hook reads `microxcaling.mx.level2_config`; setting `mx.level2_config` (another module path, another singleton) silently does nothing. | 0 of 65,536 elements changed via `mx.…`; 3,400 (mx,16), 13,854 (channel,16), 14,654 (mx,8) via `microxcaling.mx.…` |
| **Snap list is not the chip's E3M2**: the active list (the last of several definitions in `level2_utils.py`) is ±0.0625…32. | `level2_utils.py:101` |
| **Reference perplexities** in `HW_complete_integration_e2e/baselines/lut_simquant_reference.csv`: no LUT 8.25, mx/16 8.375, per_channel/16 8.25, mx/8 8.9375 (TinyLlama, bf16 printing). | research numbers, not chip numbers |

MXQuant's own port says it: `lut_quantization.py:271–275`, "per_channel … in lshifted space … FP6 E3M2 values
collapse to only 15 distinct integers …, so 16 signposts is a near-no-op".

## 1.4 mxq

- `main` has no LUT code.
- `luts` (PR #1) adds `mxq/lut/{kmeans,assign,codebook}.py` and `mxq/block/lut.py`:
  `quantize(V, fmt, axis, block_size, *, base=mxgemmini, num_signposts=16, iters=3, granularity='channel')`
  returns `(P, X)` with P already the LUT values; `lut.fit(rows)` returns `(I indices, T tables)`.
- Right: it clusters block-scaled codes, and its `(P, X)` contract and `(I, T)` split are what we need.
- Different from the chip:
  - `channel` groups one table per K-block across all columns; the chip uses one per 2^G columns across all of K.
  - its snap list includes ±0.03125, 0.09375, 0.15625, 0.21875, which are not finder-safe E3M2 values (606
    unrepresentable entries across 200 LUTs, measured earlier);
  - E3M2 only, host pick by value only, no finder, no G;
  - the base quantizer's rounding and floor are not passed through;
  - 54% of elements land on a different value than the chip rule gives (measured earlier).

### 1.4b Review of mxq `luts` (ce3b42c) against the chip, measured 2026-10-02

Script: `.claude_tmp/lutaudit/review.py`; a 256x128 weight, MXFP6 E3M2, the chip rule at the run recipe's G = 1.

| | branch | chip (RTL / spike / compiler) | measured |
|---|---|---|---|
| candidate values | `E3M2_CODEBOOK`: ±0..14 with a 0.03125 subnormal step (an E3M2 with bias 4) | the E3M2 codes the finder tells apart: ±0..14, subnormal step 0.0625 | 4 branch values are not E3M2 at all (0.03125, 0.09375, 0.15625, 0.21875) and they appear in its tables |
| grouping | `mx`: a table per 32-element block; `channel`: a table per K-block index shared by every column | a table per 2^G columns of B (rows of A, C) across all of K | 1024 / 8 tables vs the chip's 64 |
| element values vs the chip | | | `mx` differs on 14861/32768 (and is more accurate than the chip can be: mean error 0.017 vs 0.043); `channel` differs on 20469/32768 |
| fit | k-means over every element, init `linspace` over sorted values, 3 passes by default | weighted k-means over distinct codes, init at quantile midpoints, up to `fit.max_iters` | |
| empty clusters / duplicates | an empty cluster becomes 0; duplicates kept | centre kept; deduped and padded with unused codes | a 2-code group: branch table has 14 zeros, 3 distinct entries; chip 16 distinct |
| formats | E3M2 only (others warn, still snap to E3M2) | four projections, each format its own finder | |
| index pick | nearest by value | host: nearest by value; chip requant: the finder, ties to the lower index | |
| G, table capacity, finder | none | G register, 64 per table, finder | |
| output | `(P, X)` only; `lut.fit` gives `(I, T)` per row | indices + packed tables per group | |
| base quantizer | `base.quantize(V, fmt, axis, block_size)`: rounding / floor not passable | the run recipe's rounding and floor | identical codes today (mxgemmini's defaults equal the chip's), but a ties_away run could not be expressed |
| repeatability | deterministic on CPU | deterministic | |

What to keep: the module layout (`mxq.lut`, `mxq.block.lut`), the `(P, X)` contract, `(I, T)` from `fit`, the torch
batching. What to replace: the value list, the grouping, the fit, empty-cluster handling, and add the finder.

## 1.5 npu-exploration

### The chip rule, `compiler/codebook.py`

1. MX-quantize first (`operands.py`), so every value is a code.
2. Per group of 2^G rows of A or columns of B: deterministic weighted 1-D k-means over the distinct codes,
   quantile start, up to 50 passes (`codebook.py:190`).
3. Snap centroids to `codebook_values(fmt)`, the finder-safe values, derived from the format's decoder and the
   finder's fixed-point form (`_fixed_point`, `_DIFF_MASK`, mirrored from `mx_fp_math.h` and the
   `*NearestFinder.scala` modules).
4. Deduplicate, pad to 16 with unused codes smallest-magnitude first, sort, pack (`pack_codebooks`).
5. Operand indices by host pick (`assign_indices`); requant-output indices by the finder model
   (`finder_indices`).
6. A chained stage's C LUT is built from an fp32 estimate of that stage's output (`lower.py:173–239`).

### Who uses LUTs

| path | LUT modelled? |
|---|---|
| kernel path: compiler → spike | yes; spike runs the same rule |
| mxquant bit path (`models/mxquant/kernel.py`) | yes: `_device_operands` takes the compiler's LUT-decoded operands, `_device_requant` the device's requantizer |
| perplexity (`python -m models.mxquant`) | **no**: full element grid, record says `"codebook": "not modelled"` (`workload.py:79,118`, `_worker.py:104`) |
| TorchAO (`config.scheme.mxq_config`) | **no**, same as perplexity |
| ppa | LUT PE settings chosen **by format**, with a fixed `--lut fp8` |
| perf | `--lut` chosen by format; LUT sharing from `run.lut.group` |
| graph kernels (`mxgraph_emit.py:153`) | refused: the host requantizer in `mx_host.h` encodes E4M3 only |

### Recipe fields today

| field | meaning | read by |
|---|---|---|
| hardware `mx.enable_lut` | "the LUT unit is present" | recorded by ppa and perf; `check` warns when false and a LUT format is used. **The RTL ignores it (1.6).** |
| run `operand_fmt` | picks a LUT format | everything |
| run `lut.group` | G | perf only; the kernel path refuses G ≠ 1 |
| run `lut.source` | `"data"` or a .json | **nothing** |
| run `lut.pick` | `host` / `hardware`, for perplexity | **nothing** |

### Hardcoded LUT facts in models and compiler (to be removed)

| file:line | constant | should come from |
|---|---|---|
| `compiler/codebook.py:28` | `LUT_SIZE = 16` | hardware (`raddrWidth`) |
| `compiler/codebook.py:190` | 50 k-means passes, quantile start | run (`lut.fit`) |
| `compiler/codebook.py` `_fixed_point`, `_DIFF_MASK` | finder form, keyed by operand format | hardware projection format (the finder belongs to the build) |
| `compiler/formats.py:168` | `LUT_GRANULARITY = 1` | run (`lut.group`) |
| `compiler/targets/.../mxgemm_emit.py:107` | `lut_g = LUT_GRANULARITY` | run (`lut.group`) |
| `compiler/targets/.../mxgemm_emit.py:726–728` | loads `m>>G`, `n>>G` LUTs, no cap | hardware (`numEntries` = 64 per table) |
| `config/recipe.py:56` | `KERNEL_LUT_GROUP = 1` | a check against the hardware's G register, not a constant |
| `config/scheme.py:47` | `CODEBOOK` set | stays (a format property), but "is this format served?" comes from the hardware |
| `models/perf/perf.py:60` | `DEFAULT_LUT_GROUP = 1` | run (`lut.group`) |
| `models/ppa/ppa.py:46` | `CALIBRATION_LUT … "--lut", "fp8"` | hardware projection format |
| libgemmini `gemmini.cc:77–79` | 2048 LUTs per table | hardware (`numEntries`); spike must fault where the RTL would |

## 1.6 The chip

Mechanics (libgemmini `8bd6b0c`, RTL `0901baa`):
- `MX_LOAD_LUT(addr, num_luts, sel, entry_bits)` loads LUTs into table A (sel 1), B (sel 0) or C (sel 2) and sets
  the runtime `lut_en` (`gemmini.cc:1130`). `MX_LUT_DISABLE` clears it.
- An operand index selects `table[(row >> G)][index]` (`gemmini.cc:1455`, `:1195` for B by column).
- A requantized output is rounded to the element format, then the finder picks the nearest entry in its
  row group's C LUT (`gemmini.cc:1517–1564`).
- G comes from `gemmini_mxquant_config_mvout` rs2[15:0] (`gemmini.cc:1099`).

Build-time settings, `GemminiLUTConfig` (`MxConfigFragments.scala:48`):

| parameter | meaning | stock `standaloneMxFPConfig` |
|---|---|---|
| `lut` | the unit exists (`Some`) or not (`None`) | `Some(GemminiLUTConfig())` |
| `projFormat` | `LutFP6E3M2`, `LutFP6E2M3`, `LutFP8E4M3`, `LutFP8E5M2` | `LutFP6E3M2` |
| `rdataWidth` | entry width: 6 (FP6) or 8 (FP8, required for FP8 projection) | 6 |
| `raddrWidth` | log2 entries per LUT | 4 (16) |
| `numEntries` | LUTs per table, A/B/C | 64, 64, 64 |
| `numBits` | write-word width = 16 × `rdataWidth` | 96, 96, 96 |
| `lutUpdateRegularityWidth` | width of the G register | 16 |
| `actCodeWidth`, `weiCodeWidth` | per-operand deprojected width (asymmetric builds) | 0 (= `rdataWidth`) |
| `enable_lut` | declared in `GemminiConfigs.scala:116` | **read by nothing** (checked at `04d7502` and `0901baa`) |

Other builds: `allMxFPConfig` and the `e4m3Lut…` configs use `LutFP8E4M3`, 8-bit entries; one config uses
`LutFP6E2M3`; `e4m3SingleNoLutMxFPConfig` has `lut = None`.

Runtime settings, per kernel: LUT contents (any 16 codes per LUT), `lut_en`, `config_ex` format codes, `altfmt`
and `uselut`, and G. Operand indices are the host's choice; output indices are the finder's.

`config/scala/JsonGemminiConfig.scala` already reads an `mx.lut` block with `numBits`, `numEntries`,
`rdataWidth`, `raddrWidth`, `lutUpdateRegularityWidth` (and `enable_lut`), but not `projFormat`. Our Python
recipe loader does not accept `mx.lut`.

## 1.7 Mismatches, ranked

1. **`enable_lut` is a dead flag.** The stock chip has a LUT unit (E3M2 projection), yet `baseline.json` says
   `false`, which is why `check` warns on every LUT run.
2. **The projection format decides which LUT formats a build serves**, and no recipe says it. Spike runs all
   four LUT formats on any recipe.
3. **No LUT-count cap.** The emitter loads `m>>G` / `n>>G` LUTs per table; the RTL holds 64, spike 2048. A LUT
   kernel with M or N above 128 (at G = 1) would pass on spike and overflow on RTL. A risk, not a seen failure.
4. **Perplexity and TorchAO do not model LUTs**, so LUT-format perplexities are optimistic.
5. **`lut.source` and `lut.pick` are parsed but unread.**
6. **ppa prices the LUT by operand format with a fixed `--lut fp8`**, not by the build's projection.
7. **Neither research branch matches the chip** (1.3, 1.4).

What works: compiler, spike and the mxquant bit path agree on G = 1, the index rule, the finder-safe values and
the finder's tie-break; LUT kernels are bit-exact on spike.

---

# Status 2026-10-02: Part A (the refactor into the recipes) done on branch `lut-recipes`

Done, proven bit-identical (numbers below): steps 1, 3 and 7 of the plan, for the values the compiler implements
today. Not done: moving the chip rule into mxq (see the end of this section).

**Proof** (`.claude_tmp/npu/capture_lut.sh` before on main 31c753a, after on the branch; `compare_lut.py`):
sweep 20/20 runs identical; all 12 LUT kernels (linear, mlp2, mlp3 x four LUT formats, each on the build that
serves it) bit-exact on spike before and after, with identical outputs; compile 144/144 files identical
(main.c, expected.npy, operands.npz); llama_mlp and llama_attention identical; ppa and perf identical but for
the recorded `enable_lut`; command buffers differ only by `params.lut_group` / `lut_tables`. All gates,
rtl_exact, ppa, perf, torchao and the drift test (which now holds each recipe's `mx.lut` to its Scala config)
pass. baseline's build_id moved (7e2a442ca9f1a819 -> b9e05c7855dfec41); the four oracle perplexities
re-measured under it are unchanged to the last digit (7.363709878432006, 7.232781411936018,
7.266006542462204, 7.3469974682860535).

**G is now a real setting:** G = 0 and G = 2 run bit-exact on spike for fp6_e3m2 and fp8_e5m2, linear, mlp2
and mlp3. A 256x64x256 fp6 linear is refused at G = 1 (its B table would need 128 LUTs; the build holds 64)
and passes 65536/65536 at G = 2.

**Where every LUT setting now lives**

| setting | file | read by |
|---|---|---|
| the LUT unit (`projFormat`, `rdataWidth`, `raddrWidth`, `numEntries`, `numBits`, `lutUpdateRegularityWidth`, `actCodeWidth`, `weiCodeWidth`) | hardware `mx.lut` (required; `null` = no unit) | `check`, emitter (each table's capacity) |
| G | run `lut.group` | compiler (codebooks, indices, decode, requant), emitter, perf |
| how tables are made (`weights`, `activations`, `outputs`, `pick`) | run `lut` | parsed; one implemented value each (`data`, `data`, `estimate`, `host`) |
| k-means passes (`fit.max_iters`; `method` kmeans, `init` quantile) | run `lut.fit` | compiler |

`config.recipe.lut_settings(hw, run)` hands the compiler `codebook.Settings(group, max_iters)`;
`config.recipe.emitter_params(hw, run)` puts `lut_group` and `lut_tables` in `cb["params"]`. No function has a
default for either; a LUT format without them raises.

**Removed:** `formats.LUT_GRANULARITY`, `recipe.KERNEL_LUT_GROUP`, the `Lut` defaults, the G != 1 refusal, the
`enable_lut` warning, `perf.DEFAULT_LUT_GROUP`, the `g=` / 50-pass defaults in `codebook.py`, the
`None` fallback in `grade/pipeline.py`, and rtl_exact's MXQuant LUT fallback (now refused). Tests that built
LUT runs in code load the recipe files (`tests/fixtures/luts.py`).

**Different from the plan, and why**
- `codebook.LUT_SIZE = 16` stays: it is the wire's 4-bit index (`formats` bits=4), a chip fact. `check` refuses a
  build whose `raddrWidth` is not that width.
- The finder stays keyed by the operand format: every format a projection serves uses that format's own finder
  (QuantLut.scala:105-155), and `check` refuses the formats it does not serve, so a projection-keyed finder
  would choose the same function.
- The 64-LUT capacity is checked in the emitter, which is the one place that knows each stage's M and N.
- A direct format still writes G = 1 into `config_mvout` (`mxgemm_emit.NO_LUT_G`): spike reads G only on a
  LUT decode or the finder, and 1 keeps every direct program byte-identical.
- The run `lut` block accepts only what is implemented; the plan's other values (`top16`, `calibrated`, files,
  `finder`) are refused by name until a later step implements them.
- `mx.enable_lut` is kept (set `true`, the RTL default) and recorded; nothing decides from it.
- Not done: moving `codebook.py`'s rule into mxq. It is not a pure move: mxq works on values and has no element
  codes or decoders, which the rule needs (`codebook_values`, `_fixed_point`, `finder_indices`), and
  Alberto's open PR #1 already owns `mxq/lut/`. Where it goes is the Alberto question below; the compiler keeps
  the rule until then.

# Status 2026-10-04: the chip's LUT rule in mxq, on both paths (branch `LUT_implementation`)

mxq `LUT_implementation` (5554904, 34d0831, on main 2dc9073) adds `mxq.lut` and `mxq.block.lut`:
- the chip's LUT rule, ported from `compiler/codebook.py` and vectorised;
- `Scheme.rows`, so MXLinear keeps the 2^G tokens of a group in one call;
- `MXQConfig.lut`.

npu `LUT_implementation` (on origin/main e35a313):
- `compiler/codebook.py` calls `mxq.lut`. The old numpy rule is frozen in `tests/fixtures/codebook_numpy.py` as the oracle.
- The perplexity path quantizes A and B of a LUT format through `block.lut`.
- The record's `codebook` is the settings (G, max_iters; no C table; table capacity not enforced).

Alberto's `luts` rule (`mx`/`channel` grouping, fixed E3M2 snap list, empty clusters becoming 0) is replaced by the chip's. His layout is kept.

## How it is proved (each link against the one before it)

| link | check | result |
|---|---|---|
| L1 spike → compiler rule | LUT kernels on spike | 12/12 PASS 4096/4096 (unchanged) |
| L3 mxq.lut == compiler rule | `selftest_codebook_mxq.py`: tables, pick, finder, values, decode; 4 formats × G 0/1/2 × 11 operands (random + TinyLlama), CPU and CUDA | 0 failures; a mutation (max_iters 2) gives 426 failures |
| L4 bit path through mxq | capture vs `lutrec_after` | 12 LUT kernels identical, 144/144 compile files, sweep, llama graphs, ppa, perf identical |
| L5 perplexity == kernel operands | `selftest_lut_layer.py`: A and B (P, X) == `wire_to_px(quantize_operand)`, matmul equal, MXLinear chunk invariance | 36 cases, 0 failures |
| LUT off | the 6 non-LUT perplexities under the new pin, re-run (fresh keys) | all identical to every digit (below) |

LUT off, before → after: bf16 7.198867975181179, default 7.363709878432006, exact 7.232781411936018, bf16_tiles
7.266006542462204, ties_away 7.3469974682860535, fp4_e2m1 18.813015866588295. Every one is identical.

## Perplexity with the chip's LUT (TinyLlama, 16 × 2048, seed 0, bf16 7.1989)

Measured on mxq 649d8bd: the same two LUT commits before they were rebased onto mxq main 2dc9073. That rebase adds only the anchor/adder tree speedups and MXQConfig compiling by default; the worker uses neither.

| format (build) | full grid (LUT off) | LUT on A+B, G=0 | G=1 | G=2 | LUT on weights only, G=1 |
|---|---|---|---|---|---|
| fp6_e3m2 (baseline) | 7.4749 | 10.1444 | 9.9041 | 9.6604 | 8.3714 |
| fp6_e2m3 (lut_fp6e2m3) | 7.6051 | 10.0731 | 10.0465 | 10.0512 | 8.3655 |
| fp8_e5m2 (lut_fp8e5m2) | 7.4358 | not measured (OOM) | 9.8494 | 9.5955 | not measured (OOM) |
| fp8_e4m3_quad (lut_fp8e4m3) | 7.3637 | not measured (OOM) | 9.9359 | 10.0578 | 8.1880 |

- **The expectation in the plan was wrong.** It said +0.3 to +1.6 over the full grid. The measured cost at G=1 is
  **+2.4 to +2.7**:
  - the weight tables add about +0.8 to +0.9 (the weights-only column);
  - the activation tables add about +1.5 more.
- MXQuant's published LUT runs were weights-only with a table per 32-block (+0.27). That is a much finer table
  than the chip's one table per 2^G whole rows.
- G does not order the cost. G=2 is lower than G=1 for fp6_e3m2 and fp8_e5m2, and higher for the quad format.
- The three OOM cells failed because another job held most of the shared GPUs' memory. They still need a run.
- "weights only" is a diagnostic script (`.claude_tmp/npu/ppl_wonly.py`), not a recipe mode: the chip has no
  such mode.

Open:
- Projection ≠ format at the same width (fp8_e5m2 tables on a LutFP8E4M3 unit) is still accepted on both paths. Is
  that what the RTL does?
- Most of the cost is in the activation tables. Does the chip really fit them on the host per 2^G tokens?

# Part 2: plan (revised 2026-10-01)

## Goal

Every path (kernel, bit model, perplexity, TorchAO, ppa, perf) models LUTs the same way, and **every LUT setting
is written in one of the two recipe files**. Example: the hardware recipe `lut_fp8e4m3.json` (an FP8 E4M3 LUT
build) with the run recipe `fp8_e4m3_quad.json` gives a compiled kernel, its expected bits, its cost and a
perplexity that includes the 16-entry restriction, from those two files alone.

## Rules

1. **Every LUT setting is in a recipe file.** When a recipe has a LUT block, the loader requires every key in it.
   The code has no default for any of them.
2. **No overrides.** No command-line flag, environment variable or code path changes a LUT setting. Tests and
   sweeps load recipe files; nothing builds a LUT run in code.
3. **The code keeps only facts of the chip**, never choices: the 4-bit index, each finder's arithmetic, which
   wire code each operand format uses. Each is looked up by a recipe value and cites its RTL or spike source. A
   recipe that asks for something the code cannot do is refused, never quietly adjusted.
4. **One LUT rule** (in mxq), used by every path.
5. **Nothing outside npu-exploration and mxq changes**: no RTL, no spike, no Scala reader, no workspace.
6. Bit-identical where nothing should change; measured where something does.

## Where the interface changes

| where | change | breaks anything? |
|---|---|---|
| hardware recipe JSON | new `mx.lut` block in all four recipes; `enable_lut` stays and is set `true`, matching the RTL | no; `build_id` changes |
| hardware recipe JSON | three new recipes for the other LUT builds (below) | no |
| run recipe JSON | the `lut` block gets its final keys; required for a LUT format | no recipe file (none has a `lut` block); `_parse_lut` and one test change |
| run recipe JSON | four new run recipes, one per LUT format | no |
| `check(hw, run, purpose)` | new refusals (below); the "G must be 1" refusal goes | LUT runs on a build that cannot serve them are refused |
| `Hardware`, `Lut` dataclasses | `Hardware.lut` added; `Lut` fields change and lose their defaults | callers that built `Lut()` in code (tests only) |
| `MXQConfig` | LUT fields added | no |
| perplexity record | `"codebook": "not modelled"` replaced by the LUT settings used | no |
| command lines | **none** | — |

## Hardware recipe: `mx.lut`

```json
"mx": {
  "scaleSize": 32,
  "scaleSizeOut": 32,
  "enable_lut": true,
  "lut": {
    "projFormat": "LutFP6E3M2",
    "rdataWidth": 6,
    "raddrWidth": 4,
    "numEntries": [64, 64, 64],
    "numBits": [96, 96, 96],
    "lutUpdateRegularityWidth": 16,
    "actCodeWidth": 0,
    "weiCodeWidth": 0
  }
}
```

The names are the RTL's own `GemminiLUTConfig` fields; the values are copied from the Scala config the recipe
mirrors. `"lut": null` means the build has no LUT unit (`lut = None`).

| field | meaning | read by |
|---|---|---|
| `projFormat` | the projection format: which finders the requantizer has | `check` (formats served), the finder in the LUT rule, ppa `--lut` |
| `rdataWidth` | bits per LUT entry (6 or 8) | the emitter's `MX_LOAD_LUT` entry width, packing, `check` |
| `raddrWidth` | log2 entries per LUT | the LUT rule's entry count; `check` refuses anything but 4 (the instruction set's 4-bit index) |
| `numEntries` | LUTs each table A/B/C holds | `check` (LUT count per kernel), perf |
| `numBits` | write-word width, must be 16 × `rdataWidth` | `check` |
| `lutUpdateRegularityWidth` | width of the G register | `check` (largest G) |
| `actCodeWidth`, `weiCodeWidth` | code widths on asymmetric builds; 0 = `rdataWidth` | `check`; refused if non-zero until the compiler supports asymmetric LUT builds |
| `enable_lut` | kept as a copy of the RTL field | recorded only; nothing decides from it |

**Formats each projection serves**, one table in `config/recipe.py`, read from the RTL's finder selection
(`QuantLut.scala:105–155`, gemmini `04d7502`):

| `projFormat` | finders built | LUT formats served |
|---|---|---|
| `LutFP6E3M2` (stock) | E3M2 | `fp6_e3m2` |
| `LutFP6E2M3` | E2M3 | `fp6_e2m3` |
| `LutFP8E5M2` | E5M2 (altfmt 1), E3M2 (altfmt 0) | `fp8_e5m2`, `fp6_e3m2` |
| `LutFP8E4M3` | E4M3, E5M2, E3M2, E2M3 | all four |

This covers the finder (outputs). That the input decode serves the same set is still to confirm with Nicolas.

**Hardware recipes**

| recipe | mirrors | LUT unit |
|---|---|---|
| `baseline`, `flat_acc4`, `narrow_prod`, `wide_acc` | `standaloneMxFPConfig` | `LutFP6E3M2`, 6-bit |
| new `lut_fp8e4m3` | `allMxFPConfig` (= `e4m3LutMxFPConfig`) | `LutFP8E4M3`, 8-bit |
| new `lut_fp8e5m2` | `e5m2MxFPConfig` | `LutFP8E5M2`, 8-bit |
| new `lut_fp6e2m3` | `e2m3OnlyMxFPConfig` | `LutFP6E2M3`, 6-bit |

Every LUT format then has a build that serves it, so no run that works today is left without a recipe.

## Run recipe: `lut`

```json
"lut": {
  "group": 1,
  "weights": "data",
  "activations": "data",
  "outputs": "estimate",
  "pick": "host",
  "fit": {"method": "kmeans", "init": "quantile", "max_iters": 50},
  "calibration": null
}
```

| field | meaning | values |
|---|---|---|
| `group` | G: one LUT per 2^G rows of A, columns of B, rows of C | 0 ≤ G, within the G register, dividing the tile |
| `weights` | where B tables come from | `"data"` (fitted from the weights) or a path to a .json of tables |
| `activations` | where A tables come from (inputs the host quantizes) | `"data"` (k-means on each input), `"top16"` (the 16 most frequent codes of each input), `"calibrated"`, or a .json path |
| `outputs` | where C tables come from (outputs the chip requantizes) | `"estimate"` (k-means on an fp32 run of this input through the chain: what `lower.py` does today), `"calibrated"`, or a .json path |
| `pick` | who picks activation indices | `"host"` (nearest by value) or `"finder"` (the projection's finder, as the chip would) |
| `fit` | how a table is fitted | `method: "kmeans"`, `init: "quantile"`, `max_iters` |
| `calibration` | data for `"calibrated"` tables | `{"nsamples", "seed"}` when any table is calibrated, else `null` |

Both paths read every field. Calibration data is the path's own: the kernel path's input generator at
`calibration.seed`; the perplexity path's workload training split. A file of given tables is checked against
the build: every entry must be a code the build's finder can tell apart.

**What each setting needs, and whether a deployed chip could do it.** Every setting fills the same
tables the chip loads with `MX_LOAD_LUT`, so the bits stay 1-to-1 with the chip whichever is chosen. They differ
in what data they use:

| setting | needs | deployable? | recorded as |
|---|---|---|---|
| `weights: data` | the weights, once, offline | yes | deployable |
| `activations: data` | this input, k-means on the host per input (over ≤ 64 distinct codes, weighted by count) | yes, at host cost | deployable, host work counted |
| `activations: top16` | this input, one counting pass on the host | yes, at host cost | deployable, host work counted |
| `activations` / `outputs: calibrated` | separate calibration inputs, once; tables then fixed | yes | deployable |
| a .json file (e.g. tables built without data, like the hardware team's `make_lut`, `lut_mapping_demo.py:489`) | nothing at run time | yes | deployable |
| `outputs: estimate` | this input **and** an fp32 run of the whole chain before the chip runs | **no**: a best case | `deployable: false` |

Every record carries `deployable` and the host fitting work (elements counted, k-means points × passes) as
counts. A perplexity quoted for a design uses deployable settings only.

The four new run recipes write today's compiler behaviour: `group` 1, `data` / `data` / `estimate`, `host`,
k-means from quantiles for up to 50 passes.

## `check(hw, run, purpose)`

Refused on both the kernel and the perplexity path:
- a LUT format on a build with `"lut": null`;
- a LUT format the build's projection does not serve;
- a LUT format with no `lut` block, or a `lut` block on a direct format;
- `group` beyond the G register;
- `raddrWidth` ≠ 4, `numBits` ≠ 16 × `rdataWidth`, non-zero code widths;
- `calibration` missing when a table is `"calibrated"`, or written when none is.

Kernel path and perf's as-measured model only: a stage whose `m` or `n` is not a multiple of 2^G, or whose
`m >> G` or `n >> G` exceeds `numEntries` for that table (spike would accept up to 2048; the RTL holds 64).

## Hard-coded values and overrides removed

| file:line | today | after |
|---|---|---|
| `config/recipe.py:56` | `KERNEL_LUT_GROUP = 1` | removed |
| `config/recipe.py:181–186` | `Lut` defaults `source="data"`, `group=1`, `pick="host"` | no defaults |
| `config/recipe.py:378–379` | refuses G ≠ 1 | replaced by the checks above |
| `compiler/formats.py:168` | `LUT_GRANULARITY = 1` | removed; `run.lut.group` |
| `compiler/operands.py:545, 605` | `g = formats.LUT_GRANULARITY` | `run.lut.group` |
| `compiler/codebook.py:28` | `LUT_SIZE = 16` | `2 ** hw.lut.raddrWidth` |
| `compiler/codebook.py:101, 203, 236` | `g=formats.LUT_GRANULARITY` default arguments | required arguments |
| `compiler/codebook.py:190` | 50 k-means passes, quantile start | `run.lut.fit` |
| `compiler/codebook.py` `_fixed_point`, `_DIFF_MASK` | finder chosen by operand format | chosen by `hw.lut.projFormat` + altfmt, as `QuantLut.scala` does (moves to mxq) |
| `compiler/lower.py:173–239` | C tables always from the fp32 estimate | `run.lut.outputs` |
| `mxgemm_emit.py:107` | `lut_g = LUT_GRANULARITY` default | required field from the run |
| `mxgemm_emit.py:726–728` | entry width from the operand format, no LUT-count cap | entry width `hw.lut.rdataWidth`; count checked against `numEntries` |
| `grade/pipeline.py:304` | `lut_group = run.lut.group if run.lut else None` | `run.lut.group`, required |
| `models/perf/perf.py:58–60, 88, 312` | `DEFAULT_LUT_GROUP = 1` fallback | `run.lut.group`; LUT on/off from the build |
| `models/ppa/ppa.py:46, 126` | `--lut fp8` for every LUT format; LUT hardware chosen by operand format | `--lut` from `projFormat`; LUT hardware from `hw.lut` |
| `models/mxquant/workload.py:79, 118`, `_worker.py:104` | `"codebook": "not modelled"` | the settings used |
| `rtl_exact/rtl_datapath.py:236–239` | MXQuant's `apply_lut_to_mx_weight` when no wire operands are given | LUT formats only through the compiler's wire operands; that fallback refused for them |
| `tests/selftest_compile.py:73, 125`, `selftest_workload.py:186–195`, `selftest_scheme.py:119`, `selftest_torchao.py:122, 141` | LUT runs built in code | load the run recipe files |

Stays in code, as chip facts: `formats.py`'s per-format wire code, altfmt and 4-bit index (`gemmini.cc`), each
finder's comparison arithmetic (`mx_fp_math.h`, `*NearestFinder.scala`), the projection → workspace `--lut` name.

## Steps

### Step 1 — Both recipes carry every LUT setting (detail)

**Required:** nothing new.

**Provided:**
- `config/recipe.py`:
  - `mx.lut` parsed with every key required; `Hardware.lut`;
  - the formats-served table;
  - the run `lut` block with its final keys, every key required, no defaults;
  - the `check` refusals above; `KERNEL_LUT_GROUP` and the G ≠ 1 refusal removed.
- `config/hardware/`: the four recipes gain the stock block and `enable_lut: true`; three new LUT-build recipes.
- `config/run/`: `fp8_e4m3_quad.json`, `fp8_e5m2.json`, `fp6_e3m2.json`, `fp6_e2m3.json`.
- `tests/test_recipe_drift.py`: each hardware recipe's `mx.lut` equals its Scala config's `GemminiLUTConfig`.
- Tests that built LUT runs in code load these files instead. The capture sweep runs each LUT format on a build
  that serves it.
- Nothing but `check` reads the new fields yet.

**Gate:**
- `selftest_scheme`: parsing, every refusal, the formats-served table.
- Capture before/after: every kernel, graph, ppa and perf output identical apart from `build_id` and the recorded
  `enable_lut`. LUT-format runs that move to a new build are listed.
- All gates and `rtl_exact` pass.

**Portable:** every later step reads LUT settings from `hw.lut` and `run.lut`, and nothing else.

### Later steps

2. **mxq round:** the chip LUT rule moves into mxq (fit, pick by value or by finder, the finder per projection,
   the codes a finder can tell apart), every setting an argument with no default; the compiler calls it. Gate:
   every LUT kernel identical; each finder matches the chip's exhaustively against a C oracle.
3. **Compiler and emitter read the recipes**; the constants above go. Gate: identical at the new run recipes'
   values; G = 2, given-table, calibrated, finder-pick and fp6-on-an-8-bit-build kernels pass bit-exact on spike.
4. **Perplexity models LUTs** from the recipes. Gate: one Linear's LUT bits on the perplexity path equal the
   kernel path's, for every LUT format; perplexity and run time measured.
5. **TorchAO:** `mxq_config(hw, run)` fills the LUT fields. Gate: `quantize_` equals `patch`, eager and compiled.
6. **ppa and perf** take LUT presence and `--lut` from `hw.lut`, G from `run.lut`; perf reports the host fitting
   work for `data` / `top16` separately from the chip's cycles. Gate: pinned invocations; every changed number
   explained.
7. **rtl_exact and grade:** LUT formats only through wire operands. Gate: `rtl_exact` identical.
8. **Measurements:** for each LUT format on each build that serves it, and each table setting (`data`,
   `top16`, `calibrated`, a data-free file, and `estimate` as the best case): perplexity, host fitting work,
   chip cycles, area, deployable. The deployable default is then chosen from this table, not before.
9. **Docs:** field tables; the MXQuant findings (1.3) written up for its authors.

## Validation

Checks 1–5 are pass/fail and tie each layer to spike; 6 ties our perplexity harness to MXQuant's; 7 only flags
surprises.

| # | check | ties | exists? |
|---|---|---|---|
| 1 | each finder equals the C oracle `mx_fp_math.h` (spike's) exhaustively | LUT rule ↔ spike's finder | new (step 2) |
| 2 | LUT kernels bit-exact on spike, every format × build × table setting | compiler ↔ spike | today's settings yes; extended in step 3 |
| 3 | emitter bit-exact against the hardware team's own tables (`lut_mapping_demo.py`) | compiler ↔ their reference | yes (`selftest_formats`) |
| 4 | one Linear's LUT bits on the perplexity path equal the kernel path's, every format and table setting | perplexity ↔ spike | new (step 4) |
| 5 | identity: every group has ≤ 16 distinct codes ⇒ LUT bits equal non-LUT bits, for a kernel and a whole-model perplexity | the LUT step alone | new |
| 6 | our framework with MXQuant's no-LUT setup (MXFP6 E3M2, their product/accumulator, 16 samples, seed 0) vs their 7.662 | our harness ↔ theirs | new; tolerance measured |
| 7 | our chip LUT's perplexity cost vs MXQuant's +0.27 (16 entries) and +1.61 (8 entries); investigate if ours is smaller, since the chip's rule is stricter | sanity only | new (step 8) |

**MXQuant's LUT results** (`chloe-working-branch-backup` `c8404d4`, `HW_complete_integration_e2e/baselines/`,
TinyLlama, MXFP6 E3M2, seed 0) are not a target: different algorithm (1.3), weights only, one table per 32-element
block, random 3-pass k-means.

| setting | 16 samples | full test set |
|---|---|---|
| no LUT | 7.662 | 8.491 |
| `mx`, 16 entries | 7.929 (+0.27) | 8.834 (+0.34) |
| `per_channel`, 16 entries | 7.702 (+0.04) | 583.0 (broken) |
| `mx`, 8 entries | 9.269 (+1.61) | missing |

Its log header (`results/logs/fp6_2_lut.log`) says `prod=e4m3 acc=fixed e8m7`, while `run_lut_parallel.sh`
passes `--prod-e 8 --prod-m 23`; which one produced the CSVs is to confirm before check 6.

## Is software 1-to-1 with the chip?

Bits: yes. Whatever tables a setting picks are loaded into the chip, and the chip computes what our model
computes (spike and the mxquant bit path agree bit for bit). What is not 1-to-1:

| | status |
|---|---|
| `outputs: estimate` | uses data a deployed chip never has; recorded `deployable: false` |
| host fitting time for `data` / `top16` | counted as work, not as time on the host CPU |
| A and B tables restricted to finder-safe codes | only C tables need it; stricter than the chip, never looser. Measure before changing (step 8) |
| the finder model | matches spike, which mirrors the RTL; never compared to RTL simulation (out of scope) |
| asymmetric code widths | refused by `check` |
| LUTs switched off between kernels, different formats per layer | not modelled: one setting per run |
| LUT formats on graph kernels | refused (`mxgraph_emit.py:153`) |
| input decode per projection | only the finder side was read from RTL; open question |

## Not changed

RTL, spike and libgemmini, `config/scala/JsonGemminiConfig.scala`, Amanda's workspace, MXQuant, every command
line.

Note: Nicolas's Scala reader accepts five of the `mx.lut` keys and not `projFormat`, `actCodeWidth` or
`weiCodeWidth`. It already refuses keys our recipes carry today (`mx.scaleSizeOut`, `types.prodFloor`), and it
treats `"lut": null` as "keep the base config's LUT", not "no LUT unit". Our flows don't use it; tell Nicolas.

## Open questions

| for | question |
|---|---|
| Nicolas | Does each build's input decode serve the same formats as its finder (table above)? |
| Nicolas | Is the tapeout build `standaloneMxFPConfig` (`LutFP6E3M2`)? |
| Nicolas | The Scala reader differences in the note above. |
| Amanda | Can ppa price a build with no LUT unit? Which `--lut` goes with `LutFP8E5M2` and `LutFP6E2M3`? |
| Alberto | Should his `luts` branch adopt the chip rule, or should we keep both? |
| team | In full models, which Linear inputs does the host quantize (`activations`) and which does the chip requantize (`outputs`)? |
