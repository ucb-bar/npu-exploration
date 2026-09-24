# grade — run a kernel and judge the result

Takes a `KernelSpec` (from [`kernels/`](../kernels/README.md)), runs the models selected with
`--models` on the recipe's machine ([`models/`](../models/README.md)), and compares spike's result
against the mxquant model. Everything hardware-facing is a call into [`app/`](../app/README.md) and
[`compiler/`](../compiler/README.md); nothing here reimplements quantization, lowering, or codegen.

| file | role |
|---|---|
| `pipeline.py` | the executor — model gating, one command buffer per matmul stage, chain seam, grade, record |
| `metrics.py` | RMSE · MAE · relative Frobenius · the verdict (`compare`), and the record of a run without hardware (`mxquant_only`) |
| `mxquant_ref.py` | the previous bit-exact reference (MXQuant's simulator patched at runtime); `--legacy-mxquant`; removed in the next PR |
| `report.py` | writes `results/<run_id>/` — config, metrics, log, arrays |
| `telemetry.py` | stage logger; prints to stderr and appends to `log.jsonl` |

## What the verdict means

The verdict is **bit-identity against the quantization reference**, not a tolerance against fp32:

| tier | compares against | answers |
|---|---|---|
| `mxquant_recipe_exact` | the mxquant model (`models/mxquant`): mxq's systolic arithmetic on the recipe's product format and accumulator ladder, fed the wire operands the ELF carries | is the hardware correct? |
| as-shipped | mxq's MXQuant mode (`block.mxquant` codes + the `MXQUANT` arithmetic) on the recipe's ladder | how far is the published simulator from the silicon? reported, never graded |
| `fp32` | `torch.matmul` | what MX costs at all — context only; the fallback when the mxquant model is unavailable or cannot model an edge |
| `mxquant_recipe_exact (no hardware)` | nothing — spike was not selected | the model's own numbers; `pass` is `None`, the report says NO VERDICT |

A tolerance against fp32 can only ever say "close enough". Bit-identity against a reference computed
by an *independent implementation* of the same precision schedule is a real gate, which is why it is
the one that decides PASS. The reference follows the recipe: `wide_acc`, `flat_acc4` and
`narrow_prod` pass VERDICT since 2026-09-24; before, the reference hardcoded the tapeout ladder and
those three could only fail.

## What is recorded

Every run lands in `results/<timestamp>_<kernel>_<shape>/`: `config.json` (shapes, seed, recipe +
ladder, the models that ran, toolchain paths, git heads incl. `mxq_head`, the quantizer that made the
operands, and a hash of the functional model that actually ran), `metrics.json` (with
`metrics["mxquant"]`: arithmetic, schedule, window, mxq commit), `log.jsonl`, plus
`hardware_output.npy` (when spike ran), `fp32_reference.npy` and `mxquant_output.npy` (when the
model ran). The hashes are the point — a result whose model cannot be identified is not evidence.

Renamed on 2026-09-24 (older results directories keep the old keys): `correctness_vs_golden_model`
→ `correctness_vs_mxquant`; `golden_model_output=` → `mxquant_output=` (`mxquant_output.npy` is now
written; the old `golden_model.npy` never was); tier `mxquant_rtl_exact` → `mxquant_recipe_exact`
(the legacy path keeps the old string). `delta_vs_mxquant_as_shipped` kept its name but its
meaning changed: as-shipped is now mxq's MXQuant mode on the recipe's ladder rather than one
specific MXQuant bundle.

## Current results

Nine kernels, every one a single ELF, every one **bit-identical to the mxquant model**
([`models/mxquant`](../models/README.md), which `tests/selftest_mxquant.py` proves identical to the
previous reference under [`rtl_exact/`](../rtl_exact/README.md) on every kernel × format):

| kernel | shape | steps | vs reference | vs fp32 |
|---|---|---|---|---|
| `linear` | 64³ | 1 mesh | 4096/4096 | 5.91% |
| `mlp2` … `mlp8` | 64³ | 2–8 mesh, fused chain | 4096/4096 | 8.97% … 22.7% |
| `attention` | 64³ | 6 mesh + 1 host | 4096/4096 | 13.62% |
| `llama_mlp` | 32×2048 | 3 mesh + 2 host | 65536/65536 | 11.98% |
| `llama_attention` | 32×2048 | 6 mesh + 4 host | 65536/65536 | 15.02% |

The two llama kernels are a real TinyLlama decoder layer — captured activations and weights,
`d_model = 2048` kept full — reproducing the hand-written `bareMetalC/llama_*.c` kernels.

All six MX formats are bit-identical as single matmuls **and** as fused chains eight stages deep:
`fp8_e4m3`, `fp8_e4m3_quad`, `fp8_e5m2`, `fp6_e3m2`, `fp6_e2m3`, `fp4_e2m1`.

Across recipes (2026-09-24): `linear`, `mlp2` and `attention` on `baseline`, `flat_acc4`, `wide_acc`
and `narrow_prod` — 12 runs, all 4096/4096 identical at tier `mxquant_recipe_exact`.

For what has and has not run on RTL, see [`../sim/README.md`](../sim/README.md).

## Two design notes

**The tier names the reference.** `metrics.compare(hw, fp32, mxquant_output, tier=...)` records
which implementation produced the reference, and `metrics["tier"]` is `"fp32"` whenever none did, so
a weaker grade can never be mistaken for a stronger one. A model that cannot reproduce an edge
(`models.mxquant.Unavailable`) degrades the tier, never the numbers.

**Telemetry buffers before it has somewhere to write.** The results folder is named
`<timestamp>_<kernel>_<shape>`, which is not known until the run is under way, so early events are
held in memory and flushed by `attach()`. `tests/selftest_grade.py` checks that nothing is dropped in
that window.

See [`../README.md`](../README.md) for install and the run command.
