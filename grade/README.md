# grade — run a kernel and judge the result

Takes a `KernelSpec` (from [`kernels/`](../kernels/README.md)), runs it on the accelerator, and
compares the result against a reference. Everything hardware-facing is a call into
[`app/`](../app/README.md) and [`compiler/`](../compiler/README.md); nothing here reimplements
quantization, lowering, or codegen.

| file | role |
|---|---|
| `pipeline.py` | the executor — one command buffer per matmul stage, chain seam, grade, record |
| `metrics.py` | RMSE · MAE · relative Frobenius · the verdict |
| `report.py` | writes `results/<run_id>/` — config, metrics, log, arrays |
| `telemetry.py` | stage logger; prints to stderr and appends to `log.jsonl` |

## What the verdict means

The verdict is **bit-identity against the quantization reference**, not a tolerance against fp32:

| tier | compares against | answers |
|---|---|---|
| rtl-exact | the reference under `rtl_exact/rtl_datapath.install()` | is the hardware correct? |
| as-shipped | the reference as normally configured | how far is the model from the silicon? |
| fp32 | `torch.matmul` | what MX costs at all — context only |

A tolerance against fp32 can only ever say "close enough". Bit-identity against a reference computed
by an *independent implementation* of the same precision schedule is a real gate, which is why it is
the one that decides PASS.

## What is recorded

Every run lands in `results/<timestamp>_<kernel>_<shape>/`: `config.json` (shapes, seed, recipe +
ladder, toolchain paths, git heads, and a hash of the model that actually ran), `metrics.json`,
`log.jsonl`, plus the hardware and fp32 outputs as `.npy`. The model hash is the point — a result
whose model cannot be identified is not evidence.

## Current results

Nine kernels, every one a single ELF, every one **bit-identical to the quantization reference under
[`rtl_exact/`](../rtl_exact/README.md)** — the configuration in which that reference reproduces the
datapath exactly:

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

For what has and has not run on RTL, see [`../sim/README.md`](../sim/README.md).

## Two design notes

**Why the golden argument is already there.** `metrics.compare(hw, fp32, golden=None)` and
`report.write_report(..., golden_model_output=None)` both accept a golden today and are threaded
through `pipeline.run`. Filling it in touches no signature, and `metrics["tier"]` flips from `"fp32"`
to `"golden"` so a weaker grade can never be mistaken for a stronger one.

**Telemetry buffers before it has somewhere to write.** The results folder is named
`<timestamp>_<kernel>_<shape>`, which is not known until the run is under way, so early events are
held in memory and flushed by `attach()`. `tests/selftest_grade.py` checks that nothing is dropped in
that window.

See [`../README.md`](../README.md) for install and the run command.
