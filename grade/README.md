# grade — run a kernel and judge the result

Takes a `KernelSpec` (from `kernels/`), runs it on the accelerator, and compares the
result against an FP32 reference. Everything hardware-facing is a call into `app/` and
`compiler/`; nothing here reimplements quantization, lowering, or codegen.

| file | role |
|---|---|
| `pipeline.py` | the executor — one command buffer per matmul stage, chain seam, grade, record |
| `metrics.py` | RMSE · MAE · relative Frobenius · the verdict (golden slot reserved for phase 2) |
| `report.py` | writes `results/<run_id>/` — config, metrics, log, arrays |
| `telemetry.py` | stage logger; prints to stderr and appends to `log.jsonl` |

**Why the golden argument is already there.** `metrics.compare(hw, fp32, golden=None)` and
`report.write_report(..., golden_model_output=None)` both accept a golden today and are
threaded through `pipeline.run`. Phase 2 fills it in without touching a signature, and
`metrics["tier"]` flips from `"fp32"` to `"golden"` so a weaker grade can never be mistaken
for a stronger one.

**Telemetry buffers before it has somewhere to write.** The results folder is named
`<timestamp>_<kernel>_<shape>`, which is not known until the run is under way, so early
events are held in memory and flushed by `attach()`. `tests/selftest_grade.py` checks that
nothing is dropped in that window.

See the root `README.md` for commands.
