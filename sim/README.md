# sim — simulation substrates

Everything about **producing and managing** the things we run on: RTL simulators, FPGA emulation, and
the provenance of results that come off them.

Spike, VCS and FireSim all run the **same ELF** — only the launch command differs. So *launching* a
run (`run_elf(elf, simulator=...)`) lives in `compiler/…/backend/runner.py`, a few lines per
substrate; *producing* a substrate — elaboration, model builds, bitstreams — belongs here, because it
is slow, config-managed, and has nothing to do with codegen. Adding a substrate means a build recipe
here plus a small branch in the backend's `run_elf`.

## What has actually run on RTL

The compiler path (`run_kernel.py`) has **not** run on RTL. Its C compiles for `-DMX_ROCKET` and its
instruction stream matches the baremetal tests', but no simulator is driven from here yet.

The hand-written baremetal kernels in `../../software/gemmini-rocc-tests/` do run under VCS, driven
by `generators/gemmini/run_mx_vcs.sh`. That is the working RTL path today, and the one to copy.

```bash
cd generators/gemmini
./run_mx_vcs.sh MxE4M3OnlyGemminiRocketConfig      # one config over its tests
JOBS=8 ./run_mx_vcs.sh all                         # full regression
```

## What a simulated cycle costs

Measured on this design, from `sims/vcs/mx_vcs_logs_*`: `matmul_tiled_fp8_64x64` is 106,515 cycles in
21.7 s CPU, i.e. **~4,900 sim-cycles/sec**. That number is the whole reason the llama kernels have a
small-D variant — budget with it before starting a run.

| ELF (`build_mx_rocket/bareMetalC/`) | cycles (spike) | ~VCS wall time |
|---|---|---|
| `matmul_tiled_fp8_64x64-baremetal` | 106 K | 22 s |
| `llama_attention_small-baremetal` | 2.5 M | ~8 min |
| `llama_attention-baremetal` | 12.7 M | ~45 min, and it exceeds `run_mx_vcs.sh`'s `+max-cycles=10000000` |
| `llama_mlp-baremetal` | 11.7 M | ~40 min, same cap problem |
| `llama_attention_full-baremetal` | 28.7 M | ~1.6 h, same cap. **Spike-verified only — see below.** |

The host's fp32 glue is ~99.9% of every llama figure above; the mesh itself is 2–15 K cycles. Raise
`+max-cycles` for the two full-D kernels, or run the small one.

`llama_attention_full` is the complete attention sub-layer — all 32 heads, nothing truncated, so it
grades against TinyLlama's own output rather than a reimplementation. It passes on spike in 11 s with
every mesh matmul bit-exact. **It depends on a fix to the functional model** (`mx_loop_ws_spad` now
honours `loop_ws`'s `ex_accumulate` bit, which it previously discarded), and whether the MX RTL
honours that bit is unverified — so treat it as a spike result until that is checked.
[`../planning/llama_layer_hw_plan.md`](../planning/llama_layer_hw_plan.md) §10.3 has the detail.

`llama_attention_small` is a real TinyLlama attention head — six mesh matmuls including the resident
`P@V → o_proj` seam — on a hidden size sliced to 256. It passes on spike with every mesh stage
bit-exact. See [`../planning/llama_layer_hw_plan.md`](../planning/llama_layer_hw_plan.md) §9 for what the slice
costs and how to regenerate its data header.

## Planned

* **Verilator** — build the `MxGemminiRocketConfig` model out of chipyard. The first RTL-certified
  tier: below it, a number is not a hardware claim (the contract's `citable_tier: rtl`).
* **FPGA / FireSim** — bitstream builds and pins, for whole-model runs at realistic scale.
* **Hardware provenance** — which RTL revision a result belongs to. merlin's convention is one
  registry of pinned 40-char shas verified by *content*, not branch name
  (`merlin/contract/hardware_pins.yaml`, and its `hardware-pins` skill). A result attributed to the
  wrong device is worse than no result, because it gets cited. Our contract already declares
  `rtl_sim_config: MxGemminiRocketConfig`; the pin is what says *which build of it*.

Until Verilator or VCS is driven from here, spike is a functional model — `derived_from_rtl: false`,
bootstrap only. Cycle counts from it are instruction counts, not hardware claims.

See [`../README.md`](../README.md) for install and the run command.
