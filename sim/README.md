# L4 — simulation substrates

Everything about **producing and managing** the things we run on: RTL simulators, FPGA emulation, and
the provenance of results that come off them.

## The split with `compiler/`

Spike, Verilator, and FireSim all run the **same ELF** — only the launch command differs. So:

| | where | why |
|---|---|---|
| *launching* a run (`run_elf(elf, simulator=...)`) | `compiler/…/backend/runner.py` | a few lines per substrate; merlin keeps it with the backend (`targets/gemmini/backend/gemmini.py` has spike + verilator branches side by side) |
| *producing* a substrate — elaboration, model builds, bitstreams | **here** | slow, config-managed, nothing to do with codegen |

Adding a substrate means a build recipe here plus a small branch in the backend's `run_elf`.

## Planned contents

- **Verilator** — build the `MxGemminiRocketConfig` model out of chipyard. The first RTL-certified
  tier: below it, a number is not a hardware claim (the contract's `citable_tier: rtl`).
- **FPGA / FireSim** — bitstream builds and pins, for whole-model runs at realistic scale.
- **Hardware provenance** — which RTL revision a result belongs to. merlin's convention is one
  registry of pinned 40-char shas verified by *content*, not branch name
  (`merlin/contract/hardware_pins.yaml`, and its `hardware-pins` skill). A result attributed to the
  wrong device is worse than no result, because it gets cited. Our contract already declares
  `rtl_sim_config: MxGemminiRocketConfig`; the pin is what says *which build of it*.

## Status

Empty for now, deliberately. The current focus is the software bridge on spike, and spike is a
functional model — `derived_from_rtl: false`, bootstrap only. Cycle counts and hardware claims start
when Verilator lands here.

## Running today

```bash
cd <chipyard-root> && source ./env.sh          # sets $RISCV
python app/mxgemm_bringup/build_fp8_64x64.py   # emit -> build -> run on spike -> report
```
