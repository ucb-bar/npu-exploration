# config — the two recipes

Every run takes two JSON files:

- a **hardware recipe** (`hardware/*.json`, `--hw`): one machine. Every number in it is the chip.
  Changing one changes `build_id` and gives the machine its own functional model.
- a **run recipe** (`run/*.json`, `--run`): how software drives that machine. The operand format,
  how operands are rounded, the scale floor, which reducer the perplexity path multiplies with, and
  the kernel path's pass threshold. `run_id` is its digest.

`recipe.py` loads both, strictly. An unknown key is refused by name, and every run field must be
written out. `scheme.py` turns the pair into mxq's `Scheme`.

## Using it

```bash
.venv/bin/python run_kernel.py --list                                    # kernels, both recipe kinds
.venv/bin/python run_kernel.py --kernel linear --hw wide_acc             # run recipe "default"
.venv/bin/python run_kernel.py --kernel linear --hw baseline --run fp4_e2m1
.venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --run exact --gpus 0,1,2,3
.venv/bin/python -m models.spike.build_spike --hw <recipe> --force
```

`--config` is still accepted as another name for `--hw`. The old per-setting flags (`--dtype`,
`--rounding-mode`, `--scale-floor`, `--reduce`, `--tol`, `--allow-lossy-chain`) are refused with a
message naming the run field that replaced each. To change one setting, copy `run/default.json`,
edit that field and pass `--run <file>`.

| hardware recipe | product | accumulator ladder |
|---|---|---|
| `baseline` | e4m3 | m4x8 -> m5x2 -> m6x5 -> e8m7 (stock) |
| `flat_acc4` | e4m3 | e4m4 flat |
| `wide_acc` | e4m3 | e8m7 flat |
| `narrow_prod` | **e4m2** | same ladder as baseline |

| run recipe | differs from `default` in |
|---|---|
| `default` | nothing: what the chip does and every recorded result used |
| `exact` | `reduce: exact`: the format's cost alone (perplexity path only) |
| `bf16_tiles` | `reduce: bf16_tiles`: exact inside a block, bf16 across blocks (perplexity path only) |
| `fp4_e2m1` | `operand_fmt: fp4_e2m1` |

## Hardware recipe fields

All of these are in `build_id` except the labels (`name`, `description`, `provenance`).

| field | meaning | read by |
|---|---|---|
| `array.meshRows`, `array.meshColumns` | the mesh dimension `dim`; must be equal | mxquant (window), spike (`-DGEMMINI_DIM`), ppa (`--cols`), perf (`--rows/--cols`), emitters |
| `array.tileRows`, `array.tileColumns` | PEs per tile | nothing in Python |
| `types.meshProdPrecisionList` | per-lane product precision; must be uniform (spike has one `prod_e/prod_m`) | mxquant, spike, ppa (`--prod`) |
| `types.meshAccPrecisionList` | the accumulator ladder down the column, one entry per lane | mxquant, spike, ppa (`--rows`) |
| `types.*[].expWidth`, `sigWidth` | exponent bits; significand bits including the implicit bit (mantissa = `sigWidth - 1`) | same |
| `types.*[].count`, `isRecoded`, `pad` | passed through, not read | |
| `types.prodFloor` | a product below 2^prodFloor is flushed to zero (MxFPMul: -16); `null` for no flush | mxquant |
| `mx.scaleSize` | elements per E8M0 scale on the operands | mxquant, spike (`GROUP`) |
| `mx.scaleSizeOut` | the requantizer's output group | spike (`GROUP_OUT`) |
| `mx.enable_lut` | codebook decode hardware present | perf (`--lut`); the perplexity path refuses it |
| `accumulator.acc_read_full_width`, `acc_read_small_width` | accumulator read widths | nothing in Python |
| `scratchpad.banks`, `scratchpad.rows` | scratchpad geometry | emitters (`bank_num`, `bank_rows`) |
| `implementation.clock_ns` | target clock period | ppa (`--clock-ns`), perf (`--clock-ns`) |
| `implementation.utilization` | placement utilization | ppa (`--util`) |
| `provenance.*` | where each number was taken from | people |

## Run recipe fields

| field | meaning | default |
|---|---|---|
| `operand_fmt` | MX operand format: `fp8_e4m3`, `fp8_e5m2`, `fp8_e4m3_quad`, `fp6_e3m2`, `fp6_e2m3`, `fp4_e2m1` | `fp8_e4m3` |
| `rounding` | operand rounding: `rne` or `ties_away` | `rne` |
| `scale_floor` | the block maximum is floored here before the scale is taken | 2^-23 |
| `reduce` | how codes are multiplied: `hardware` (the recipe's array), `exact`, `bf16_tiles` | `hardware` |
| `allow_lossy_chain` | run a chain whose codebook cannot be chosen exactly | `false` |
| `fp32_tol` | kernel pass threshold on relative Frobenius error against fp32 | 0.15 |

`name` and `description` are labels and stay out of `run_id`. The perplexity cache key does not
include `allow_lossy_chain` or `fp32_tol`, because the perplexity path never reads them.

## Which path runs what

`recipe.check(hw, run, path)` refuses, before any work, what a path cannot follow. The perplexity
path (mxq alone) runs any pair whose format and reducer exist. The kernel path (emitters, spike,
the chip's requantizer) is fixed in several places, which `recipe.py` names as constants:

| constant | value | fixed by |
|---|---|---|
| `KERNEL_DIM` | 16 | the emitters' tile plan and libgemmini's `DIM` |
| `KERNEL_BLOCK` | 32 | `mx_host.h` `MX_BLOCK`, `compiler/formats.BLOCK`, spike's `GROUP` |
| `KERNEL_SCRATCHPAD` | 4 banks x 4096 rows | libgemmini `gemmini_params.h` |
| `KERNEL_ROUNDING` | `rne` | the requantizer in `gemmini.cc` and `mx_host.h` |
| `KERNEL_SCALE_FLOOR` | 2^-23 | the requantizer |

The kernel path also needs `reduce: hardware`, since it grades the chip. `tests/test_recipe_drift.py`
holds these constants and `baseline.json` equal to libgemmini, the Chisel source, the emitters and
`rtl_exact/`.

## Recipes to mxq (`scheme.py`)

| function | gives | from |
|---|---|---|
| `mxq_format(dtype)` | mxq's format name (`MXFP8_E4M3`, `MXFP4`, ...) | `run.operand_fmt` |
| `quantizer(hw, run)` | `block.mxgemmini.quantize` with block size, rounding and scale floor passed explicitly | `mx.scaleSize`, `run.rounding`, `run.scale_floor` |
| `mxgemmini(hw)` | `MXGEMMINI(prod_e, prod_m, prod_floor)` | product list, `types.prodFloor` |
| `datapath(hw)` | `(mxgemmini(hw), [(e, m) per lane], window = dim)` | plus the ladder and `dim` |
| `shipped_datapath(hw)` | the same on `MXQUANT(prod_e, prod_m)`, the as-shipped definition | same |
| `scheme(hw, run)` | an mxq `Scheme`: the quantizer for A and B, and the reducer `run.reduce` names | all of the above |

Refused with `RecipeError`, never approximated: a non-uniform product list, an accumulator list
whose length is not the mesh dimension, and at model level (`scheme()`) an `enable_lut` recipe.
The codebook formats run on their full element grid on the perplexity path, and its record says so.
`tests/selftest_scheme.py` holds `scheme(hw, run).matmul` bit-identical to the extracted hardware
model (`rtl_exact/mxmesh/fp8`) on every recipe.

## Silicon cost (ppa) and predicted performance (perf)

`models/ppa/ppa.py` maps a hardware recipe and the run's operand format onto the MxGemmini area and
power model (`../MxGemmini-workspace/ppa`, or `MX_PPA_ROOT`). The clock and utilization come from
`implementation`. The model is calibrated at 16x16 in tstech16c at tt0p8v25c; those are labels of the
calibration, not settings. Standalone: `python -m models.ppa.ppa --hw <recipe> [--run <run>] [--json]`.

`models/perf/perf.py` maps the same pair plus each stage's GEMM shape onto the performance model
(`../MxGemmini-workspace/ppa/perf/perf_model.py`). Its prediction is a full kernel timeline and is
not comparable to spike's stage cycles, which count operations. Both are recorded and labelled.
Standalone: `python -m models.perf.perf --hw <recipe> [--run <run>] --m 64 --k 64 --n 64 [--json]`.
