# Hardware config

Every run reads two JSON files:

- a **hardware config** (`--hw`, `config/hardware/*.json`): one accelerator. Every number in it is the
  chip. Changing one changes its `build_id`, which gives that hardware its own spike model.
- a **run config** (`--run`, `config/run/*.json`): how software drives the hardware: operand format,
  rounding, scale floor, pass threshold. Its digest is the `run_id`.

Both take a name in their folder or a path. `config/recipe.py` loads them strictly: an unknown key is
refused by name, and every run field must be written out. To change one setting, copy a file, edit
it, and pass the path.

## Shipped hardware configs

| config | product | accumulator ladder | LUT unit |
|---|---|---|---|
| `baseline` | e4m3 | m4x8 -> m5x2 -> m6x5 -> e8m7 (the tapeout) | FP6 E3M2 |
| `flat_acc4` | e4m3 | e4m4 flat | FP6 E3M2 |
| `wide_acc` | e4m3 | e8m7 flat | FP6 E3M2 |
| `narrow_prod` | e4m2 | as baseline | FP6 E3M2 |
| `lut_fp6e2m3`, `lut_fp8e5m2`, `lut_fp8e4m3` | e4m3 | as baseline | the named format |

`config/scala/JsonGemminiConfig.scala` reads the same JSON into a Chisel `GemminiArrayConfig`, so one
file defines the hardware for the RTL as well.

## Hardware config fields

Everything except the labels (`name`, `description`, `provenance`) goes into `build_id`.

| field | meaning | read by |
|---|---|---|
| `array.meshRows`, `array.meshColumns` | mesh dimension `dim` (must be equal) | all steps |
| `array.tileRows`, `array.tileColumns` | PEs per tile | the RTL only |
| `types.meshProdPrecisionList` | product precision per lane (must be uniform) | quantize, evaluate, cost |
| `types.meshAccPrecisionList` | accumulator precision down the column, one entry per lane | quantize, evaluate, cost |
| `types.*[].expWidth`, `sigWidth` | exponent bits; significand bits including the implicit bit | same |
| `types.prodFloor` | products below 2^prodFloor flush to zero (`null`: no flush) | quantize |
| `mx.scaleSize` | elements per E8M0 scale on the operands | quantize, evaluate |
| `mx.scaleSizeOut` | the requantizer's output group | evaluate |
| `mx.lut` | the LUT unit as built (the RTL's `GemminiLUTConfig`), or `null` | compile, cost |
| `scratchpad.banks`, `scratchpad.rows` | scratchpad geometry | compile, evaluate (cycle model) |
| `implementation.clock_ns`, `utilization` | target clock, placement utilization | cost |
| `provenance.*` | where each number came from | people |

## Run config fields

| field | meaning | default |
|---|---|---|
| `operand_fmt` | `fp8_e4m3`, `fp8_e5m2`, `fp8_e4m3_quad`, `fp6_e3m2`, `fp6_e2m3`, `fp4_e2m1` | `fp8_e4m3` |
| `rounding` | `rne` or `ties_away` | `rne` |
| `scale_floor` | the block maximum is floored here before the scale is taken | 2^-23 |
| `scale` (optional) | where the block scale puts the block max: `mxgemmini`, `ocp`, `ocp_below_top`, `ocp_no_clip` | `mxgemmini` |
| `reduce` | how codes are multiplied: `hardware`, `exact`, `bf16_tiles` | `hardware` |
| `allow_lossy_chain` | run a chain whose codebook cannot be chosen exactly | `false` |
| `fp32_tol` | pass threshold on relative Frobenius error vs fp32 (when no quantized model is available) | 0.15 |
| `vector` (optional) | precision of softmax / RMSNorm on the perplexity path: `null` or `"bf16"` | absent |
| `lut` (LUT formats) | how the 16-entry tables are made: `group`, `weights`, `activations`, `outputs`, `pick`, `fit` | per format in `config/run/` |

## What the compile path accepts

The perplexity path (quantize only) runs any pair whose format and reducer exist. The compile and
evaluate steps are fixed by the emitters and libgemmini; `config.recipe.check` refuses anything else
before any work:

| constant | value |
|---|---|
| mesh | 16 x 16 |
| block (scale group) | 32 |
| scratchpad | 4 banks x 4096 rows |
| rounding | `rne` |
| scale floor | 2^-23 |
| reduce | `hardware` |

A LUT format needs a hardware config whose `mx.lut` serves it and a run config with a `lut` block.
`tests/test_recipe_drift.py` holds these constants and `baseline.json` equal to libgemmini, the Chisel
source, the emitters and `rtl_exact/`.
