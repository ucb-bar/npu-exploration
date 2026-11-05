# Step 1: Quantize

The quantize step runs the **quantized model**: an independent implementation of your hardware's MX
arithmetic, built on the [mxq](https://github.com/chloe-wong/microscaling-quant) library
(`microscaling-quant/` submodule). It gives the exact bits the hardware must produce, and the error
those bits carry against fp32.

```bash
npu.py --hw baseline --model m.py:Block --only quantize       # seconds, no toolchain
```

## How it is built

- **Operands** are the bytes the ELF carries: `compiler/operands.quantize_operand`, with codebooks for
  LUT formats. A chained stage's input comes from `requantize_chained`, which transcribes the device's
  requantizer.
- **Arithmetic** is one `mxq.matmul.systolic` call on the hardware config's datapath
  (`config/scheme.datapath`): the product format, the accumulator ladder, window = mesh dimension.
- **Block quantizer** (`models/mxquant/block.py`): round to nearest even, block max floored at 2^-23,
  as the hardware requantizer does.

`config/scheme.py` maps the two configs onto mxq: `quantizer(hw, run)`, `datapath(hw)` and
`scheme(hw, run)`. `tests/selftest_scheme.py` holds `scheme(hw, run).matmul` bit-identical to the
extracted hardware model (`rtl_exact/mxmesh/`) on every hardware config.

**As shipped** is also reported: mxq's MXQuant mode on the same ladder, the published simulator's
answer. It measures the simulator-vs-silicon gap and is never graded.

## Hardware-exact references (`rtl_exact/`)

The quantized model is checked against two hardware-exact datapath models kept in `rtl_exact/`:

- `mxmesh/fp8.py`, `fp4.py`: the hardware team's mesh models, extracted verbatim by
  `rtl_exact/extract_model.py` (`tests/selftest_extracted.py` keeps them in step with upstream);
- `rtl_datapath.py` with `mxgemmini_rtl*.json`: the patch under which MXQuant's simulator is
  bit-identical to the hardware (`tests/verify_rtl_exact.py`, needs the MXQuant clone).

MXQuant as shipped differs from the hardware in three places, and all three are needed:

| | MXQuant as shipped | MxGemmini hardware |
|---|---|---|
| product | round to nearest | mantissa truncation, no exponent clamp |
| accumulate | fp32 add, then quantize the sum to the lane | quantize both addends to the lane (RNE), then add exactly |
| cross-tile | fp32 accumulation of scaled tiles | round each scaled tile to bf16, accumulate in bf16 |

## Formats

| `operand_fmt` | elements | LUT |
|---|---|---|
| `fp8_e4m3` | 8-bit E4M3 | no |
| `fp8_e4m3_quad` | 8-bit E4M3, four products per PE | yes |
| `fp8_e5m2` | 8-bit E5M2 | yes |
| `fp6_e3m2`, `fp6_e2m3` | 6-bit | yes |
| `fp4_e2m1` | 4-bit E2M1 | no |

LUT formats encode each element as a 4-bit index into a 16-entry table (`compiler/codebook.py`), one
table per 2^G rows of A / columns of B, with G and the fit from the run config's `lut` block.

## Whole-model accuracy: the perplexity path

The same arithmetic can be patched into a language model's linear layers (`mxq.nn.patch`) and
measured as WikiText-2 perplexity, one worker per GPU:

```bash
.venv/bin/python -m models.mxquant --list
.venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --dry-run          # which layers
.venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --gpus 0,1,2,3
.venv/bin/python -m models.mxquant --workload tinyllama --hw none --gpus 0,1,2,3          # bf16 baseline
```

The run config's `reduce` separates the costs: `hardware` is the configured array, `exact` the format
alone with a perfect multiplier, `bf16_tiles` exact inside a block and bf16 across blocks. Results are
cached under `results/accuracy/<key>.json`, keyed on everything the number depends on (model, samples,
seed, `build_id`, format, rounding, mxq commit, torch and transformers versions).
`tests/selftest_workload.py` holds this path and the kernel path to the same bits.
