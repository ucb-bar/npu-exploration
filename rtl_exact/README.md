# rtl_exact — the MXQuant config that IS the hardware

**Use this config for any MXQuant exploration whose numbers are meant to describe MxGemmini.**
Under it, MXQuant's simulated MX matmul is **bit-identical** to the hardware — 65536/65536 elements,
max abs diff 0.0, on a real TinyLlama MLP. Verified 2026-09-06, and re-checkable in one command:

```bash
cd generators/gemmini/npu-exploration
.venv/bin/python3 rtl_exact/verify_rtl_exact.py
```
```
  as shipped         :  10.63% vs fp32    17.90% vs hardware
  rtl_exact installed:  11.59% vs fp32     0.00% vs hardware
  identical elements : 65536/65536   max abs diff 0
PASS -- MXQuant under rtl_exact/mxgemmini_rtl.json is BIT-IDENTICAL to the hardware.
```

Nothing here modifies the MXQuant repo: `rtl_datapath.install()` rebinds one method at runtime, and
`verify_rtl_exact.py` extracts MXQuant's prod/acc bundle from its own `chloe-branch-all` if that
branch is not checked out.

## Using it

```python
import eval_complete                       # MXQuant's prodacc_bundle
from rtl_exact import rtl_datapath

cfg = rtl_datapath.load_config()
rtl_datapath.install(eval_complete)        # the three hardware behaviours
sim = eval_complete.MXLinearSim(layer, cfg.mx_fmt, False, *cfg.product,
                                cfg.acc_schedule, 0, 0, window=cfg.window)
```

For MXQuant's own CLI sweeps, the accumulator schedule alone is in `acc_schedule.csv`, in the format
`load_schedule()` reads:

```bash
python eval_complete.py --mx-fmt MXFP8_E4M3 --prod-e 4 --prod-m 3 --window 16 \
       --acc-schedule .../rtl_exact/acc_schedule.csv ...
```

That gets the schedule right but **not** the three rounding behaviours below — the CLI cannot express
them today, so a run configured that way is the "as shipped" column above, not the hardware.

## What differs, and why all three matter

MXQuant already models the narrow mesh datapath: it quantizes every outer product and re-quantizes
the running sum per lane, which is structurally what `gemmini.cc` and the RTL do. Three details
differ:

| | MXQuant as shipped | MxGemmini hardware |
|---|---|---|
| **product** | `float_quantize(rounding="nearest")` | mantissa **truncation**, no exponent clamp at the product stage |
| **accumulate** | fp32 add, then quantize the **sum** to the lane | quantize **both addends** to the lane (RNE), then add exactly — so the product is quantized *twice* |
| **cross-tile** | fp32 accumulation of scaled tiles | round the scaled tile to **bf16**, accumulate in bf16 |

**They do not decompose.** Measured on the same workload, each change applied *alone*:

| config | vs fp32 | vs hardware |
|---|---|---|
| as shipped | 10.63% | 17.90% |
| + RTL product only | 9.03% | 8.83% |
| + RTL accumulate only | 8.65% | 12.55% |
| + both | 11.31% | 3.98% |
| + all three | **11.59%** | **0.00%** |

Every single change *lowers* the error versus fp32, and two of the three make the divergence from
hardware *worse*. Do not quote the effect of one of them measured with the others unmatched.

## Files

| file | what |
|---|---|
| `mxgemmini_rtl.json` | **the config** — formats, mesh geometry, the per-lane accumulator schedule, the three semantic rules, provenance (RTL and spike file:line) and the verified numbers |
| `acc_schedule.csv` | the schedule alone, in the format MXQuant's `--acc-schedule` reads |
| `rtl_datapath.py` | the three behaviours, and `install()` to put them into MXQuant's `MXLinearSim` |
| `fixture_llama_mlp.npz` | one real llama MLP's inputs plus the **hardware's** output (1.2 MB) |
| `verify_rtl_exact.py` | the gate above |
| `make_fixture.py` | regenerates the fixture from a fresh capture |

## Two things to know

**The arithmetic primitives are imported, not transcribed.** `rtl_datapath` pulls
`mx_product_quantize_trunc`, `fp_quantize_rne`, `fp_add_exact`, `q_bf16_rne` and `bf16_accum_add`
from `fp8_matmul_model` in the gemmini tree, so there is one implementation of each and it cannot
drift. Set `MXGEMMINI_ROOT` if that tree is not at `../../software/gemmini-rocc-tests`; the module
raises rather than falling back to a lookalike.

**It is slow.** The golden's `fp_quantize_rne` (exp<8) and `fp_add_exact` are exact-dyadic *scalar
Python loops* — fine for a layer, painful for a full perplexity sweep. Vectorizing them, with an
elementwise check against the scalar versions, is the obvious next step before running this at model
scale.

**What "the hardware" means here.** `Y_hw` in the fixture is `fp8_matmul_model.tiled_matmul_hwlike`,
which spike reproduces element-for-element: `bareMetalC/llama_mlp.c` reports 0/65536 mismatches on
these operands, and the same source builds for the MxGemminiRocketConfig RTL path. So a match
against this fixture is a match against the datapath, not against another Python model.

Background and the full measurement history: `../planning/llama_layer_hw_plan.md` §8.2b.

See [`../README.md`](../README.md) for install and the run command.
