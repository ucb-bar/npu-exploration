# Step 3: Evaluate

The evaluate step runs the compiled ELF on spike, with a functional model of your hardware loaded as
spike's extension, and checks the output **bit for bit** against the quantized model (step 1).

```bash
npu.py --hw baseline --model m.py:Block --only compile,evaluate
```

## The functional model

`models/spike/build_spike.py` builds `libgemmini.so` from the hardware config: one build per
`build_id`, cached. A config equal to the stock machine reuses the shipped library. The same library
carries an event-driven cycle model (mesh, DMA, scratchpad and accumulator banks, L2, DRAM, the Rocket
host), run beside the bits with `GEMMINI_MODE=both`. One run gives both the bits and modelled cycles;
the cycle model never changes the bits (`tests/selftest_timing.py`).

## What the verdict means

| tier | compared against | answers |
|---|---|---|
| `mxquant_recipe_exact` | the quantized model, fed the same wire operands the ELF carries | is the hardware correct? decides PASS |
| as shipped | mxq's MXQuant mode on the same ladder | how far the published simulator is from the silicon; reported only |
| `fp32` | `torch` in float32 | what the format costs; the fallback when `quantize` is skipped |
| no hardware | nothing (`evaluate` skipped) | the quantized model's own numbers; `NO VERDICT` |

A tolerance against fp32 can only say "close enough". Bit identity against an independent
implementation of the same precision schedule is a real gate, so it decides PASS. The tier is always
recorded, so a weaker grade cannot pass for a stronger one.

## What each run records

`results/<timestamp>_<model>_<shape>/`:

| file | contents |
|---|---|
| `config.json` | shapes, seed, both configs, steps that ran, toolchain paths, git heads, a hash of the spike model that ran |
| `metrics.json` | error vs fp32 and vs the quantized model, cycles, `timing` (cycle model summary), `ppa`, `perf` |
| `log.jsonl` | every log line |
| `hardware_output.npy`, `mxquant_output.npy`, `fp32_reference.npy` | the arrays compared |

The pipeline is `grade/pipeline.py`. The quantize and cost models do not need spike's output, so they
run in parallel with it and are joined at the grade.
