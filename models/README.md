# models — one folder per model of the recipe's machine

A **recipe** (`config/recipes/*.json`) defines one MX-Gemmini. Each folder here is one model of that
machine: it takes the recipe (and, where it applies, the kernel), answers one question, and owns
the line `run_kernel.py` prints for it. `run_kernel.py --models` picks which ones run:
`default` is `reference,mxquant,spike,ppa,perf`; `all` adds `accuracy`; any comma list works.

| folder | question it answers | inputs | output | its line | cost |
|---|---|---|---|---|---|
| `reference/` | what does the kernel compute in plain float32? | kernel | `y` | `VERDICT` on the fp32 tier when no mxquant model is available | ms |
| `mxquant/` | which bits must the recipe's machine produce for this kernel? | kernel, recipe, operand format, edge map from the lowering | `y`, every intermediate, the as-shipped `y` | `VERDICT PASS/FAIL hardware == mxquant` with spike, `MXQUANT … NO VERDICT` without | seconds |
| `spike/` | what does the functional model of that machine produce? | recipe (`build_spike.py` patches and builds `libgemmini.so` per `build_id`) | the run itself lives in `grade/pipeline.py` for now | — | seconds to build, seconds to run |
| `ppa/` | what does the machine cost in silicon? | recipe | area, power, pJ/op | `PPA` | ms |
| `perf/` | how long does this kernel take on that machine? | recipe, stage shapes | predicted cycles, utilisation, energy | `PERF` | ms |
| `accuracy/` | what does the machine's arithmetic do to a language model? | recipe | TinyLlama WikiText-2 perplexity next to bf16 | `PPL` | ~12 min on 4 GPUs, then cached |

`__init__.py` is the registry (`NAMES`, `DEFAULT`, `select`) and the one place the
`microscaling-quant/` submodule is put on `sys.path` (`paths()`), so every model imports `mxq` the
same way and a missing submodule fails soft with one message (`mxq_missing()`).

## The mxquant model

`mxquant/mxquant.py` computes the bits the recipe's machine must produce. Operands are the bytes the
ELF carries (`app/mxq_golden.quantize_operand` → `wire_to_px`, codebooks included; a chained stage's
A operand comes from `requantize_chained`, the transcription of the device's requantizer). The
matmul is one `mxq.matmul.systolic` call on the recipe's arithmetic (`config/scheme.datapath`:
`MXGEMMINI(prod)`, the accumulator ladder, window = mesh dim). Before 2026-09-24 this model hardcoded
the tapeout ladder, so `wide_acc`, `flat_acc4` and `narrow_prod` could never pass VERDICT; now it
follows the recipe. The tier string is `mxquant_recipe_exact`.

**As shipped** is mxq's MXQuant mode: `block.mxquant` operand codes and the `MXQUANT` arithmetic on
the recipe's ladder. It is the published simulator's answer, reported as the model-vs-silicon gap
and never graded. Before 2026-09-24 the line came from a specific MXQuant bundle; the number changed
with the redefinition (linear/baseline: 51/4096 identical to the hardware instead of 53).

`mxquant/block.py` is MXQuant's block-quantizer API (`quantize_mx_block32`, `_broadcast_scales`,
`BLOCK`) computed by mxq with round-to-nearest-even and the block max floored at 2^-23, which is
what MXQuant's `_po2` and the hardware requantizer do. `app/mxq_golden.py` imports these three names
from here, so compiling a kernel no longer needs the MXQuant clone. `tests/selftest_block.py` holds
it bit-identical to MXQuant on 60 fixture cases (ties, subnormals, zero and sub-2^-23 blocks, ragged
shapes, all formats). It lives in this folder because the operand quantizer is this model's
numerics; compile shares it on purpose.

`grade/mxquant_ref.py` is the previous implementation (MXQuant's simulator patched at runtime).
`run_kernel.py --legacy-mxquant` grades with it; `tests/selftest_mxquant.py` proves the two
identical on every kernel × format (24 pairs). It goes away in the next PR.

## The accuracy model

`accuracy/accuracy.py` puts the recipe's Scheme (`config/scheme.scheme`, the same functions the
mxquant model grades spike against) into TinyLlama's linear layers with `mxq.nn.patch` and measures
WikiText-2 perplexity the way MXQuant's published numbers were measured: 16 samples × 2048 tokens,
seed 0, attention projections left in bf16 (`rules.py: mxquant_layers`). One subprocess per GPU
(`_worker.py`, `--gpus 0,1,2,3`), always a subprocess, so the pipeline never initialises CUDA.
Results are cached under `results/accuracy/<key>.json`; the key hashes everything the number
depends on (model, samples, seed, rules, recipe `build_id`, operand format, rounding, scale floor,
mxq commit). The bf16 baseline is measured once the same way.

```bash
.venv/bin/python -m models.accuracy --config baseline --dry-run            # which layers get the Scheme
.venv/bin/python -m models.accuracy --config baseline --gpus 0,1,2,3
.venv/bin/python run_kernel.py --kernel linear --config wide_acc --models all --gpus 0,1,2,3
```

Reproduction (2026-09-24, `tests/oracle/accuracy_baseline.json`): with `--rounding-mode ties_away
--scale-floor 1e-38` (mxq's defaults) the baseline recipe reproduces mxq's recorded `hw_fp8` run
exactly, 7.343833269506588, and the unpatched model reproduces MXQuant's bf16 number,
7.188464705866791, when run under the interpreter those were taken with (torch 2.9.1, transformers
4.57.3). Under the repo's `.venv` (torch 2.14.0, transformers 5.17.0) the same samples give 7.346034
and 7.198868: the bf16 model's own loss moves with the torch and transformers versions, so the
versions are part of the cache key and of every record, and a perplexity is only comparable to one
taken in the same environment. The standing number for the hardware's rounding (rne, 2^-23 floor)
in the `.venv` is 7.365560971111733; `tests/selftest_accuracy.py` checks a cached measurement
against the oracle whenever one exists for the running environment. Recipes mxq cannot run at model
level (codebook formats, non-uniform product lists) are refused before any GPU work; without a GPU
the model reports UNAVAILABLE and the grade stands.

## What is not here yet

The lowering and the spike run still live in `grade/pipeline.py`; `app/mxq_golden.py` (wire
encoding, codebooks, requantizer) is still under `app/`. Both move in the next PR, with the compile
mode entry point.
