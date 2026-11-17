# MicroMX projection kernels

The `mxgemmini` package provides mixed-precision projections and real
transformer-layer examples for the DIM16 BF16-PE image. It supports TinyLlama,
Qwen2.5-7B and Llama-2-7B layer-5 captures. Existing FP8 entrypoints, default
packing flags and Makefile `all` targets keep their behavior.

See [measured results](MICROMIX_RESULTS.md) for the current hardware summary
and remaining work.

## Projection API

Include `mx_micromix_dim16.h` before the device API to select the image's
software parameters. `mx_micromix.h` exposes `mxmm_project_fp8` and
`mxmm_project_bf16`; callers provide weights, output and workspace.

Weights are K×N matrices divided into contiguous K32 bands:

| Policy | FP4 E2M1 | FP6 E3M2 | FP8 E4M3 |
|---|---:|---:|---:|
| `mix50` | 50% | 25% | 25% |
| `mix625` | 62.5% | 25% | 12.5% |

These are fixed allocations. FP6 occupies one byte per element. FP4 packs
adjacent weight columns and adjacent activation rows into the low/high nibbles.
Block scales use `floor(log2(amax))` without a format-maximum adjustment, and
conversion rounds to nearest with ties to even. BF16 input uses exponent zero
for an all-zero block; FP8 input supplies its existing E8M0 scales.

FP8 input can be row-major (`block=0`) or blocked. A positive block size must
be divisible by 32 and divide K; the layer dispatcher checks this before
packing. Output, workspace, weight groups and weight scales need 64-byte base
alignment. Workspace is at least `mxmm_workspace_bytes(M,K)` bytes. Callers
must provide valid buffer capacities, finite inputs and nonoverlapping storage;
the basic projection API does not check arbitrary alias spans.

`mxmm_integer_pack` enables the integer encoder. Its compatibility spelling is
`mxmm_round2_pack`; both refer to the same flag and default to zero.
`mxmm_fast_pack` defaults to one. Flags, loop state and counters are local to
each translation unit, so set them in the translation unit making the calls.

The software E4M3FN codec and the image's FP8 encoding domain differ at large
magnitudes. The physical MAC maps codes 121–127 and 249–255 to NaN; codes
120/248 represent finite 256. Captured, block-normalized operands avoid the
unsupported range. Encoder parity tests and RTL code-times-one probes do not
establish physical correctness for every possible encoded operand pair.

## Measured profiles

`gen/micromix_profiles.json` records twelve model/regime/mix configurations,
including exact model revisions, geometry and image/run evidence. It contains
mixed-format profiles only. `mx_micromix_profiles.h` is generated from it;
including the header does not apply a profile.

```c
#include "mx_micromix_dim16.h"
#include "mx_micromix_profiles.h"

mxmm_schedule schedule;
if (mxmm_profile_apply(MX_LLM_QWEN25_7B, MXMM_PROFILE_PREFILL,
                      MXMM_PROFILE_MIX50, &schedule)) return 1;
const mxmm_profile *profile = mxmm_profile_get(
    MX_LLM_QWEN25_7B, MXMM_PROFILE_PREFILL, MXMM_PROFILE_MIX50);
```

Profile application rejects an unsupported selection, a null schedule or a
live prepared lease without changing state. Callers apply the returned
attention and cache controls separately. Decode profiles use
`NC=256, RUN=8, fence_groups=1` and optimized scalar glue (`fast_glue=1`).
Prefill uses `NC=128, RUN=4`; Llama keeps group fences enabled.

| Model | Decode mix50 | Decode mix625 |
|---|---|---|
| TinyLlama | GQA8 tiled FMA, mode 4 | GQA8 FMA, mode 1 |
| Qwen2.5-7B | GQA7 tiled FMA, mode 5 | GQA7 FMA, mode 2 |
| Llama-2-7B | MHA tiled FMA, mode 6, cache tile 16 | MHA4 FMA, mode 3, cache tile 0 |

Mode 0 is scalar attention. Mode 4 uses a two-key QK tile and a 32-key PV
block with four heads and four output lanes. Mode 5 uses a two-key QK tile
and a 16-key PV block with seven heads and two output lanes. Mode 6 uses an
eight-key QK tile and independent-head PV tiles with 16 output lanes.

The measurements apply to their recorded image and captures. Native builds
have functional replay checks; they have not been retimed merely by being
repackaged. Attention headers require strict FP and are not implicitly included
by prefill code compiled with fast-math.

## Prepared activations

`mx_micromix_prepared.h` provides API version 2 for mixed-format prefill with
M divisible by 64. `mxmp_prepare_fp8` packs an immutable source into dedicated
workspace. Two or three `mxmp_emit_fp8` calls consume it, then `mxmp_release`
ends the lease. Each emit drains before returning, and only one lease can be
live in a translation unit.

Use a new nonzero capture epoch for each layer invocation and increasing
phases within it. Input bytes, scales and weight descriptors must remain
unchanged until release. Workspace needs a 64-byte base and at least
`mxmm_workspace_bytes(M,K)` bytes.

The low-level prepare checks the first consumer. Before calling it, ensure
workspace is disjoint from every future consumer's weights and scales; an
emit-time rejection cannot undo prior packing writes. Each emit checks its
own consumer and output spans. `mx_micromix_layer.h` handles the complete
seven-projection sequence and prevalidates each shared phase before packing:
Q/K/V share one prepared source, gate/up share another, and O/down use the
canonical path. Zero-initialize its context and call `mxmm_layer_begin` once
per invocation with an increasing epoch.

When moving from VPU-managed attention to projection scale loads, preserve
select-half-1 → hardware/compiler fence → select-half-0 → fence. The supplied
image needs the idle cycle between selects to retire the previous scale half.
The prefill examples include this sequence.

## Decode row layouts

`mx_micromix_decode_rows.h` provides a separate, opt-in issue-height selector.
Mode 0 uses canonical M32 projection. Mode 1 issues 32 FP4 rows and 16
FP6/FP8 rows while retaining 32-row packing, scale pitch, workspace and output
allocation. It writes output rows 0–15 and leaves rows 16–31 untouched.

The row layout has a real-layer functional FPGA check for Qwen mix625, but no
repeated full-layer timing qualification. Every measured profile therefore
keeps `decode_issue_mode=0`. See [decode replay](MICROMIX_DECODE.md) for caller
preconditions, explicit overrides and the distinction from attention modes.

## Build and check

Initialize `software/gemmini-rocc-tests`, its `riscv-tests` submodule and
`riscv-tests/env`. The MicroMX targets do not use `rocc-software`. Set `RISCV`
to a RISC-V bare-metal toolchain
prefix. Spike also requires `dtc` on `PATH`. The matching libgemmini must support direct E3M2 accumulation and
full-BF16 small products.

```sh
make micromix-test HOST_CC=cc PYTHON=python3
make micromix-gen-test PYTHON=python3       # NumPy required
make micromix-spike-test RISCV=/path/to/riscv-tools \
  MICROMIX_LIB=/path/to/libgemmini_mxbf16_dim16.so
make micromix-attention-test HOST_CC=cc PYTHON=python3
make micromix-layer-spike-test RISCV=/path/to/riscv-tools
```

Host tests cover strict-FP and prefill fast-math packing, finite BF16 encoder
parity, numeric gates and prepared lifetime/key/overlap checks. The format test
disables x86 DAZ/FTZ to match RISC-V; numeric gates are also tested with DAZ/FTZ
enabled. Host command tests check mixed16 packing, command shapes, ordered scale
retirement and rejection before side effects, with device calls recorded
rather than executed.
Spike projection tests check 8,192 independent output words per mix, and the
layer test exercises all seven projections and future-consumer overlap guards.

`mx_micromix_numeric.h` checks full BF16 vectors at relative L2 ≤ 1%, rejects
NaN/Inf by exponent bits, and uses the residual-subtracted denominator for
layer updates. A zero reference norm requires exact numeric agreement. Bit
mismatches are reported separately. Same-mix agreement measures kernel
correctness; quantization error against an unquantized model is a separate
quality result.

## Offline data and full-layer examples

`gen/gen_micromix_weights.py` takes finite FP32 K×N arrays named
`W_q,W_k,W_v,W_o,W_g,W_u,W_d` in an NPZ. It emits a 64-byte-aligned binary,
header and SHA manifest, refuses existing outputs and bounds temporary memory
by column tiling. It checks emitted FP8 codes against the image's finite domain.
Q/K rotary half-column permutation is explicit, and Q remains unscaled. Bias,
RMS and RoPE data come from a separate capture.

```sh
python3 gen/gen_micromix_weights.py --raw layer5.npz \
  --head-dim 128 --mix mix50 --output-prefix data/layer5_mix50
```

Prefill captures contain 64 queries after a 2,048-token prefix. TinyLlama uses
the two-packed-head path; Qwen and Llama use the single-head path, with Qwen
biases and Llama's bias-free branch. The capture header supplies `E2E_*`
dimensions/offsets and `E2E_AT` into its same-stem binary. Independent same-mix
references contain complete Q/K/V/O/gate/up/down/Hout BF16 vectors.

```sh
python3 gen/build_micromix_prefill.py --model llama2_7b --mix mix50 \
  --capture /data/llama_layer.h --weights /data/layer5_mix50.h \
  --reference /data/llama_numeric_ref.h \
  --reference-manifest /data/llama_numeric_ref.json \
  --riscv /path/to/riscv-tools --target spike --out build/llama50-check
```

The builder requires a fresh output directory, verifies reference hashes and
finite values, and pins compiler inputs before and after compilation. Existing
binary objects are reused only when their `.data` matches the supplied blob.
The example runs a canonical baseline and the selected profile with fresh
cache state, checking projections, Hout and the residual update against both
independent references and the first baseline snapshot.

The legacy native loops, VPU operators and `attn_flash.c` remain available.
`mx_native_state.h` shares TU-local loop state without importing testutils into
the direct API; both legacy/native include orders have RV64 link tests.
Checkpoints and generated model blobs remain outside the source repository.
These examples evaluate one transformer layer: decode attention runs on the
scalar core and projections run on the mesh. They are not a complete
inference engine or a perplexity evaluation.
