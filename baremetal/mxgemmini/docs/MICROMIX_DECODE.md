# Batch-one MicroMX decode replay

`gen/build_micromix_decode.py` evaluates one pretrained transformer layer using
eight sequential teacher-captured inputs after a 2,048-key prefix. It supports
TinyLlama-1.1B, Qwen2.5-7B and Llama-2-7B with fixed `mix50` or `mix625` K32
precision bands. Projections run on Gemmini; RMSNorm, RoPE, attention, SiLU and
residuals run on the scalar core.

Both passes use the same selected profile. They check reproducibility rather
than compare baseline and candidate speeds. Each projection and Hout vector is
checked against an independent reference, and Hout-minus-input has a separate
residual-update check. The first pass/trial supplies the same-profile snapshot.
The replay checker requires exact reference bits and independent Hout checksums.
Each trial restores the prefix and poisons future cache entries. Cache
initialization has separate and inclusive counters.

## Build from offline exports

Supply capture, weights and independent reference files explicitly:

```sh
python3 gen/build_micromix_decode.py --model tiny --mix mix50 \
  --capture-header "$CAPTURE/llama_decode_capture.h" \
  --capture-bin "$CAPTURE/llama_decode_capture.bin" \
  --weights-header "$WEIGHTS/llama_micromix_mix50.h" \
  --weights-bin "$WEIGHTS/llama_micromix_mix50.bin" \
  --reference-bin "$REFERENCE/llama_decode_expect.bin" \
  --reference-manifest "$REFERENCE/reference.json" \
  --rocc-root "$ROCC_ROOT" --riscv-tool-prefix riscv64-unknown-elf- \
  --output-root "$FRESH_OUTPUT" --targets host spike
python3 tests/test_micromix_decode.py --build-root "$FRESH_OUTPUT" \
  --spike "$SPIKE" --gemmini-lib "$CORRECTED_DIM16_MODEL"
```

Spike requires `dtc` on `PATH`. The matching model library must include direct
E3M2 and full-BF16 small-product support. `--targets fpga` produces a firmware
ELF; board execution is separate. Host and Spike counters are functional
checks, not hardware speed measurements.

| `--model` | Capture basename | Weight basename | Reference basename | Reference family |
|---|---|---|---|---|
| `tiny` | `llama_decode_capture` | `llama_micromix_<mix>` | `llama_decode_expect` | `products_v2` |
| `qwen` | `qwen_layer_e2e` | `qwen_micromix_<mix>` | `qwen_decode_expect` | `products_v2_qwen7b_v2_exp32cr` |
| `llama` | `llama7b_layer_e2e` | `llama7b_micromix_<mix>` | `llama7b_decode_expect` | `products_v2_llama2_7b_exp32cr_v1` |

References contain eight complete BF16 stage vectors under the full-BF16
product semantics. Qwen and Llama references also specify independently checked
FP32 exponential rounding. The C implementation does not generate its own
reference outputs.

The builder refuses an existing output directory. It validates model geometry,
reference hashes and finite values before writing files or invoking tools.
Capture, weight and reference symbols must be 64-byte aligned. `--objects-dir`
reuses existing objects only after their actual ELF `.data` matches the raw
binary. Model data and generated binaries belong outside the source tree.

Compiler inputs are hashed before and after compilation, including header
`-M` dependencies, runtime C/assembly/link script, objects and toolchain files.
A source capsule preserves the SDK, profile and builder used for each ELF.
Generated cases bind each token to the word-FNV64 checksum of its independent
Hout vector; the checker compares that checksum with the guest output.

The Makefile provides equivalent `micromix-decode` and `micromix-decode-test`
targets through `MICROMIX_DECODE_ARGS` and `MICROMIX_DECODE_TEST_ARGS`.
`make micromix-attention-test` checks synthetic attention/cache cases, while
`python3 tests/test_micromix_decode.py` tests builder preservation without
model assets.

## Attention and cache policy

Builds use `-fno-fast-math -ffp-contract=off`, retaining explicit FMA sites.
RV64 builds include `mx_micromix_dim16.h` before the device API; the default
DIM32 RoCC parameters describe a different target. Projection schedules use
`NC=256, RUN=8, fence_groups=1` and optimized scalar glue.

The builder follows `gen/micromix_profiles.json` by default. Tiny mix50 selects
tiled GQA8 (mode 4), Qwen mix50 selects tiled GQA7 (mode 5), and Llama mix50
selects independent-head tiled MHA (mode 6) with `cache_time_tile=16`. Their
mix625 rows retain GQA8 mode 1, GQA7 mode 2 and MHA4 mode 3 respectively; Llama
mix625 keeps cache tile 0. Attention and cache controls are separate fields.

Llama uses logical capacity 2,056, physical head pitch 2,057 and score stride
2,056. Its extra physical slot remains poisoned. Other models use 2,056 for
all three values. The blocked initializer preserves every float mirror word
and future NaN.

`--attention-mode 0..6` and Llama's `--cache-time-tile 0|16` explicitly override
the profile. Override replays carry no new FPGA performance qualification.
Geometry dispatch occurs outside the specialized loops. These examples check
one layer, not complete autoregressive inference or perplexity; quantization
quality against the unquantized model is separate from same-mix correctness.

## Opt-in mixed row layout

`mx_micromix_decode_rows.h` adds an independent issue-height selector:

- `MXMM_DECODE_ISSUE_M32` (0) delegates to canonical 32-row projection.
- `MXMM_DECODE_ISSUE_MIXED16` (1) issues 32 FP4 rows and 16 FP6/FP8 rows.

All measured profiles keep `decode_issue_mode=0`. The mixed16 layout has a
functional real-layer FPGA check for Qwen mix625, including complete stage and
update vectors and buffer guards. It has no repeated full-layer timing result
and is not a selected performance profile.

`--decode-issue-mode 1` is available for mixed Qwen decode. Both passes use the
same issue mode and are labelled `explicit_unqualified_override_replay`.
Explicit mode 0 is also an override. Neither choice changes attention policy.

Mixed16 accepts 1–16 valid rows and requires FP4 plus FP6 and/or FP8. Packing
and A-scale pitches remain 32 rows. Reserve `mxmm_workspace_bytes(32,K)` bytes
and 32*N BF16 output words: the first 16 rows are written, including numeric
zero padding; rows 16–31 are untouched. Output, workspace, weight groups and
scales need 64-byte bases. Use group fences and a schedule that fits 2,048
weight-scale bytes per run, such as the selected 256/8/fence1 schedule.

Callers must supply finite BF16 inputs, supported packed weight codes, valid
capacities and disjoint buffers. The API does not check arbitrary alias spans.
It owns a synchronous device stream and must not interleave with a prepared
lease. The public wrapper performs select-half-1 → fence → select-half-0 →
fence before packing; that work is included in projection timing. Internal
`mxdr_band_*` functions omit this transition and are not entrypoints.
