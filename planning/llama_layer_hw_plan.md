# A real TinyLlama decoder layer on MxGemmini: MLP first, then attention

**Created** 2026-09-05 · **Status** plan agreed, Step 1 next.

Goal (user, 2026-09-05): a kernel `.elf` that runs on **mxgemmini** — spike (`-DSPIKE_SIM`) and the
standalone RTL (`-DMX_ROCKET`) — with **real llama data** and **real back-to-back operations**: a
real MLP at llama dimensions first, then a full attention layer. `../software/gemmini-rocc-tests/`
is the reference and the home of the deliverable, exactly as the existing `matmul_tiled_*` family.

## Decisions taken (user, 2026-09-05)

| # | Decision |
|---|---|
| D1 | **True dims, sliced outputs.** `d_model = 2048` kept FULL; slice only the output side — one attention head (`head_dim = 64`) and a contiguous slice of the 5632 FFN neurons. Every Q/K/V/gate/up value is then an *exact* real llama quantity; only `down_proj`/`o_proj` are honest partial sums (over the chosen neurons / the one head). |
| D2 | **MLP first, attention after.** Two tests, not one fused layer ELF. |
| D3 | **All non-matmul glue runs on the Rocket host in fp32** — RMSNorm, residual, SiLU·up, softmax + causal mask, RoPE. User: *"you can run them all in fp32 in rocket, so that doesn't need to be mirrored in pytorch"* — so the golden models them in plain fp32 and downstream checks are **tolerance-based**, not bit-exact. See §4. |

## 1. Why the existing data cannot be reused

`data_evalrun_512` (from `app/capture_llama_tiles.py`) logs one 512×512 `(A_square, W_square)` pair
per projection at a **random, unrecorded** `(t0, i0, o0)` offset (`log_pairs_from_eval.py:183-195`),
and skips `k_proj`/`v_proj` entirely (`out_features = 4*64 = 256 < 512`). Two consequences:

* `gate_proj`'s out-features and `down_proj`'s in-features are different random windows, so
  `down(silu(gate)*up)` cannot be composed from them — the chain would be arithmetic on real
  numbers that never met in the model.
* attention has no K and no V.

So Step 1 is a **new capture** of one decoder layer: real hidden states plus the layer's actual
weight matrices, sliced consistently, from the same model and corpus as before.

## 2. Hardware facts that shape the design (verified 2026-09-05)

1. **The MX loop path ignores `A_transpose` / `B_transpose`.** `mx_loop_ws_spad`
   (`libgemmini/gemmini.cc:1144`) does `(void)rs1;` and never reads the transpose bits that
   `gemmini_loop_ws_spad`'s signature carries — they are only honoured by the stock int8
   `loop_ws` (`:711`). So `S = Q@Kᵀ` needs a **host byte transpose** of K. The scale half is free:
   K's requant scales come out `[T][head_dim/32]` and B wants `[group][col]`, which is the same
   transpose the chain seam already does (`chain_seam_hw_notes.md` §2).
2. **Scratchpad is 16384 rows × 16 B = 256 KB** (`BANK_NUM 4 × BANK_ROWS 4096`), shared by the A
   tiles, the B tiles and the resident output. So one matmul needs `M*K + K*N + out_bytes ≤ 256 KB`.
   That is the binding constraint on every shape below, and it is why `seq = 32` and why the
   d_model-wide outputs (`down_proj`, `o_proj`) are emitted in **two N-chunks of 1024**.
3. **`out_mx_fmt` selects the drain** (`gemmini.h:301`, arg 12 of `gemmini_extended3_config_ex`):
   `3` = BF16 non-requant, flat row-major in the internal spad, drained by a flat
   `gemmini_extended_mvout` (`matmul_tiled_fp8_64x64.c:150-160`); `0` = FP8 requant, plus the
   requantizer's E8M0 codes to the address given to `gemmini_mxquant_config_mvout`.
4. **Back-to-back on device already works and is the template.**
   `matmul_tiled_fp8_64x64_chain.c` keeps MM1's requant output resident in the operand-A *tiled*
   layout (`LOOP_WS_REQUANT_TILED`, rs2 bit 10) and its scales resident
   (`gemmini_mxquant_config_mvout_resident`), so MM2 reads both in place with zero DRAM traffic.
5. **FP8 requant is exactly MXQuant end to end** (`chain_seam_hw_notes.md` §8, §9.4): scale
   `X = 2^floor(log2 amax)`, no `- log2_pmax` term, codes byte-identical. So a host-side quantizer
   written to that convention agrees with the device's own.
6. **`mx_smem` accumulates and is never cleared** (`gemmini.cc:1190`). Every stage therefore needs a
   disjoint `C_spad` — and the same property is what would let a K-tiled matmul accumulate across
   `loop_ws` calls, which this plan deliberately does **not** rely on (RTL behaviour unverified,
   `chain_seam_hw_notes.md` §5).

## 3. The shape, and where every byte goes

TinyLlama-1.1B-Chat: `d_model 2048`, `heads 32`, `head_dim 64`, `kv_heads 4` (GQA, q head *h* uses
kv head *h/8*), `intermediate 5632`, 22 layers.

```
SEQ    = 32     real consecutive wikitext2 tokens
D      = 2048   FULL hidden size            -> RMSNorm and the residual are EXACT
NF     = 64     FFN neurons of 5632         -> gate/up EXACT per neuron; down is a partial sum
HEAD   = 1      query head + its kv head    -> Q/K/V EXACT; o_proj is that head's real contribution
```

`NF` is a multiple of 32 (the E8M0 block) and 16 (the PE tile). Scratchpad budget, in rows of 16 B:

| matmul | A | B | out | total | fits 16384 |
|---|---|---|---|---|---|
| `G = Xn @ Wg` `[32,2048]×[2048,64]` | 4096 | 8192 | 256 (bf16) | 12544 | ✓ |
| `U = Xn @ Wu` (A already resident) | 4096 | 8192 | 256 | 12544 | ✓ |
| `Y = H @ Wd` `[32,64]×[64,2048]`, **N-chunked ×2** | 128 | 4096 | 4096 | 8320 | ✓ |
| `Q/K/V = Xn @ W*` `[32,2048]×[2048,64]` | 4096 | 8192 | 256 | 12544 | ✓ |
| `S = Q @ Kᵀ` `[32,64]×[64,32]` | 128 | 128 | 64 | 320 | ✓ |
| `O = P @ V` `[32,32]×[32,64]` | 64 | 128 | 128 | 320 | ✓ |
| `Yo = O @ Wo` `[32,64]×[64,2048]`, **N-chunked ×2** | 128 | 4096 | 4096 | 8320 | ✓ |

Un-chunked `down_proj`/`o_proj` would need 16512 rows — 128 over. Hence the two chunks.

Baked data ≈ 0.9 MB → ~4 MB of C text per header. Spike: ~13 M MACs for the MLP, seconds.
Verilator: ~50 K compute cycles plus mvin.

## 4. What is checked, and what "bit-exact" can still mean

The host glue runs in fp32 on Rocket (D3) and is **not** mirrored bit-for-bit in Python, so a strict
byte-compare cannot be claimed for anything downstream of a host stage. The gate is layered instead:

| level | check | strictness |
|---|---|---|
| host quantization agreement | the host's own fp8 codes + E8M0 scales for each intermediate vs the golden's | reported as a mismatch count, **not fatal**. Expected 0: e4m3 has 3 mantissa bits, which absorbs an fp32 last-ulp difference in `expf`/`rsqrtf`. |
| device stage output | each matmul's bf16/fp8 output vs the golden computed from the *same* operands | **bit-exact** whenever the level above reported 0 mismatches — which is the normal case, so the hard gate survives |
| end to end | final tile vs the fp32 torch reference **of the same slice** | relative Frobenius, tolerance (expect ~5%, cf. `chain_seam_hw_notes.md` §9.7) |

The fp32 reference is the *sliced* computation (same neurons, same head), not the full layer — the
sliced quantity is what the device actually computes. The full-layer value is printed alongside for
context, never as a pass criterion.

## 5. Artifacts

| # | file | what |
|---|---|---|
| 1 | `npu-exploration/app/capture_llama_layer.py` | capture one real decoder layer: hidden states in/out, both RMSNorm weights, q/k/v/o + gate/up/down weight slices, RoPE cos/sin, and the fp32 references. One `.npz`. |
| 2 | `gemmini-rocc-tests/gen_llama_layer.py` | operands → MXQuant quantization → bit-exact mesh model per stage → `include/llama_mlp_<shape>.h` (and later `llama_attn_<shape>.h`) |
| 3 | `gemmini-rocc-tests/include/mx_host.h` | host-side fp32 MX library for baremetal C: bf16↔float, e4m3 encode/decode, E8M0 block quantize (MXQuant convention), rmsnorm, silu, softmax, rope |
| 4 | `gemmini-rocc-tests/bareMetalC/llama_mlp.c` | the MLP ELF, dual-mode `SPIKE_SIM` / `MX_ROCKET` |
| 5 | `gemmini-rocc-tests/bareMetalC/llama_attention.c` | the attention ELF |

## 6. Steps

| # | step | gate | state |
|---|---|---|---|
| 1 | `capture_llama_layer.py` — real layer capture | the npz exists; `Xn @ Wg` in numpy matches torch's own `gate_proj` output on the sliced neurons | **DONE** 2026-09-05 — rel 2.45e-3 (bf16 forward vs fp32 recompute), RMSNorm 5.12e-3 |
| 2 | `gen_llama_layer.py` MLP path + `llama_mlp.h` | header emits; python chain reproduces the fp32 reference within the expected MX error | **DONE** — 5.0 MB, rel_fro 11.5928% |
| 3 | `include/mx_host.h` | unit-checked against `mxq_golden` on the captured tensors before it is trusted in C | **DONE** — 0/65536 code, 0/2048 scale mismatches, natively compiled |
| 4 | `bareMetalC/llama_mlp.c` on **spike** | stages bit-exact vs golden; end-to-end within tolerance | **DONE** — every stage 0 mismatches, rel_fro 115933 ppm |
| 5 | both ELFs built for **MX_ROCKET** | compiles on the RTL path | **DONE** — `build_mx_rocket/bareMetalC/`; running them needs an RTL sim binary, not built here |
| 6 | attention golden (RoPE, causal softmax, the K transpose) | python chain matches torch attention for that head | **DONE** — 6.0 MB, rel_fro 13.4157% |
| 7 | `bareMetalC/llama_attention.c` on spike | as above, plus the resident seam | **DONE** — 6 matmuls bit-exact, rel_fro 134161 ppm |
| 8 | notes back into `chain_seam_hw_notes.md` + this file | — | this file done; §9 below |

## 8. Results (2026-09-05)

Both kernels PASS on spike with **every mesh stage bit-exact** against the golden, and the host's
fp32 glue reproduces the golden's own operand codes byte for byte — so the layered gate of §4
collapsed to its strongest rung: nothing drifted, so nothing needed a tolerance.

```
llama MLP: M=32 D=2048 F=64
host  rmsnorm+quant: codes differ 0/65536, scales differ 0/2048 vs golden
mesh  G = Xn @ Wg : 0/2048     mesh  U = Xn @ Wu : 0/2048
host  silu(G)*U + quant: codes differ 0/2048, scales differ 0/64 vs golden
mesh  Y = H @ Wd  : 0/65536 differ from golden (2 chunks of 1024)
grade MLP output vs fp32 reference : rel_fro 115933 ppm   residual out: 305 ppm
cycles mesh 11543, host 11619644

llama attention: M=32 D=2048 head_dim=64
mesh  Q,K,V = Xn @ Wq/Wk/Wv : 0/2048 each
host  RoPE(Q,K) + K transpose: Q codes 0/2048, K^T codes 0/2048
mesh  S = Q @ K^T : 0/1024
host  causal softmax + quant: P codes 0/1024, V codes 0/2048
mesh  O = P @ V   : 0/2048 codes, 0/64 scales (requant -> spad)
mesh  Y = O @ Wo  : 0/65536 (O resident, scales resident, 2 chunks)
grade attention out vs fp32 reference: rel_fro 134161 ppm   residual out: 545 ppm
cycles mesh 15467, host 12596206
```

Bit-exactness here is a real gate, not a tautology: the golden comes from
`fp8_matmul_model.tiled_matmul_hwlike` (python) and the result from `gemmini.cc` (C++), two
independent implementations of the same precision schedule.

### 8.1 The resident seam, and how it is proven

`O = P@V` → `Y = O@Wo` is the only seam in either kernel with no host op in it, and it runs fully on
device: `LOOP_WS_REQUANT_TILED` deposits O in the operand-A tiled layout,
`gemmini_mxquant_config_mvout_resident` writes its E8M0 bytes into the act-scale window transposed
(`a_off = group*M + row`), and `o_proj` issues **no A mvin and no A-scale load**.

The `Y` check is what proves the scale half. Had the resident write not happened, that matmul would
have run against the previous stage's leftover A-scales — `P`'s 32 bytes where 64 are needed — and
`Y` could not have matched a golden computed from O's own scales. The code half is proven separately
by reading O back out of the scratchpad (`mvout_detile`, read-only) and comparing to `O_OUT`.

**The MLP has no such seam.** Its only matmul-to-matmul junction is `gate/up → down`, and SwiGLU
sits in it: `silu(G) * U` is not a matmul and there is no hardware for it, so those values must
reach the scalar core. Requant-to-spad there would keep them on device but the host still could not
read them without a mvout, and it would double-quantize for a strictly worse result. What the MLP
does reuse is the A side: `Xn` is mvin'd once and both projections read it in place, with only the
B-side scale window reloaded. Same for attention's Q/K/V, which share one resident `Xn` across three
matmuls.

### 8.2 Where the error comes from, and what MXQuant already models

Both kernels land well above the ~5% the 64–256-deep `matmul_tiled_*` tests report
(`chain_seam_hw_notes.md` §9.7). Splitting on identical operands and identical host glue, varying
only what each matmul does:

| stage of the model | MLP | attention |
|---|---|---|
| MX block quantization alone (exact-arithmetic matmul on dequantized operands) | 6.57% | 8.74% |
| + in-tile product + per-lane accumulator quantization, product rounded to NEAREST | 8.46% | — |
| + product TRUNCATED instead (as the PE does) | **11.31%** | — |
| + cross-tile bf16 accumulate (the full datapath, = spike and the RTL) | **11.59%** | **13.42%** |

### 8.2b MEASURED: MXQuant's own model reproduces spike BIT-EXACTLY, after three changes

Rather than reason about the difference, `prodacc_bundle/eval_complete.py` was extracted from
`origin/chloe-branch-all` and its `MXLinearSim` driven directly with the captured llama tensors and
the RTL's schedule. Three hooks were added to a copy, each defaulting to the shipped behaviour and
verified inert (the unpatched copy reproduces the original output exactly):

| MXQuant `MXLinearSim`, RTL schedule | vs fp32 | vs spike |
|---|---|---|
| as shipped | 10.63% | 17.90% |
| + RTL product quantizer | 9.03% | 8.83% |
| + RTL accumulate | 8.65% | 12.55% |
| + both | 11.31% | 3.98% |
| + bf16 cross-tile accumulate | **11.59%** | **0.00% — 65536/65536 identical, max abs diff 0.0** |
| spike / hardware model | 11.59% | — |

So the divergence is **fully accounted for**, by exactly three differences:

1. **Product quantizer.** `float_quantize(rounding="nearest")` vs the PE's mantissa TRUNCATION with
   no exponent clamp at the product stage (`mx_product_quantize_trunc`, `fp8_matmul_model.py:202`).
2. **Accumulate step.** MXQuant computes `S_red = S_red + outer_q` in fp32 and quantizes the SUM;
   the hardware quantizes BOTH the running sum and the incoming product to the lane's precision and
   then adds exactly — `fp_add_exact(fp_quantize_rne(C, e, m), fp_quantize_rne(outer, e, m), e, m)`
   (`fp8_matmul_model.py:631-637`, mirroring `gemmini.cc:1230-1233`). The product is therefore
   quantized TWICE in hardware: once to e4m3, again to the accumulator lane.
3. **Cross-tile accumulate.** fp32 in the model, bf16 in hardware.

**These do not decompose.** Each change ALONE lowers the error versus fp32 (10.63% -> 9.03% or
8.65%), and two of the three make the divergence from spike WORSE; only all three together land on
the hardware. Any claim about "the product rounding is worth N points" measured with the other two
unmatched is meaningless — an earlier draft of this section made exactly that mistake, using a
stand-in model rather than MXQuant's own, and got both the magnitude and the SIGN wrong.

**What it means in practice.** As shipped, MXQuant reports 10.63% where the hardware gives 11.59% —
within a point on the norm while diverging 17.90% element-wise. Its aggregate accuracy numbers
(perplexity) are therefore roughly right; it is not predicting the values this datapath produces.
Making it exact costs ~5 lines, and that is now a permanent artifact: **`rtl_exact/`** in this
repo (NOT in the MXQuant checkout, which stays clean — `rtl_datapath.install()` rebinds one
method at runtime). It carries the config, the schedule in MXQuant's own CLI format, a fixture
holding the hardware's output, and `verify_rtl_exact.py`, which re-establishes the bit-exact
claim in one command. Use that config for any MXQuant exploration meant to describe MxGemmini.

**The middle row is the one MXQuant already simulates**, so the format is NOT the whole story and
neither is it a gap in MXQuant's methodology. `prodacc_bundle/eval_complete.py` on
`origin/chloe-branch-all` implements structurally the same datapath as `gemmini.cc`: a 16-lane
window, every outer product quantized to `(prod_e, prod_m)`, and the running sum re-quantized per
lane from an `--acc-schedule`. `systolic_simulation/end_to_end_schedule.txt` carries
`product_precision: B=7 e=4 m=3`, i.e. exactly the RTL's `prod_e=4, prod_m=3`, and the sweeps
(`sweep_accum_vs_ppl.py`, `sweep_acc_schedules.py`, `find_optimal_acc_precision.py`) drive it to
perplexity end to end.

So an MXQuant e2e run **given the RTL's schedule** should predict this hardware to within a few
tenths of a point. Three differences remain, all small:

* **Product rounding differs — and this is the big one, 2.9 points**: qtorch
  `float_quantize(rounding="nearest")` vs `gemmini.cc`'s truncating `mx_product_quantize_trunc`.
  See the table above.
* **The shipped schedule is not the RTL's.** `end_to_end_schedule.txt` is an optimization result
  (`lane01 e2m12`, `lane02 e1m9`, …); the RTL is `acc_e = {4 x15, 8}`,
  `acc_m = {4,4,4,4,4,4,4,4,5,5,6,6,6,6,6,7}`. The model predicts the schedule it is given. Every
  number in the table above uses the RTL's, so the 2.9 points are NOT a schedule artefact.
* **Cross-tile accumulation is fp32 in the model** (`C = C + S_red * scale_map`) and **bf16** in
  hardware (`bf16_accum_add` of a bf16-rounded scaled tile, `fp8_matmul_model.py:718-724`,
  `gemmini.cc:1190`). Worth 0.28 points at K = 2048 — measured, not estimated. The bundle's own
  README says it is a software datapath model, "not the RTL golden model".

**A wrong explanation, corrected.** An earlier draft of this section attributed the 6.57% -> 11.59%
gap to the bf16 accumulate over a 2048-deep reduction, on the strength of `sqrt(2048) * 2^-9 ~ 9%`
closing it in quadrature. That was a coincidence: switching the cross-tile accumulate to fp32 moves
the result only 11.59% -> 11.31%. The error is in-tile — truncating products and the narrow per-lane
accumulators — not in the depth of the reduction.

**The convention is exactly MXQuant.** Operands and the FP8 requantizer both follow the end-to-end
convention (`X = 2^floor(log2 amax)`, no `- log2_pmax`, ties away from zero, saturate at ±448); the
goldens call `quantize_mx_block32` itself rather than transcribing it, `mx_host.h` reproduces those
codes 0/65536, and spike matches bit-for-bit. FP8 only — FP4/FP6 were never migrated
(`chain_seam_hw_notes.md` §9.4), which does not bite here since both kernels are FP8.

Measured with two throwaway scripts (`mxq_vs_hw.py`, `crosstile.py`) in the session scratchpad; both
are ~50 lines over `gen_llama_layer` and worth re-deriving rather than preserving.

### 8.3 The host glue is 99.9% of the cycles

`mesh 11543, host 11619644` — the scalar fp32 RMSNorm + MX quantization costs a thousand times what
the mesh does. Some of that is spike counting instructions rather than cycles, but the ratio is the
point: at this shape the accelerator is not the bottleneck, the glue is. RMSNorm alone touches
M*D = 65536 elements, and the quantizer touches the same 65536 with a 7-step binary search each.
This is the argument for the vector lane, and for `chain_seam_hw_notes.md`'s "no seam at all".

### 8.4 Two traps worth not re-discovering

* **`-DSPIKE_SIM` comes from `RUNNER`.** `bareMetalC/Makefile:111` adds it only when `$(RUNNER)`
  contains `spike`. Building the sub-make target without `RUNNER=spike` silently takes the MMIO
  command-mimic path, and the ELF traps on a store to `0x40084010` with `tohost = 1337` — the same
  symptom as a stale `libgemmini.so`. Overriding riscv-tests' weak `handle_trap` to print the cause
  and epc turned that from a guess into one line, and it is kept in both kernels for that reason.
* **libm is unusable as-is on the baremetal path.** `-lm` sits inside `CFLAGS`, i.e. *before* the
  sources, so it resolves nothing; and once appended properly, newlib's libm needs `__errno`, which
  `-nostdlib` leaves undefined. `mx_host.h` supplies the stub, and `LIBS := -lm` was added after the
  sources. Only `expf` still needs it — every power-of-two operation (`ldexpf`, `log2f`, `floorf`)
  was replaced with exact bit manipulation, which is both dependency-free and immune to a library
  rounding `log2f(8)` to 2.9999997 and moving a block scale by a whole exponent.

Makefile note, from `chain_seam_hw_notes.md` §9.3: generated headers must be in the bareMetalC
Makefile's prerequisites or a regenerated golden silently re-runs the old data **and passes**. The
wildcard there is `include/matmul_*.h`; the new `llama_*.h` names need adding.

### 8.5 Reproducing

```bash
# 1. capture one real decoder layer (once; needs the venv + HF cache)
cd generators/gemmini/npu-exploration && .venv/bin/python3 -m app.capture_llama_layer

# 2. generate both headers (~30 s each; the mesh model runs every stage)
cd ../software/gemmini-rocc-tests
PATH=../../npu-exploration/.venv/bin:$PATH \
  ../../npu-exploration/.venv/bin/python3 gen_llama_layer.py mlp attn

# 3. build for spike -- RUNNER=spike is what defines -DSPIKE_SIM (see 8.4)
cd /path/to/radiance-cy-dev && source ./env.sh
T=$PWD/generators/gemmini/software/gemmini-rocc-tests
cd $T/build_spike/bareMetalC && make -f $T/bareMetalC/Makefile abs_top_srcdir=$T XLEN=64 \
  PREFIX=examples-bareMetalC src_dir=$T/bareMetalC RUNNER=spike \
  llama_mlp-baremetal llama_attention-baremetal

# 4. run -- ALWAYS with --extlib, against the in-tree .so
spike --extlib=$PWD/../../../libgemmini/libgemmini.so --extension=gemmini llama_attention-baremetal

# 5. the same sources for the RTL path
cd $T/build_mx_rocket/bareMetalC && make -f $T/bareMetalC/Makefile abs_top_srcdir=$T XLEN=64 \
  PREFIX=examples-bareMetalC src_dir=$T/bareMetalC EXTRA_CFLAGS=-DMX_ROCKET \
  llama_mlp-baremetal llama_attention-baremetal
```

Regression-checked after the shared-Makefile change (`LIBS := -lm`):
`matmul_tiled_fp8_64x64`, `..._requant`, `..._chain` and `matmul_tiled_fp4_64x64` all still PASS.

## 7. Honest limits, stated up front

* `down_proj` and `o_proj` are **partial sums** — over `NF` of 5632 neurons, and over 1 of 32 heads.
  Every *operand* is real and exact; the *reduction* is truncated, and the fp32 reference is
  truncated the same way.
* One head means no head concatenation and no cross-head interaction.
* `seq = 32` is a real but short context; the causal mask is a real 32×32 lower triangle.
* Host glue in fp32 on a scalar core is not what an accelerator would do — it is the honest routing
  (`npu_exploration_bridge_plan.md` §13.2: the mesh has no reduction hardware), and it is measured
  separately from the accelerator cycles.

## 9. A small-D variant, for RTL simulation (2026-09-14)

**Why.** Nothing had run on RTL. At the captured D = 2048 the host's fp32 glue is 12.6 M cycles
against the mesh's 15 K (section 8.3), and VCS on this design runs ~4900 cycles/s (measured:
`matmul_tiled_fp8_64x64` = 106515 cycles in 21.7 s CPU, `sims/vcs/mx_vcs_logs_*`). That is hours per
run, and it overruns the `+max-cycles=10000000` that `run_mx_vcs.sh` passes. The host cost is linear
in D, so a D-slice is the knob.

**What was added.**

| # | artifact | what |
|---|---|---|
| 1 | `gen_llama_layer.py --d D --tag SUFFIX` | `slice_capture()` cuts the hidden axis of the existing capture npz and RECOMPUTES `ref_mlp`/`ref_attn` from the sliced tensors (a sliced reference is not the slice of a reference). Emits `include/llama_{attn,mlp}<tag>.h`. |
| 2 | `bareMetalC/llama_attention_small.c` | two lines: `#define LLAMA_ATTN_HEADER "include/llama_attn_small.h"` and include `llama_attention.c`. |
| 3 | `llama_attention.c` spad map | the 11 addresses were baked decimal literals assuming D = 2048; now DERIVED from `LLAMA_M/D/H` via `ROWS8`/`ROWS16`, with three `LLAMA_SPAD_REQUIRE` compile-time checks on the live ranges. All 11 reproduce the old values exactly at D = 2048. |

**Honest limits of the slice**, beyond section 7's. RMSNorm now normalizes over 256 features rather
than 2048, so `xn` is NOT the real llama normalized activation; Q/K/V join `Wo`/`Wd` as partial sums
over 256 of 2048 input features. Every operand is still a real llama value at its real index and the
fp32 reference is truncated identically, so the grade describes what the device computes. The mesh
goldens are bit-exact either way, which is what the kernel actually gates on.

Result on spike: `M=32 D=256 head_dim=64`, all six mesh matmuls 0 mismatches, `rel_fro 109020 ppm`,
`cycles mesh 2345, host 2532108` -- ~5x less host work than the full kernel, so ~8 min of VCS.

### 9.1 A rounding-mode split the slice exposed

`include/mx_host.h::mx_e4m3_encode` broke ties AWAY FROM ZERO. MXQuant's `quantize_mx_block32` --
which every golden here is generated from -- rounds HALF TO EVEN, and so does the datapath model
(`fp_quantize_rne`, `fp8_matmul_model.py:631`). A tie needs `v / X` to land precisely midway between
two E4M3 magnitudes (e.g. 0.78125, between 0.75 at m=4 and 0.8125 at m=5).

The split was invisible because BOTH sides were stale in the same direction: commit `eba2fcb`
(2026-09-11, "update all headers to match modified rounding mode") regenerated the `matmul_*`
headers but NOT `llama_attn.h` / `llama_mlp.h`, which stayed at their 2026-09-05 generation under the
old rule. `mx_host.h` agreed with those stale goldens, so everything passed. Generating a header
today against current MXQuant put the two conventions in one ELF: 95/2048 V codes wrong.

Fixed on both sides -- `mx_host.h` now rounds ties to even, and both llama headers were regenerated.
The grade moved, because the goldens did:

| kernel | rel_fro, stale golden | rel_fro, regenerated |
|---|---|---|
| `llama_mlp` | 115933 ppm | **119815 ppm** |
| `llama_attention` | 134161 ppm | **150192 ppm** |

All three kernels pass on spike with every stage 0 mismatches. `selftest_quantizer`,
`selftest_mx_host` and `selftest_formats` pass.

**The lesson worth keeping:** a generated golden and the host library that must agree with it are two
copies of one convention. Nothing in the build ties them together -- the Makefile's prerequisite
wildcard rebuilds an ELF when a header changes, but nothing regenerates a header when the convention
does. `matmul_*` was regenerated by hand and `llama_*` was forgotten.

### 9.2 DONE (2026-09-15): the llama kernels moved to npu-exploration/baremetal/mxgemmini

**Decision (user, 2026-09-14), carried out 2026-09-15.** User: *"lets contain all the higher
level kernel development in this repo. Baremetal C should contain only the proof of concept,
basic testing."*

`software/gemmini-rocc-tests/bareMetalC/` is the ISA-level test suite -- one test per instruction
behaviour, each self-contained, each runnable by `run_mx_vcs.sh`. `llama_mlp.c`,
`llama_attention.c`, `llama_attention_small.c` and `llama_attention_full.c` are not that: they are
whole-application kernels with captured model data, and they carry generators
(`gen_llama_layer.py`, `gen_llama_attn_full.py`) and a host runtime (`include/mx_host.h`) that have
nothing to do with the ISA tests around them.

What the move would involve, when it happens:

* the four `.c` kernels, `include/mx_host.h`, and both generators move under `npu-exploration/`
  (`kernels/baremetal/` is the obvious home -- it is already where "what to run" lives);
* they keep building against `gemmini-rocc-tests`' headers and Makefrag, so the build rule has to
  reach back into it rather than the sources living there;
* `bareMetalC/Makefile`'s test list loses four entries and its `include/llama_*.h` wildcard;
* `run_mx_vcs.sh` keeps working, since it takes an explicit `BUILD_DIR`/`BINARY`.

**What actually moved**, to `npu-exploration/baremetal/mxgemmini/`: the four kernels into `src/`,
`mx_host.h` into `include/`, both generators into `gen/`, and the generated data into `data/` --
which is now GITIGNORED and rebuilt by `make data`, because tracking it is exactly how
`llama_attn.h` sat stale through a rounding-mode change (section 9.1) while every test passed.
`gen_matmul_llama.py` stayed behind: it generates the `matmul_*.h` data for the ISA tests.

A self-contained `Makefile` builds both modes into `out/baremetal/{spike,mx_rocket}/`, reaching into
`gemmini-rocc-tests` only for the ISA headers and the riscv-tests runtime.

Verified after the move: all four kernels build and PASS on spike, and the ISA suite still builds.

**Two things the move turned up.**

1. **A second, stale copy of `mx_host.h`** lives at
   `compiler/targets/mx_gemmini_rocket/backend/runtime/mx_host.h`, and it still had the
   ties-away-from-zero encoder that section 9.1 corrected. `selftest_mx_host.py` tests THAT copy,
   and passes, because its random operands contain no exact tie -- the identical blind spot that hid
   the bug the first time. Both copies now round ties to even. **They are still two copies of one
   convention, which is the same hazard in a different place: worth deduplicating.**

2. **`llama_mlp` had no scratchpad asserts.** At `BANK_ROWS 2048` its map ran off the end of an
   8192-row scratchpad and it completed with 69,569 wrong elements rather than failing. It now
   carries the same `LLAMA_REQUIRE` live-range checks `llama_attention` has, so an unworkable
   config is a build error naming the region that overflowed.

### 9.3 All four kernels made scratchpad-adaptive (2026-09-15)

`llama_mlp` and `llama_attention` carried fixed maps sized for 16384 rows and, at `BANK_ROWS 2048`,
needed two things no placement could give them:

* the **weight tile cannot sit beside the activations**. A `[32,2048]x[2048,F]` projection needs Xn
  (4096 rows) and B (8192 rows) resident together = 12288, against a whole scratchpad of 8192. The
  fix is to SPLIT the D-deep contraction into K-tiles that accumulate into one output region --
  `ex_accumulate = 0` on the first, `1` after. Only possible since section 10.3;
* the **output chunks no longer need to be disjoint**. They were only ever disjoint because every
  matmul accumulated; with overwrite semantics one region is drained and reused per chunk.

Both now derive `KTILE` and `NCHUNK` at compile time from `BANK_NUM * BANK_ROWS`, taking the largest
that fits. Measured, same ELF sources, both configs:

| kernel | 16384 rows | 8192 rows | rel_fro, both |
|---|---|---|---|
| `llama_mlp` | 1 K-tile of 2048, 2 chunks of 1024 | 2 K-tiles of 1024, 4 chunks of 512 | 119815 ppm |
| `llama_attention` | 1 K-tile of 2048, 2 chunks of 1024 | 2 K-tiles of 1024, 4 chunks of 512 | 150192 ppm |
| `llama_attention_small` | 1 K-tile of 256, 1 chunk of 256 | same | 109020 ppm |
| `llama_attention_full` | `proj N=64`, 2 chunks of 1024 | `proj N=16`, 4 chunks of 512 | 212040 / 210655 ppm |

All four PASS at both sizes with every mesh matmul bit-exact, and **every graded number is identical
across the two** -- so splitting a 2048-deep reduction into two accumulating 1024-deep matmuls is
exact, not approximate. At 16384 rows the projections pick a single K-tile spanning all of D, i.e.
the original one-matmul schedule: the adaptive version costs nothing on the larger config.

## 10. The FULL attention sub-layer -- all 32 heads (2026-09-14)

**Goal (user):** one attention in mxfp8 on spike with nothing truncated. `llama_attention.c` runs one
head, so `o_proj` is a partial sum over 1 of 32 and the only reference is a numpy reimplementation of
that slice. With every head present the result IS the layer's real attention output, and it can be
graded against the model itself.

| artifact | what |
|---|---|
| `app/capture_llama_layer.py --all-heads` | keeps the full `[D][D]` q/o and `[D][KVD]` k/v projections; adds `attn_torch`, hooked from `self_attn`'s own output |
| `gemmini-rocc-tests/gen_llama_attn_full.py` | quantizes, runs every stage through the mesh model, emits a 10.8 MB BINARY BLOB + an offsets header |
| `gemmini-rocc-tests/bareMetalC/llama_attention_full.c` | the kernel; tiling planned at compile time from `BANK_NUM * BANK_ROWS` |

Result on spike, **11 seconds**: every mesh matmul 0 mismatches across all 32 heads.
`rel_fro 212040 ppm` vs the fp32 reference and `210655 ppm` vs the model's own output -- matching the
python chain's 21.2050% / 21.0664% to within 1 ppm.

### 10.1 Grade against the model, not a reimplementation

`h_mid - h_pre` is mathematically the attention output, and it is the WRONG way to get it: the
residual stream is **66x larger** than the attention output it carries, so differencing two bf16
values loses most of the precision to cancellation. Measured: 2.39e-2 differenced vs **7.81e-3** from
`self_attn`'s own forward hook, on identical math. The hook is what the capture now stores.

### 10.2 Data as a linked blob, not C initializers

9.1 MB of weights as `0x%02x, ` text is ~56 MB of C source. The blob is linked with
`objcopy -I binary` (`--set-section-alignment .data=64`, since the default is 1 and the header casts
to `uint16_t*`), and the generated header carries offsets. The blob is also **tiling-independent** --
an output column depends only on its own column of B -- so changing the scratchpad size replans the C
without regenerating any data.

### 10.3 ROOT CAUSE: the MX path discarded the accumulate bit

Stock `loop_ws` takes `ex_accumulate` in rs1 bit 0 and, on the first k-tile, OVERWRITES rather than
accumulates (`gemmini.cc:712` and `:848`). That is what lets an output region be reused, and every
caller in this repo already passes it -- `gemmini_loop_ws_spad`'s macro ends
`| ((full_C) << 1) | (ex_accumulate)`.

`mx_loop_ws_spad` threw rs1 away (`(void)rs1;`, `:1169` -- the same line that drops the transpose
bits) and did `smem[idx] = bf16_accum_add(prev, scaled)` unconditionally. So the MX path ALWAYS
accumulated. Consequences, all of which this repo had absorbed as facts of life:

* "every stage needs a DISJOINT `C_spad`" (section 2 fact 6) was not a hardware property at all --
  it was this omission;
* an output region could never be reused for the life of the ELF, so the total mesh output an ELF
  could produce was capped by the scratchpad. Full attention needs **26624 rows** of distinct output
  addresses against **16384** available -- 1.62x over, before a single operand byte. No tiling fixes
  that, because retiling changes chunk shapes, not the total number of output elements;
* `mvout` frees the SPAD rows but never touches `mx_smem`, which is a separate bf16 shadow
  accumulator indexed by `C_spad` -- which is why "drain to DRAM and overwrite" does not work.

**There was also no way to clear it.** `config_reset_funct` and `loop_conv_ws_config_1_funct` are
BOTH 16 (`libgemmini/gemmini.h:216, :243`) and the conv branch is tested first (`:2248` vs `:2290`),
so `config_reset_funct` is dead code; `resetted` is set true once at startup and never cleared.

**Fix (2026-09-14):** honour the bit in `mx_loop_ws_spad` -- `const bool ex_accumulate = rs1 & 1;`
plus a `clear_smem_if_overwrite(M_DIM, N_DIM)` called in all three format branches, zeroing the
region up front (equivalent to stock's per-tile guard, one pass). Regression on spike:
**71 passed, 0 regressions**; the single failure, `matmul_ws_mx_generic`, fails identically on the
unpatched model (verified by rebuilding the original and re-running -- `tohost = 1337`, an unhandled
trap, not a numeric mismatch).

With regions reusable, the kernel collapsed: no reset scaffolding, no slot rotation, one output
region per phase, and o_proj passing `ex_accumulate = 1` for every head after the first so the 32
contributions add in place.

**UNVERIFIED ON RTL.** The fix is to the functional model. Whether the MX RTL honours `ex_accumulate`
is unknown -- `chain_seam_hw_notes.md` section 5 already records cross-call accumulation as
unverified there. Stock gemmini RTL has the accumulator overwrite bit, so the MX RTL plausibly does
too, but until that is checked `llama_attention_full` is a spike result. The ELF builds for
`-DMX_ROCKET` and is at `build_mx_rocket/bareMetalC/llama_attention_full-baremetal`.

### 10.4 Two bugs the kernel hit on the way, both worth not re-discovering

* **B-side scale windows are `[K/32][N]` with `b_off = group * N + col`.** Slicing N columns out of a
  `[GD][N_full]` table is a GATHER, not a contiguous run -- the rows are `N_full` apart. Indexing it
  as contiguous made the first mesh stage ~100% wrong.
* **The requantizer writes O's E8M0 as `[M][GH]`, but the A-side window wants `[GH][M]`.**
  `llama_attention.c` never had to care because its o_proj read them RESIDENT, written transposed by
  the hardware. This kernel re-mvins O afresh, so the host must do that transpose itself.

## 11. Build-artifact cleanup (2026-09-15)

`out/` had grown to 111 MB across 54 build directories and 82 ELFs -- almost all of it one-off
debugging runs of `run_kernel.py` (the compiler track) from early September. Cleaned to the kernels
that are actually maintained.

### 11.1 What "runs successfully on spike" turned out not to mean

The first instinct was to keep whichever ELFs still pass. Running all 82 against the current
`libgemmini.so` gave **82/82 exit 0, no error markers** -- including every ELF built before the
2026-09-14 `ex_accumulate` fix (section 10.3). That is not a sign the fix is irrelevant: a compiler
track stage ELF *prints* its output and `grade/pipeline.py` computes the verdict **host-side**. The
ELF has no pass/fail concept, so "does it run" cannot discriminate between a good kernel and a dead
sweep. Selection has to be by name and by whether the kernel is in `kernels/registry.py`.

### 11.2 Kept

| path | why |
|---|---|
| `out/build/{llama_mlp,llama_attention}` | the llama kernels on the compiler track |
| `out/build/mlp{2,3,4,6,8}` | the chained-MLP kernels, all registered in `kernels/registry.py` |
| `out/baremetal/{spike,mx_rocket}/` | the four hand-written kernels |
| `out/layer_capture/*.npz` | **not** build output -- the capture is the INPUT to `make data`, and regenerating it needs the venv plus a HuggingFace download |

### 11.3 Deleted (48 directories, ~5.4 MB, plus three regenerable caches)

`bside_*`, `marg_*`, `probe_*`, `inv_*`, `chain_g*`, `capsule_MB0`, `requant_probe`, `decomp` --
37 sweep directories, referenced **nowhere** in the repo (no script, doc, or test names any of
them). Also `attention`, `linear`, `torch_linear` (not llama, not mlp) and `mlp2d`..`mlp8d`.

The `*d` variants are worth a note: they are not in the kernel registry and nothing references the
script that made them, so unlike every other deletion here they are **not regenerable**. They were
ad-hoc dtype sweeps; the information that survived them is in this plan and in
`chain_seam_hw_notes.md`.

Three more, all verified regenerable before removal rather than assumed:
* `out/mxquant_prodacc/` -- a cache, repopulated by `grade/mxquant_ref.py:_prodacc_dir()` from
  `git show origin/chloe-branch-all:prodacc_bundle/*`. Confirmed the ref still resolves and still
  carries all three files; deleting it only costs one `git show`.
* `out/requant_oracle` -- the compiled x86 oracle, rebuilt by `tests/selftest_requant.py`.
* `out/artifacts/chain_2gemm/` -- an `--artifacts` RTL-replay bundle from September 1.

`out/` is now 104 MB, 44 MB of which is the capture that must not be regenerated casually.

### 11.4 Header regeneration, and why the timestamps lied

`gen/*.py` were both last edited 09-14 23:49 while `llama_mlp.h`, `llama_attn.h` and
`llama_attn_small.h` were generated 09-14 13:45-13:46 -- ten hours stale, the exact pattern that
produced the wrong-golden bug in section 9.1. Regenerated all four via `make data` and diffed
against a backup: **all four came back byte-identical**, `llama_attn_full.bin` (10.8 MB) included.
The 23:49 edit was the directory move (path constants, argparse), which does not touch emitted data.

This is the check worth repeating rather than the conclusion worth remembering: a generator newer
than its output is a question, not a verdict, and `cmp` answers it in seconds.

Rebuilt for both targets and re-ran on spike after regeneration -- `llama_mlp` 119815 ppm,
`llama_attention` 150192 ppm, `llama_attention_small` 109020 ppm, `llama_attention_full`
212040 ppm (210655 vs the model's own output). Every mesh matmul bit-exact, every number unchanged
from sections 9.3 and 10.

**A tooling trap worth not repeating:** the wait loop `until ! pgrep -f "gen_llama_attn_full.py"`
never terminates, because the waiting shell's own command line contains the pattern, so `pgrep`
always matches itself. The generator had finished 5 hours earlier. Match on something the waiter
does not itself contain, or poll the output file's mtime.

### 11.5 `results/` pruned to llama + mlp

456 run records, **30.6 MB** (not the 45 MB `du` reports -- that is block overhead on ~1400 tiny
files). Deleted the 164 `linear` and `attention` records, 6.3 MB. Now 292 records, 24.3 MB.

Two things that had to be checked first, both of which would have been silent damage:

* **`results/baseline_20260908.json` is not a run record.** It is the frozen differential baseline
  from `merlin_glue_port_plan.md` Step 0 -- the sha256-of-output gate every later change is measured
  against -- and it sits loose at the root of `results/`, so a "delete everything that is not
  llama/mlp" pass takes it out. Kept.
* **It names seven specific run directories**, two of which (`20260908-233711_linear_64x64x64`,
  `20260908-233727_attention_...`) are exactly the kernels being pruned. Those two are preserved as
  the baseline's evidence; deleting them would have left the gate pointing at nothing. Verified all
  seven still resolve after the prune.

`mlp2d`..`mlp8d` records (7 runs, 0.1 MB) were **kept**, unlike their `out/build` counterparts in
11.3. With those build directories gone and unreproducible, these records hold the only surviving
measurements from those sweeps, and they cost 0.1 MB.

### 11.6 Stripping the arrays (done)

Per record: `hardware_output.npy` + `fp32_reference.npy`, `config.json`, `metrics.json`,
`log.jsonl`. The split was **91.2% `.npy`**, 5.7% `.json`, 3.2% `.jsonl` -- the graded history
itself (every `VERDICT`, recipe, cycle count) was under 3 MB of the 24.3 MB.

Stripped the arrays from 271 of 292 records: **24.3 MB -> 3.7 MB**, 542 arrays removed.

**The hash had to be preserved first, and it was not already recorded.** `metrics.json` has no hash
at all, and the only `sha256` in `config.json` is `provenance.libgemmini_id.sha256` -- the *model's*
identity, not the output's. The output hash existed in exactly one place: `output_sha256` in
`baseline_20260908.json`, for 7 runs. Deleting the arrays would have destroyed it for the other 285.

So before removing anything, every record got an `arrays.sha256.json` sidecar (~150 B) recording
each array's hash, shape, dtype and original size. The convention was reverse-engineered rather than
assumed: the baseline's `output_sha256` is `sha256(arr.tobytes())` -- the **array bytes**, not the
`.npy` file bytes, which hash differently because of the numpy header. Verified by reproducing all
7 baseline entries exactly, then re-verified that the 7 sidecars agree with the gate.

Arrays were **kept** for 21 records: the 7 the baseline references, plus the newest run of each of
the 16 kernels (2 overlap). So the frozen gate still re-verifies from real data -- confirmed after
the strip -- and the current state of every kernel is still checkable bit-for-bit. Older runs keep
their verdicts and their hashes, and can still be compared to each other by hash; they just cannot
have a *new* statistic computed over their outputs.

`results/` is now 3.7 MB of content across 292 records, with every `metrics.json` and sidecar intact.

## 12. The FPGA bisection ladder (2026-09-20)

**The report (user).** The llama kernels fail *catastrophically* on an FPGA MxGemmini bitstream
while PASSING on spike, and `matmul_tiled_fp8_64x64_chain` plus the 128x128 fp8 test PASS on that
same bitstream. The kernels run to completion and print, and **every mesh stage differs -- including
`Q`, the very first matmul.** The bitstream is DIM = 16, and the failing ELFs were built well before
the 2026-09-19 dim32 work.

So this is not "MX fp8 on RTL", and it is not a numeric drift. Something llama's matmuls do that the
ISA tests do not is wrong from the first instruction stream, and the list of candidates is short
enough to enumerate and test.

### 12.1 What `Q` actually does differently

`Q = Xn @ Wq` is `[32,2048] x [2048,64]`. Against `matmul_tiled_fp8_64x64` (`I=J=K=4`):

| delta | `Q` | the passing ISA tests |
|---|---|---|
| tile grid | `I=2, J=4` -- **non-square** | 4x4 and 8x8, always square |
| reduction depth | `TK=128` in ONE `loop_ws` | 4, or 8 at 128x128 |
| A-side scale window | 2048 B | 128 B |
| B-side scale window | **4096 B** | 128 B (512 B at 128x128) |

Three other features appear further down the kernel (K-tiled accumulate, output-region reuse, the
requant->spad resident seam, a strided A mvin) and two of them are recorded here as **unverified on
RTL** (10.3).

**Ruled out before building anything.** The B-side scratchpad tile ordering looks contradictory --
`matmul_tiled_fp8_64x64.c` places B at `(j*tiles_K + k)*DIM` and `mvin_B` at `(k*tiles_J + j)*DIM`
-- but both agree with what `loop_ws` walks (`B_sp_addr_start + (k*J + j)*DIM`, `gemmini.cc:808`
and `:1291`). The ISA test swaps its loop variable names and is square, so the apparent transpose
cancels. Not a bug.

**Also ruled out: the K-tiled accumulate, for THIS failure.** At `BANK_ROWS 4096` the projections
pick a single K-tile spanning all of D, so `Q` issues one `loop_ws` with `ex_accumulate = 0` and a
contiguous A mvin. The K-tiling only appears in a `BANK_ROWS 2048` build -- which is what
`include/gemmini_params.h` currently selects, and which is the wrong geometry for a DIM=16
bitstream. See 12.4.

### 12.2 The ladder

`baremetal/mxgemmini/src/mx_ladder.c` plus ten two-line rungs `src/mxl0.c` .. `src/mxl9.c`, data
from `gen/gen_mx_ladder.py`. Each rung is **one delta** from the shape that already passes, so the
first rung that fails on the FPGA names the feature and everything below it is a consequence.

| rung | the one delta | isolates | ELF |
|---|---|---|---|
| `mxl0` | none: 64x64x64, through llama's OWN helpers | the calling convention, not the shapes | 36 KB |
| `mxl1` | M=32 -> `I=2, J=4` | a **non-square tile grid** | 30 KB |
| `mxl2` | K=256 -> `TK=16` | a deeper reduction | 49 KB |
| `mxl3` | K=1024 -> `TK=64`, 2 KB B scales | depth and scale-window size | 125 KB |
| `mxl4` | K=2048 -> `TK=128`, 4 KB B scales | **exactly `Q`** | 226 KB |
| `mxl5` | `mxl4` as 2 accumulating K-tiles | **`ex_accumulate` on RTL** (10.3) | 227 KB |
| `mxl6` | two matmuls into one `C_spad` | **region reuse / smem clear on RTL** | 41 KB |
| `mxl7` | requant->spad tiled + resident scales | the `P@V -> o_proj` seam, non-square | 34 KB |
| `mxl8` | A as a column slice, strided mvin | the DMA row pitch | 158 KB |
| `mxl9` | no mesh at all -- `mx_host.h` vs golden | the Rocket's fp32 / `expf` | 252 KB |

Goldens come from `fp8_matmul_model.tiled_matmul_hwlike` through `gen_matmul_llama._run_mesh` -- the
same model the llama headers use -- so a rung passing means what `llama_attention` passing means.
Operands are seeded pseudo-random rather than captured tensors: the ladder tests STRUCTURE, and a
self-contained generator needs neither the capture npz nor a HuggingFace download.

`mxl0`..`mxl5` and `mxl8` are one driver: a plain rung is the K-tile loop at `LAD_KTILES = 1`, which
is exactly how `llama_attention.c` degenerates on a 16384-row scratchpad. The shared path is the
honest one, not a convenience.

**Result: all ten PASS on spike**, which is the precondition for reading anything into an FPGA
failure -- a rung failing there would be a bug in the ladder.

```
make ladder          # both targets;  make run-ladder  runs them on spike in order
```

### 12.3 Two things the ladder fixed on the way

* **`include/mx_mesh.h`.** The mvin/mvout/`mesh_matmul` helpers now live in one header, lifted
  verbatim from `llama_attention.c`. `mxl0`'s claim is "the helpers, not the shapes", and a second
  copy of those helpers would make that a tautology the moment the two drifted.
  `llama_attention.c` and `llama_mlp.c` still carry their own copies; switching them over is a
  separate change, deliberately not bundled with a debugging artifact.
* **A `#define DIM 16` that lies silently.** Every kernel here overrides `gemmini_params.h`'s `DIM`
  and says nothing about it. The ladder `#warning`s when the two disagree, which is the difference
  between "this ELF is pinned to the goldens it carries" and a dim32 bitstream quietly running
  dim16 tiling.

### 12.4 `gemmini_params.h` is shared, and it moves

Commit `abd690d` set it to `DIM 16 / BANK_ROWS 4096 / ACC_ROWS 512`; `22be6a0` set it to
`DIM 32 / BANK_ROWS 2048 / ACC_ROWS 256`. It is one file, shared with the ISA suite, flipped per
bitstream. `DIM` is harmless for these kernels because they override it -- **`BANK_ROWS` is not
overridden**, and it silently replans every kernel:

| | `BANK_ROWS 4096` | `BANK_ROWS 2048` |
|---|---|---|
| projections | 1 K-tile of 2048 | 2 K-tiles of 1024, i.e. `ex_accumulate` |
| `o_proj` / `down_proj` | 2 chunks of 1024 | 4 chunks of 512 |

A DIM=16 build made while the header says 2048 therefore takes a schedule that depends on
`ex_accumulate` -- the one thing 10.3 flags as unverified on RTL -- for no reason. Reverted to the
dim16 geometry for this work; `mxl5` is what decides whether the 2048 schedule is usable at all.

### 12.5 Sliced llama kernels, for reading a failure rather than bisecting one

`llama_mlp` had no sliced variant at any size and `llama_attention` stopped at D=256. Added
`llama_mlp_small` (D=256), and `llama_mlp_tiny` / `llama_attention_tiny` at **D=64** -- the smallest
hidden size the E8M0 block (32) and the mvin tile (DIM) allow. `llama_mlp.c` gained the
`LLAMA_MLP_HEADER` hook `llama_attention.c` already had, so one source builds them all.

| kernel | D | header | ELF | rel_fro |
|---|---|---|---|---|
| `llama_mlp_small` | 256 | 0.7 MB | 161 KB | 118732 ppm |
| `llama_mlp_tiny` | 64 | 0.2 MB | 67 KB | 120889 ppm |
| `llama_attention_tiny` | 64 | 0.4 MB | 104 KB | 99856 ppm |

All three PASS on spike with every mesh stage bit-exact, and `llama_mlp` at D=2048 is unchanged at
119815 ppm -- so the header hook cost nothing. At D=64 the projections collapse to one K-tile and
the output to one chunk, so these kernels are *structurally smaller*, not just faster: if
`llama_attention_small` fails on hardware and `llama_attention_tiny` passes, the fault scales with
D or with the chunk count.

### 12.6 How to use this on the FPGA

Run `out/baremetal/mx_rocket/mxl0` .. `mxl9` **in order**. Every rung prints its own `DIM`, the
scratchpad it planned against, its spad map, a PASS/FAIL line and the first eight mismatching
elements -- the PATTERN of the mismatch is the diagnosis (all-wrong vs one-tile-wrong vs
every-other-column-wrong are three different bugs) and a bare count cannot tell them apart.

Two rungs test a hypothesis rather than just reporting: `mxl5` says outright that a failure with
`mxl4` passing means the RTL is discarding `ex_accumulate`, and `mxl6` counts how many of its
outputs equal `bf16(golden#1 + golden#2)` -- a high count being the signature of a shadow
accumulator that was never cleared.

### 12.7 MEASURED on RTL (MxGemminiRocketConfig, VCS, 2026-09-20)

| rung | shape | K-groups | A win | B win | RTL |
|---|---|---|---|---|---|
| `mxl0` | 64x64x64 | 2 | 128 B | 128 B | **PASS** |
| `mxl1` | 32x64x64, `I=2 J=4` | 2 | 64 B | 128 B | **PASS** |
| `mxl2` | 32x256x64 | 8 | 256 B | 512 B | **PASS** |
| `mxl3` | 32x1024x64 | **32** | 1024 B | 2048 B | **PASS** |
| `mxl4` | 32x**2048**x64 | **64** | 2048 B | 4096 B | **FAIL 2036/2048** |
| `mxl5` | the same, 2 accumulating K-tiles of 1024 | 32/call | 1024 B | 2048 B | **PASS** |
| `mxl6` | two matmuls into one `C_spad` | — | — | — | **PASS** |
| `mxl7` | requant->spad + resident, `I=2 J=4 Kt=2` | — | — | — | **FAIL: 0/2048 codes, 36/64 scales** |

**Four things this settles, three of which were open questions in this file.**

1. **A non-square tile grid is FINE.** `mxl1` passes at `I=2, J=4`. The leading hypothesis for
   "every mesh stage differs" was wrong.
2. **`ex_accumulate` IS honoured by the MX RTL, and cross-call accumulation is BIT-EXACT.** `mxl5`
   is `mxl4`'s matmul split into two accumulating K-tiles, graded against `mxl4`'s single-shot
   golden, and it returns 0/2048. Section 10.3's "**UNVERIFIED ON RTL**" is now verified, in the
   affirmative. So is 10.3's other half: `mxl6` shows `ex_accumulate = 0` really does OVERWRITE an
   output region on RTL, i.e. the shadow accumulator is cleared.
3. **A single `loop_ws` breaks somewhere between K=1024 and K=2048.** Values come back with the
   right sign and order of magnitude but 5-60% wrong -- a reduction that is not summing what it
   should, not a broken address walk. Note the reduction crosses **32 E8M0 groups exactly** at the
   cliff, which is what a 5-bit group index would do. Four things move together there and
   `mxl10`..`mxl14` separate them (see the truth table in `gen/gen_mx_ladder.py`).
4. **The requantizer computes the right values and writes the wrong output scales.** `mxl7` gets
   `0/2048` FP8 codes -- deposited correctly in the tiled operand-A layout -- and `36/64` E8M0
   bytes wrong. `Y` then fails as a consequence, since it reads those scales resident.

   `matmul_tiled_fp8_64x64_chain` PASSES on this bitstream **and checks the DRAM scales the same
   way `mxl7` does** (`sf[i*GN + b] == C1_scales_out[i][b]`, i.e. `[M][GN]`), so this is not a
   plain transposed write -- that would break the chain test too. What differs is the tile counts:
   the chain test is `I=J=Kt=4`, `mxl7` is `I=2, J=4, Kt=2`. `mxl15`..`mxl18` move `I`, `J` and
   `Kt` one at a time off the chain test's own shape, and the requant rung now DUMPS all `M*GN`
   scale bytes plus two layout hypotheses, because a count cannot tell a transpose from a wrong
   row pitch and those are different bugs.

### 12.8 What this means for the kernels, right now

* **`llama_mlp` has no requant stage at all** -- `G`, `U` and `Y` are all BF16 non-requant. So the
  ONLY thing standing between it and a passing RTL run is the K=2048 projection, and `mxl5` proves
  the fix is exact: cap the per-`loop_ws` K at 1024 and let the K-tiles accumulate. A
  `BANK_ROWS 2048` build already does exactly that by accident -- the "wrong" geometry of 12.4
  picks the working schedule.
* **`llama_attention` needs both fixes.** Its `O = P @ V` is `M=32, K=32, N=64` -- `I=2, J=4,
  Kt=2`, which is `mxl7`'s shape exactly. Capping K fixes Q/K/V and `o_proj`'s operand side, but
  the resident seam stays broken until the requantizer's scale write is. Nor does the `llama_
  attention_full` workaround of 10.4 (mvout the scales and transpose them on the host) help: the
  DRAM copy is wrong too.
* `llama_attention_small` / `_tiny` (D=256, D=64) already take one K-tile under 1024, so their
  projections should pass on RTL and they should fail at `O = P @ V` -- a prediction worth checking,
  because it would confirm the two faults are independent.

### 12.9 Applied: the K cap, and the geometry pin (2026-09-20)

**`LLAMA_KTILE_MAX = 1024`** in `llama_mlp.c` and `llama_attention.c`, labelled as a WORKAROUND for
the RTL fault rather than a design choice, overridable with `-DLLAMA_KTILE_MAX=2048` to reproduce
the failure. Both now plan `2 K-tile(s) of 1024` where they planned `1 of 2048`, and every graded
number on spike is UNCHANGED -- 119815 / 313 ppm for the MLP, 150192 / 609 for attention -- which is
`mxl5`'s bit-exactness reproduced at the kernel level.

**`LLAMA_BANK_ROWS = 4096`** pinned in all three big kernels, with a `#warning` on a mismatch. This
was learned rather than designed: mid-bisection `include/gemmini_params.h` flipped back to
`DIM 32 / BANK_ROWS 2048` (it is shared with the ISA suite and tracks whichever bitstream is being
built -- 12.4), and `mxl4` stopped COMPILING because its `LAD_REQUIRE(fits)` fired. That guard is
the only reason this surfaced as a build error instead of a silently retuned run: `DIM` was already
overridden in every kernel, `BANK_ROWS` was not, and 9.3's deliberate scratchpad adaptivity means a
flip produces *correct numbers for the wrong machine*. The ladder pins both (`LADDER_DIM`,
`LADDER_BANK_ROWS`) and warns; the adaptivity survives as an override.

**Predictions now on the table**, worth running because each one discriminates:

| kernel | expected on RTL | why |
|---|---|---|
| `llama_mlp`, `_small`, `_tiny` | **PASS** | no requant stage at all; the capped projection was the only fault |
| `llama_attention`, `_small`, `_tiny` | Q/K/V/S correct, **fail at `O = P@V`** | that stage is `mxl7`'s shape exactly |
| `llama_attention_full` | still fails in the projections | its `mesh_matmul(LLAMA_M, LLAMA_D, nc, ...)` keeps the full 2048 contraction in ONE call and has no K-tiling to cap -- adding it there means threading `ex_accumulate` through the projection loop, which is a real change, not a macro |

If `llama_mlp` passes and `llama_attention` fails exactly at `O = P@V`, the two RTL faults are
confirmed independent and the requantizer's scale write is the only thing left between this repo
and a real attention layer on hardware.

### 12.10 Root causes, and where they are written up (2026-09-21)

The two faults separated and both now have dedicated files; this section is only the index.

* **`planning/rtl_mx_faults_handoff.md`** — both faults as measured, exact repro commands, the
  simulator caveats, and the full ruled-out list. Written as a standalone handoff.
* **`planning/rtl_fault_b_kdepth.md`** — Fault B's root cause.

**Fault A (requantizer E8M0 output scales): FIXED.** A regression from the mesh-widening commits
`46736b1..9eb04f3`, which generalized `MxRequantizer.scala`'s `scale_e8m0` from one wire to
`Vec(numOutBlocks, ...)`. Only block 0 was ever written; block 1 read back as 0
(`WireDefault(0.U)`). Measured: `GN=1` completely clean (`mxl18`, 0/64 scales and 0/4096 on Y),
`GN=2` always broken regardless of `I`/`J`/`Kt`, FP8 codes perfect throughout. Both
`matmul_tiled_fp8_64x64_requant` and `matmul_tiled_fp8_64x64_chain` catch it — the latter being a
test the FPGA bitstream passes, which is the before/after in one line. `mxl7` now passes.

**Fault B: NOT "K=2048".** The label was wrong, and §12.7's "crosses 32 E8M0 groups exactly" was a
red herring. The governing quantity is the **B-side scale window**:

```
rows required = (N/16) * (K/32) = N*K/512 = the B-side scale window in bytes / 16
rows readable = 2 banks x 64 rows = 128        (ScaleFactorMem.scala, read_row_addr_w is 7 bits)
```

32 groups was the capacity divided by J, not a group-index width. `mxl3` needs exactly 128 rows and
passes; `mxl4` needs 256 and wraps. The write path is wider than the read path, so the upper half of
the scales lands in the double buffer's other half and the read wraps to row 0 — **the second half
of the reduction is scaled by the first half's scale bytes**, which is precisely the observed
right-sign, right-magnitude, 5-60%-wrong signature.

**The software cap is expressed in the wrong variable.** `LLAMA_KTILE_MAX = 1024` is correct only
while N <= 64; the real invariant is `N*K <= 65536`. Worth generalizing even after the RTL is fixed.

Predicted and decisive: `mxl12` (N=32, K=2048 — **64 groups**) should PASS, and `mxl13` (N=128,
K=1024 — **32 groups**) should FAIL. No hypothesis about group count or reduction depth predicts
that inversion; only the `J*G` product does.

### 12.11 Workaround removed, replaced by the real invariant (2026-09-21)

Both RTL faults are fixed and all 18 mesh ladder rungs pass (mxl9, the host-glue rung, is still
untested -- it costs 10.65M cycles, past the default `+max-cycles`). So the `LLAMA_KTILE_MAX = 1024`
cap of 12.9 came out. It was **replaced rather than deleted**, because the cap was expressed in the
wrong variable and would have been wrong again the moment a kernel widened its output:

```c
#define LLAMA_SCALE_ROWS_MAX 256                  // ScaleFactorMem, per double-buffer half
#define SCALE_ROWS(k, n) ((n) * (k) / 512)        // one matmul needs (N/16)*(K/32) rows
#define KFITS(kt) (P_COST(kt) <= SPAD_TOP && SCALE_ROWS((kt), LLAMA_H) <= LLAMA_SCALE_ROWS_MAX)
```

The scale window is a SECOND budget alongside the scratchpad, and on a deep contraction it binds
first. Bounding on `N*K` instead of `K`:

* the cap **lifted itself** when the RTL went from 128 usable rows to 256 -- `D = 2048` at `H = 64`
  needs exactly 256, so the projections are one K-tile again with no edit;
* it still catches a future kernel that widens N, where a K-only cap would not;
* `llama_attention_full`'s `PROJ_N` is now chosen against it too. At D = 2048 the scratchpad already
  forces 64, but a larger scratchpad would have silently picked 128 and wrapped the scale rows;
* every kernel asserts it at COMPILE time (`proj_scale_fits`, `oproj_scale_fits`, …). Fault B was
  invisible for exactly as long as nothing asserted this.

Rebuilt and re-run on spike: all seven kernels PASS, schedules back to `proj 1 K-tile(s) of 2048`,
and **every graded number identical** to before the workaround went in -- 119815/313 ppm for the
MLP, 150192/609 for attention, 212040/210655 for the full 32 heads.

### 12.12 What is left

`baremetal/mxgemmini/llama.jobs` queues all seven on RTL, cheapest first.

**`MAX_CYCLES` has to be raised.** These kernels are host-glue-dominated -- `llama_attention` is
mesh 15467 against host 12.6M cycles (8.3) -- so the full-D ones exceed the default
`+max-cycles=10000000` and cannot finish at it, for the same reason `mxl9` never did. The D=64 and
D=256 variants are 8-30x cheaper and are the right first check.

`llama_attention_full` has **never run on RTL**. Its projection is `K=2048 @ N=64` = 256 scale rows,
exactly the new ceiling, and it needs `ex_accumulate` and output-region reuse, both of which
`mxl5`/`mxl6` confirmed on RTL. If it passes it is the first hardware number for the whole attention
sub-layer, gradeable against TinyLlama itself.

## 13. Scaling up: the ladder to a whole model on spike (2026-09-21)

**Goal (user):** keep lowering larger and larger blocks of llama until the FULL model runs on spike.
`llama_attention_full` (all 32 heads) and `llama_mlp` (64 of 5632 neurons) were the starting point.

| rung | what | blob | goldens | spike | state |
|---|---|---|---|---|---|
| **R1** `llama_mlp_full` | ALL 5632 FFN neurons | 37.2 MB | 47 s | ~3 min | **DONE** 2026-09-21 |
| **R2** `llama_layer` | one COMPLETE decoder layer, graded vs `h_out` | ~48 MB | ~1 min | — | next |
| **R3** `llama_layers_N` | N stacked layers, residual carried forward | 45 MB x N | ~10 min @ 22 | — | — |
| **R4** `llama_full` | embed + 22 layers + final norm + `lm_head` -> logits + perplexity | ~1.07 GB | ~15 min | ~25 min | — |

### 13.1 MEASURED: the golden model, not spike, was the thing that did not scale

`fp8_matmul_model.tiled_matmul_hwlike` is the reference every kernel here is graded against, and it
runs at **0.42 MMAC/s**, essentially flat in N:

```
M=32 K=2048 N=64       4.2 MMAC    11.22s
M=32 K=2048 N=256     16.8 MMAC    39.54s
M=32 K=2048 N=1024    67.1 MMAC   160.86s
```

One MLP sub-layer is 1.1 GMAC, a decoder layer ~1.4 G, the whole model ~33 G -- **22 hours** on one
core. Spike itself is 60x faster than that (27 MMAC/s measured on `llama_attention_full`), so the
generator, not the simulator, was the wall.

It is removable, and section 10.2 had already established why: **an output column depends only on
its own column of B.** `gen/mesh_par.py` splits the columns across a process pool (this box has 256
cores), and its self-test checks the result **bit-identical on raw bits** -- not `allclose` -- to the
serial path, because every golden in the repo would silently inherit any error here. Measured on R1:

```
G = Xn @ WG  [32,2048]x[2048,5632]  369 MMAC  14.8s on 176 cores  (25.0 MMAC/s)
U = Xn @ WU  [32,2048]x[2048,5632]  369 MMAC  14.4s on 176 cores  (25.6 MMAC/s)
Y = H  @ Wd  [32,5632]x[5632,2048]  369 MMAC  17.7s on  64 cores  (20.9 MMAC/s)
```

**The split is on N ONLY.** The model accumulates along K across tiles in bf16, so splitting K would
change the accumulation order and therefore the VALUE -- the golden would stop being the datapath's.

### 13.2 R1: the full MLP, and the two budgets that shaped it

At F = 5632 the two budgets bind in OPPOSITE directions, which is what makes this a new shape rather
than a bigger `llama_mlp.h`:

* **the scratchpad caps the OUTPUT** -- `[M][F]` as BF16 is 22528 rows against 16384, so gate/up
  cannot land whole however they are placed. Chunked on F (88 x 64).
* **the scale window caps the CONTRACTION** -- `down_proj` contracts over K = 5632, needing 704
  B-side rows at N = 64 and 352 A-side rows at M = 32, against 256. So it MUST be K-tiled. This is
  the first kernel forced into that at full scale, and it runs **44 K-tiles of 128**, bit-exact
  against a golden computed with the contraction whole. That is `mxl5`'s property holding at 44x
  rather than 2x.

**The A-side scale term is newly asserted.** `llama_mlp.c` only ever bounded the B side, which is
correct exactly while `N >= M` -- true for every earlier shape and NOT true in general. R1 asserts
`SCALE_ROWS(KTILE, LLAMA_M)` as well (`down_ascale_fits`, `proj_ascale_fits`).

**The matmul count is invariant.** Both budgets reduce to a cap on the PRODUCT `K*N` per call
(131072), so any schedule issues the same 88 matmuls per projection; the tiling decides DMA, not
work. That is why the plan prefers the fewest K-tiles: at `PKTILE == D` the projection contracts in
one call and Xn is moved in ONCE for all 176 matmuls.

Result on spike, every mesh stage bit-exact:

```
llama MLP FULL: M=32 D=2048 F=5632 (ALL neurons)
plan  spad 16384 rows: proj 88 chunk(s) of 64 x 1 K-tile(s) of 2048 | down 2 chunk(s) of 1024 x 44 K-tile(s) of 128
mesh  G = Xn @ Wg : 0/180224    mesh  U = Xn @ Wu : 0/180224
mesh  Y = H @ Wd  : 0/65536  (2 chunks of 1024, 44 K-tiles of 128)
grade MLP out vs fp32 reference      : rel_fro 112572 ppm
grade MLP out vs THE MODEL's own out : rel_fro 112565 ppm
cycles mesh 1831008, host 50370217
```

`MLP_TORCH` is new and is the point: with every neuron present `down_proj` is a complete reduction,
so the result IS the layer's real MLP output and `layer.mlp`'s own forward hook is a reference the
sliced capture could not have. `--all-neurons` gates on it at capture time (rel 4.40e-3, bf16
forward vs fp32), the same way `--all-heads` gates on `attn_torch`.

### 13.3 What R4 costs, stated now rather than discovered later

The full model is **~1.07 GB** of blob: 22 layers x 45.4 MB plus a 67.6 MB `lm_head`
(`tie_word_embeddings: false`, vocab 32000). That is simply 1.1B params at 8 bits plus E8M0 scales,
and it is not reducible by tiling. Spike's default is `-m 2048` MiB, so R4 needs it raised; this box
has 2.2 TB of RAM and 3.3 TB of disk, so neither generation nor execution is constrained.

**Per-stage goldens stop at R2.** Storing a bf16 golden for every mesh stage of 22 layers would
roughly double the blob for a gate that R1/R2 already provide. R3/R4 grade the final logits against
torch plus wikitext2 perplexity, with the per-stage check available behind `-DCHECK_ALL_STAGES`.

### 13.4 R2: one COMPLETE decoder layer (2026-09-21) -- DONE

`src/llama_layer_full.c` + `gen/gen_llama_layer_full.py`, 47.9 MB blob. Attention (all 32 heads) and
the MLP (all 5632 neurons) joined by both RMSNorms and both residuals, as
`LlamaDecoderLayer.forward` does it. **PASSES on spike with every mesh stage AND both residual seams
bit-exact.**

```
plan  spad 16384 rows | attn: proj N=64, 32 heads, o_proj 2 x 1024
plan  mlp: proj 88 x 64 (1 K-tile of 2048), down 2 x 1024 (44 K-tiles of 128)
mesh  Q/K/V 0/65536, 0/8192, 0/8192   S 0/32768   O 0/65536+0/2048   Yattn 0/65536
host  h_mid = h_pre + Yattn : 0/65536       <-- THE SEAM
mesh  G 0/180224  U 0/180224  Ymlp 0/65536 (44 K-tiles)
host  h_out = h_mid + Ymlp  : 0/65536
grade attention out vs the model's : 210655 ppm
grade MLP out       vs the model's : 165941 ppm   (on the DEVICE's h_mid)
grade LAYER OUT h_out vs THE MODEL's h_out : 5217 ppm
cycles mesh 2397247, host 82252535
```

**THE SEAM IS THE RESULT, not the plumbing.** `llama_mlp_full` is handed torch's `h_mid`; this
kernel's MLP runs on the one the DEVICE produced -- `h_pre` plus an attention output carrying the MX
error of 32 heads and four matmul stages. Measured consequence:

| quantity | standalone | inside the layer |
|---|---|---|
| h_mid vs the model's own | (given) | **0.3186% drift** |
| MLP output vs the model's | 11.2575% | **16.5951%** |

**The errors COMPOUND, they do not add** -- +5.34 points on the MLP purely from a 0.32%-perturbed
input, because RMSNorm of a perturbed vector is a different vector and its fp8 codes differ. No
amount of per-sub-layer testing predicts this, and it is the quantity that will govern R3/R4: 22
layers compound 22 times. Every `M_*` golden was re-derived from the device's `h_mid`; none of
`llama_mlp_full.bin`'s bytes are reusable.

**The layer output is nonetheless 0.52%**, far below either half, because the residual stream is 37x
larger than the MLP output it carries. Both facts matter: the sub-layer error is large, and the
layer-to-layer signal is well preserved. Which of those dominates after 22 layers is exactly what R3
measures and neither number predicts.

**Two traps handled up front:**

* **`-0.0`.** `gen_matmul_llama.bf16_bits` forces `+0` where `mx_host.h:mx_f32_to_bf16_rne` keeps the
  sign bit. Harmless for every previous golden; NOT harmless here, because the residual's result is
  re-quantized and one ulp flips an E4M3 code. `gen_llama_layer_full._bf16_rne` mirrors the C.
* **Blob name collisions.** Both chains want `XN_CODES`, `XN_SCALES`, `Y_OUT`, and `Blob.off` is a
  dict -- a duplicate silently overwrites an offset while both copies still occupy the buffer, so the
  header would point one stage's operand at another's bytes. The chains take a `prefix` (`A_`/`M_`).

**The two sub-layer generators were refactored into reusable chains** (`attention_chain`,
`mlp_chain`) and switched to the parallel golden. Both refactors were proven inert by regenerating
each blob and checking it **byte-identical** (md5 + `cmp`) to the pre-refactor artifact --
`llama_attn_full.bin` also dropped from ~12 min to 42 s.

### 13.5 Next: lower a layer through merlin (user, 2026-09-21)

The ladder so far is HAND-WRITTEN C at the ISA-intrinsic level -- every `gemmini_*` call is one
`.insn r`, and the tiling, scratchpad addresses, scale-window loads and `ex_accumulate` bits are all
chosen by hand. The user's direction: **R2 by hand, then lower it through merlin.**

What that needs, from `kernels/registry.py`'s own statement of its limit ("a new file is only needed
for something that is not a chain of matmuls ... those need new lowering in the backend"): the
merlin target today expresses CHAINS OF MATMULS (`linear`, `mlp2`). A decoder layer is not one --
RMSNorm, RoPE, causal softmax, SwiGLU and the residual all sit between the matmuls and none has
lowering. `mx_host.h` already has all of them as unit-checked fp32 C.

**R1 and R2 are the conformance target.** They are graded bit-exact per stage, so the compiler's
output can be required to match them BYTE for BYTE rather than merely "run" -- a far stronger gate
than either track has had, and the reason to do them in this order.

### 13.6 R3/R4: the stacked model (2026-09-21)

R3 and R4 are ONE kernel, not two: `src/llama_model.c` loops over layers, so the layer count lives
in the DATA and a 2-layer bring-up build and the full 22-layer model are the same source. The blob
holds NL identically-laid-out blocks and layer n is addressed as
`LAYERS + n * LLAMA_LAYER_STRIDE + LOFF_<name>`; the identical layout is ASSERTED at generation
time rather than assumed.

New artifacts:

| file | what |
|---|---|
| `app/capture_llama_model.py` | one forward pass, hooks on every layer -> `out/model_capture/layer<N>.npz` (22 x 178 MB) + `model.npz` (embed out, final norm, lm_head, true logits, labels). One file per layer because a single npz would be ~4 GB and could not be loaded incrementally. |
| `gen/gen_llama_model.py` | chains the layers, each on the PREVIOUS layer's device output; emits the uniform-stride blob |
| `src/llama_model.c` | the kernel: layer loop + final RMSNorm + lm_head + on-device NLL/perplexity |
| `src/llama_model_l2.c` | the 2-layer bring-up shim (169 MB instead of ~1 GB) |

**THE FULL PER-STAGE BIT-EXACT GATE IS KEPT FOR EVERY LAYER.** 13.3 had assumed goldens would have
to be dropped at this scale. That was wrong, and the error is worth recording: goldens here are
ACTIVATION-sized (`[M] x something` at M = 32), not weight-sized -- ~1.2 MB per layer against ~45 MB
of weights, so all 22 layers' goldens cost ~26 MB on a ~1.05 GB blob, **2.4%**. A divergence is
therefore localized to a stage of a layer instead of only appearing in the logits.

**Reference (torch, fp32, the same 32 wikitext2 tokens): nll 3.350216, ppl 28.5089.**

#### Bring-up on 2 layers -- PASSES, everything bit-exact

```
llama MODEL: 2 layer(s)  M=32 D=2048 F=5632 heads=32 kv=4 + lm_head
layer  0  mesh 0  host 0  seam 0  |  h_out vs model: rel_fro  84860 ppm
layer  1  mesh 0  host 0  seam 0  |  h_out vs model: rel_fro 124896 ppm
mesh  logits = Xf @ W_lm : 0/1024000 differ (500 chunks of 64)
cycles mesh 6796331, host 423358586
llama MODEL test PASSED (2 layer(s) + lm_head; every mesh stage, every seam and every host stage bit-exact)
```

The drifts reproduce the generator's 8.4869% / 12.4907% exactly and the device's own NLL matches the
python to three decimals, so the layer loop, the stride addressing, the cross-LAYER seams and the
head are all validated before the expensive build.

**A TRUNCATED STACK PLUS A HEAD IS NOT A MODEL.** `lm_head` reads the hidden state after ALL 22
layers; over 2 it yields well-formed logits of nothing -- ppl **35026** against the model's 28.5089,
reproduced identically by the kernel and the generator. That is correct behaviour, not a failure,
and the generator now warns, the generated header repeats the warning, and `llama_model_l2.c`
documents it. Its bit-exact gate is still valid; only the accuracy numbers are meaningless.

#### Early layers drift far more than layer 5 did

| layer | h_out drift vs the model |
|---|---|
| 0 | 8.4869% |
| 1 | 12.4907% |
| 5 (R2, given torch's input) | 0.5217% |

Not a contradiction: the residual stream GROWS with depth (`|h_out|` is 0.5 at layer 0 against 140
at layer 5), so the same absolute MX error is relatively enormous early and small later. Whether
that compounding keeps up with the growing stream is the question R4 answers, and neither number
predicts it.

### 13.7 R4: THE WHOLE MODEL ON SPIKE (2026-09-22) -- DONE

**TinyLlama-1.1B, all 22 decoder layers plus the final RMSNorm and lm_head, as one MXFP8 chain on
MxGemmini. PASSES on spike with EVERY mesh stage, EVERY residual seam and EVERY host stage
bit-exact against the golden.**

```
llama MODEL: 22 layer(s)  M=32 D=2048 F=5632 heads=32 kv=4 + lm_head (fp8 e4m3 + E8M0)
plan  spad 16384 rows | attn proj N=64, o_proj 2 x 1024 | mlp proj 88 x 64, down 2 x 1024 (44 K-tiles of 128)
layer  0..21   mesh 0  host 0  seam 0   (all of them)
host  final rmsnorm+quant: 0 byte(s) differ
mesh  logits = Xf @ W_lm : 0/1024000 differ (500 chunks of 64)
grade logits vs torch      : rel_fro 130693 ppm
grade argmax agrees on 28/32 tokens
grade nll  MX 3.492  vs torch 3.350
grade ppl  MX 32.863  vs torch 28.507
cycles mesh 55078167, host 2093926309
llama MODEL test PASSED (22 layer(s) + lm_head; every mesh stage, every seam and every host stage bit-exact).
```

Blob 1120 MB (1.1B params at 8 bits + E8M0 scales + 26 MB of goldens); ELF 1.12 GB; `spike -m4096`.
Generation 2183 s on 256 cores; the spike run ~1 h. The kernel's own on-device perplexity
arithmetic reproduces the generator's to the printed precision (32.863 vs 32.8657).

#### THE HEADLINE NUMBER

**wikitext2 perplexity 32.86 in MXFP8 against 28.51 in fp32** on these 32 tokens -- a 15%
degradation -- with the argmax agreeing on 28 of 32 tokens.

#### THE ERROR DOES NOT COMPOUND WITH DEPTH

This was the open question of 13.6, and the answer is the opposite of the naive expectation:

| layer | h_out drift | layer | h_out drift |
|---|---|---|---|
| 0 | 8.486% | 11 | 14.470% |
| 1 | 12.490% | 12 | 14.484% |
| 2 | 16.751% | 13 | 14.497% |
| 3 | 16.753% | 14 | 14.512% |
| 4 | 16.759% | 15 | 14.508% |
| 5 | 16.767% | 16 | 14.532% |
| 6 | 16.779% | 17 | 14.693% |
| 7 | 14.432% | 18 | 14.594% |
| 8 | 14.444% | 19 | 14.724% |
| 9 | 14.453% | 20 | 14.805% |
| 10 | 14.470% | **21** | **21.654%** |

It rises over three layers, then is **FLAT for eighteen** (14.4-14.8%), and even falls at layer 7.
The mechanism: the residual stream GROWS with depth (|h_out| 0.5 at layer 0, 140 at layer 5), so
each layer's absolute MX error is diluted by a larger carrier. Per-layer error multiplication would
have predicted a useless model well before layer 22; that is not what this datapath does.

Layer 21 jumps to 21.65% and the logits then come back to **13.07%** -- the final RMSNorm
renormalizes, discarding the scale error that inflated the last residual.

#### What this does and does not establish

* **Does:** the whole model's arithmetic, on real weights and real tokens, matches the datapath
  model bit-for-bit at every one of ~2000 mesh stages. Any RTL divergence is now attributable.
* **Does NOT:** attribute the 32.86 between the MX FORMAT and the DATAPATH. 8.2b's decomposition
  (format ~57%, product truncation + per-lane accumulate + bf16 cross-tile the rest) was measured
  for ONE sub-layer, and has not been re-run end to end. `rtl_exact/` + the three MXQuant hooks make
  that a single run if the number is wanted.
* **Does NOT:** run on RTL. At 2.09 G host cycles this is far past any practical VCS budget; the
  D=64/D=256 sliced variants remain the RTL path.

#### 13.8 Next

The user's direction (2026-09-21), now unblocked: **lower a layer through merlin**, with R1/R2/R4 as
a byte-for-byte conformance target rather than a smoke test. See 13.5 for what the backend needs.

#### 13.7b The perplexity numbers are a 32-TOKEN WINDOW, not a benchmark (MEASURED 2026-09-22)

`ppl 32.86 vs 28.51` was written up above as "wikitext2 perplexity". **That framing is wrong and is
corrected here.** 28.51 is not TinyLlama's wikitext2 perplexity -- the standard number is ~7.7, and
the difference is the evaluation window, not the model. Measured on the stock bf16 model, standard
protocol (concatenate the test split, non-overlapping windows, mean cross-entropy):

| seqlen | windows | ppl |
|---|---|---|
| 32 | 1 (THE CAPTURE'S WINDOW) | **28.509** |
| 32 | 20 | 31.410 |
| 128 | 20 | 16.475 |
| 512 | 8 | 10.901 |
| 2048 | 4 | **9.298** |

28.509 reproduces the capture's 28.5089 exactly, so the NLL arithmetic is right; the quantity is
just a different one. **Context length is the whole story.** At M = 32 every token is predicted from
at most 31 tokens, position 0 from BOS alone, and the mean context is ~16 tokens against ~1024 at
seqlen 2048. The remaining gap from 9.30 to the published ~7.7 is sample size (4 of 166 windows).

**A hypothesis that turned out FALSE, recorded so it is not repeated:** the window's text
(`'<s> \n\n = Robert Boulter = \n\n\n\n\n'`, an article header with a proper name) was assumed to be
unusually hard and a contributing factor. It is not -- that window scores 28.5 against 31.4 for the
average 32-token window, so it is slightly EASIER than typical. Context length accounts for all of it.

**What this does and does not affect:**

* **The MX-vs-baseline comparison is unaffected.** Both numbers come from identical tokens, identical
  context and identical fp32 NLL arithmetic; only MXFP8 vs bf16 differs. The ratio
  **32.863 / 28.507 = 1.153** is a fair measure of what this datapath's quantization costs.
* **The absolute numbers must not be quoted against published perplexities.**
* **The baseline is bf16, NOT fp32.** `eval_simquant.get_model` loads with
  `torch_dtype=torch.bfloat16` (TinyLlama's native dtype, and what MXQuant uses); logits are cast to
  fp32 only for storage and the NLL. Earlier text in 13.7 calling it fp32 is wrong. Note this is a
  DIFFERENT sense of "fp32" from R1/R2's `ref_mlp`/`ref_attn`, which are fp32 numpy recomputations
  over bf16-exact operands.
* **Whether the 1.153 ratio holds at longer context is UNMEASURED.** It is plausible but not shown.

**Getting a benchmark-comparable number is real work, not a flag.** `M = 32` is forced by the
scratchpad: `Xn` alone is `M*D/16` rows -- 4096 at M = 32, and 16384 (the ENTIRE scratchpad) at
M = 128, leaving nothing for a weight tile. Longer context needs the projections K-tiled and the
`[M][M]` score matrix retiled. Evaluating more 32-token windows is cheaper but is one full model run
(~1 h on spike, plus ~36 min of goldens) per window.

#### 13.7c The baseline, settled: 7.88 at seqlen 2048 (MEASURED 2026-09-22)

`app/eval_ppl.py` (new) computes the unquantized baseline under the standard protocol, loading the
model exactly as the rest of the repo does. **TinyLlama-1.1B-Chat, bf16, wikitext2, seqlen 2048,
16 windows: ppl = 7.8821**, against the commonly cited ~7.7; the gap is sampling (32768 of 341469
tokens, 9.6%).

So the model is fine and the earlier confusion was entirely the evaluation window:

| quantity | ppl |
|---|---|
| **bf16, seqlen 2048, 16 windows -- THE BENCHMARK NUMBER** | **7.8821** |
| bf16, the capture's single 32-token window | 28.509 |
| **MXFP8, that same 32-token window** | **32.863** |

**A sampling error of mine, recorded so the shape of it is not repeated.** 13.7b reported 9.298 for
seqlen 2048 from FOUR windows and treated it as near-converged. It is not: windows 1-3 are the
hardest in the set (9.70, 12.44, 11.15) and stopping there lands 18% high. The running mean is
7.396 by window 8 and 7.882 by window 16. `eval_ppl.py` therefore prints the running perplexity per
window, so convergence is visible rather than assumed.

**Still unmeasured: whether the 1.153 MX ratio holds at long context.** It is measured at seqlen 32,
where the model is far from its operating point. Getting the MX number at 2048 needs the kernel to
run there, which is the retiling work 13.7b describes -- `Xn` alone would need the whole scratchpad
at M = 128.

**Environment note.** The repo's `.venv` has a **CPU-only torch** (`2.14.0+cpu`), so it cannot use a
GPU on any machine. Do NOT swap that build in to get one: the bit-exact mesh goldens come from a
torch-based model and the ladder's claims rest on those artifacts. `eval_ppl.py` falls back from
MXQuant's loader to plain `transformers` on ImportError precisely so it can run in a separate
minimal GPU venv (torch+cu12x, transformers, datasets, sentencepiece, accelerate) with no qtorch
build. The two load paths were verified to give the identical perplexity (28.5088 both ways).

### 13.9 MEASURED: the 4-point drop is HALF format, HALF datapath (2026-09-22)

The kernel lands at ppl 32.866 against the bf16 model's 28.509 -- ~15%, far worse than published
MXFP8 inference results, which are typically near-lossless. `app/ablate_mx.py` runs the SAME 22-layer
chain, the same captures and the same fp32 host glue under three arithmetic models and converts
8.2's per-tensor split into perplexity, which is the currency that decides whether the FORMAT or the
HARDWARE needs changing.

| model | nll | ppl | d(ppl) vs bf16 | logits rel_fro |
|---|---|---|---|---|
| torch bf16 (reference) | 3.3502 | 28.5089 | -- | -- |
| **fp32** -- no quantization, the harness sanity check | 3.3536 | 28.6064 | +0.10 | 0.98% |
| **mx-exact** -- MXFP8 operands, EXACT fp32 accumulate | 3.4159 | 30.4450 | +1.94 | 6.89% |
| **mx-rtl** -- the kernel (measured on spike, not recomputed) | 3.4924 | 32.8657 | +4.36 | 13.07% |

```
SPLIT  format   (MXFP8 operands, exact accumulate) : +1.84 ppl
       datapath (prod e4m3 trunc, acc e4m4..bf16)  : +2.42 ppl
```

**THE DATAPATH COSTS MORE THAN THE FORMAT.** That is the result. The format behaves about as the
literature predicts; what makes this look bad against published MXFP8 numbers is that **those assume
fp32 accumulation and this hardware does not**:

```
PROD_PRECISION = [(4, 3)] * 16                                       every product -> e4m3
ACC_PRECISION  = [(4, 4)]*8 + [(4, 5)]*2 + [(4, 6)]*5 + [(8, 7)]*1   (exp, frac) per lane
```

Eight of sixteen accumulator lanes carry 4 exponent and 4 mantissa bits; exactly one is bf16.

**Nothing is broken.** The `fp32` row validates the harness end to end (0.98% logit error vs torch,
which is just bf16-forward vs fp32-recompute), and the kernel is bit-exact against its model at
~2000 stages. The 4 points are design cost, not a defect.

**It cross-validates 8.2 independently.** That section measured format 6.57% / full datapath 11.59%
as a per-tensor error on ONE sub-layer, by a different route (MXQuant's `MXLinearSim`). This gets
6.89% / 13.07% on the logits of the whole model. Two independent methods, same story.

#### Where the leverage is

1. **The product quantizer TRUNCATES instead of rounding.** 8.2 measured that one change as worth
   ~2.9 points of tensor error (8.46% -> 11.31%). Round-to-nearest costs essentially no area and is
   the cheapest win available.
2. **The accumulator schedule** is the structural cost: widening the early lanes trades area for
   accuracy directly, and this table is what to sweep.
3. **`lm_head` is quantized too**, and it produces logits directly -- commonly kept in higher
   precision. UNTESTED; a plausible share of the +1.84, and cheap to check with `ablate_mx.py`.

#### Caveats

* All of this is the **32-token window**, where the baseline is 28.51 rather than the benchmark 7.88
  (13.7b/c). The SPLIT is a like-for-like comparison so it is sound, but whether the same
  proportions hold at seqlen 2048 is unmeasured.
* `mx-exact` quantizes with the same MXQuant encoder the kernel uses and then accumulates in fp32
  numpy; it is the format's cost with a perfect datapath, not any particular competitor's silicon.

### 13.10 The datapath model's scalar primitives, vectorized (2026-09-22)

**Why.** `rtl_exact` makes MXQuant's `MXLinearSim` bit-identical to the hardware, and `install()` is
device-aware, so an RTL-exact perplexity sweep on a GPU *should* have been available. It was not, and
the reason was two functions in `fp8_matmul_model.py`:

```python
flat = x.detach().cpu().reshape(-1).tolist()                       # fp_quantize_rne, exp_bits < 8
out_flat = [fp_quantize_rne_scalar(float(v), exp_bits, man_bits) for v in flat]
```

`fp_quantize_rne` (exp<8) and `fp_add_exact` were **scalar Python loops over `.cpu().tolist()`** --
MEASURED at 2.02 and 0.74 Melem/s. The accumulator schedule is `[(4,4)]*8 + [(4,5)]*2 + [(4,6)]*5 +
[(8,7)]*1`, so **15 of 16 lanes hit them**, and they are the innermost operation of the datapath.
On a GPU they would have been *worse*: a device->host->Python->device round trip per call, ~2048
times per matmul. Projected cost of an RTL-exact sweep as-is:

| seqlen | per 2048-token window |
|---|---|
| 32 | 23 h |
| 128 | 93 h |
| 512 | 371 h |
| 2048 | **1483 h** (62 days) |

**What was done.** `_rne_vec` and `_add_exact_vec` in `fp8_matmul_model.py`, mirroring
`_round_dyadic_to_scalar` branch for branch on integer tensors. Both are device-agnostic, so they
run wherever the tensor lives.

**The gate: `test_vector_primitives.py`.** Bit-pattern equality against the scalar reference -- not a
tolerance -- over values chosen to hit every branch: normals, subnormals, the subnormal/normal
boundary, exact RNE ties, overflow to inf, signed zeros, NaN and inf, across eight (exp, man) pairs.
`MXG_SCALAR_PRIMITIVES=1` forces the reference path so the test can obtain it.

**Two real bugs it caught**, both of which would have silently corrupted every golden:

1. **Underflow is UNSIGNED.** `_round_dyadic_to_scalar` returns a literal `0.0` when the rounded
   subnormal significand is zero, discarding the sign -- so a negative value that underflows becomes
   `+0.0`. Applying the sign after the zero gave `-0.0`. Deliberately NOT the same as an input of
   `-0.0`, which the `x == 0.0: return x` branch passes through with its sign. 1673/40940 elements.
2. **Wide formats overflow int64.** The exact sum of two (e6,m7) values spans ~93 bits. Rather than
   truncate, `_add_exact_vec_is_exact()` computes the bound and falls back to the scalar path; the
   RTL's exp_bits=4 lanes need ~44 bits and are comfortably inside.

**A trap worth naming: `_rne_general` (line 90) is dead code AND divergent.** It looks like exactly
this vectorization and is never called. It disagrees with the scalar reference on **~17% of
elements** (3429/20056 at e4m4). Do not wire it in.

**THE THRESHOLD IS THE POINT.** The two consumers sit on opposite sides of it:

```
n=256  1.02x | n=1024  3.19x | n=4096  8.46x | n=16384+  11-14x     (single-threaded, e4m4)
```

`tiled_matmul_hwlike` -- the kernel goldens -- works per DIM-tile, so n = 16*16 = **256**, exactly
the neutral point, and switching it wholesale measured SLOWER end to end (`_run_mesh` 1.73s vs
1.55s). `MXLinearSim` works on the full `[M][N]` and is what this exists for. `_VEC_MIN_ELEMS = 1024`
dispatches on size; the golden path is unchanged (bit-identical output, 1.31s vs 1.62s) and the
perplexity path gets the speedup. Override with `MXG_VEC_MIN_ELEMS`.

An earlier version used int64 and float64 throughout and fell off a cache cliff at `[2048,2048]` --
173 Melem/s at 1M elements against 7.6 at 4M. The quantizer needs only int32 (significands are
< 2**24) and the frexp scaling is exact in float32; that is what the current code uses.

**Gates, all re-run after the change:**

* `test_vector_primitives.py` -- PASSED, bit-identical on all eight formats
* `rtl_exact/verify_rtl_exact.py` -- **PASS, 65536/65536 identical, max abs diff 0**: still
  bit-identical to the hardware, which is the claim that had to survive
* `_run_mesh` output unchanged bit for bit against the forced-scalar path

#### 13.10b What it actually bought, measured

Single-threaded, best of 3, (e4,m4), vectorized vs forced-scalar:

| tile | `fp_quantize_rne` | `fp_add_exact` |
|---|---|---|
| `[32,64]` (the rtl_exact fixture) | 4.4x | 3.8x |
| `[128,2048]` | 12.0x | 7.1x |
| `[512,2048]` | 13.1x | 8.2x |
| `[2048,2048]` | 12.1x | 8.6x |

The datapath does two quantizes and one add per k-step, so **~10x on CPU** at realistic sizes.
`verify_rtl_exact.py` end to end: wall time unchanged (72s vs 75s) but **CPU time 895s -> 134s**,
6.7x less compute -- its fixture is only `[32,64]`, the smallest row above.

**Revised projection, and a correction.** An earlier estimate in conversation put the speedup at
~300x by assuming the k-loop would become memory-bandwidth-bound. It does not on CPU; torch's
elementwise throughput is the limit. At the measured ~10x:

| seqlen | as-is | vectorized (CPU, 1 thread) |
|---|---|---|
| 32 | 23 h | ~2.3 h |
| 128 | 93 h | ~9 h |
| 512 | 371 h | ~37 h |
| 2048 | 1483 h | ~148 h |

So **CPU alone does not make a seqlen-2048 RTL-exact sweep practical.** What the change actually
unlocks is the GPU: the `.detach().cpu().tolist()` round trip is gone, both primitives are pure
tensor ops, and `rtl_datapath.install()` was already device-aware. These are elementwise kernels, so
a GPU should add a large further factor on top of the 10x -- **unmeasured here, because this box has
no GPU** (the repo `.venv` also ships `torch 2.14.0+cpu`; see 13.7c on why that build must not be
swapped).

**Next, in order:** run the primitives on a GPU box and measure; then an RTL-exact sweep at seqlen
128-512, which is where perplexity is already near the benchmark (16.5 / 10.9 against 7.88) and the
+1.84 / +2.42 format-vs-datapath split of 13.9 can be re-measured at realistic context.

### 13.11 `app/ppl_datapath.py` -- perplexity under the datapath, at real context, on a GPU

`ablate_mx.py` (13.9) answered format-vs-datapath at the captures' 32-token window in numpy on CPU.
This runs the same question at any seqlen on the live HF model, device-agnostic, through four
arithmetic models:

| mode | what |
|---|---|
| `bf16` | the unquantized model -- the baseline (7.88 at seqlen 2048, 13.7c) |
| `mx-exact` | MXFP8 operands, EXACT fp32 accumulation -- the format's cost with a perfect datapath |
| `mx-bf16acc` | ... plus the hardware's cross-tile step: each 32-group reduced exactly, then folded in via `bf16_accum_add`, the golden's own function |
| `rtl-exact` | the full datapath via `MXLinearSim` under `rtl_exact/`, the configuration `verify_rtl_exact.py` gates as BIT-IDENTICAL to hardware |

```bash
python3 -m app.ppl_datapath --mode bf16       --seqlen 2048 --windows 16
python3 -m app.ppl_datapath --mode mx-exact   --seqlen 2048 --windows 16
python3 -m app.ppl_datapath --mode mx-bf16acc --seqlen 2048 --windows 16
python3 -m app.ppl_datapath --mode rtl-exact  --seqlen 512  --windows 4
```

**MEASURED at seqlen 32, 1 window, 32 threads** (reproduces the baseline exactly, so the harness is
sound):

```
bf16        28.5088
mx-exact    30.0920   (+1.58)
mx-bf16acc  29.1229   (+0.61)   <- LOWER than mx-exact
```

**THE DECOMPOSITION IS NOT MONOTONIC.** These modes were built expecting format -> +cross-tile ->
+lanes to be additive. It is not: bf16 cross-tile accumulation *partially cancels* the format error.
`rtl_exact/README.md` already records the same shape of behaviour on its own three-way split --
"every single change lowers the error versus fp32, and two of the three make the divergence from
hardware worse". **Do not quote one of these deltas as "the cost of X" measured with the others
unmatched.** (One 32-token window is also a tiny sample; this needs the real runs.)

**Scope, which bounds the answer.** Only the LINEAR layers are quantized (q/k/v/o, gate/up/down, and
`lm_head` with `--lm-head`). Attention's per-head `Q@K^T` and `P@V` are not, though the kernel
quantizes them. Sized on the same tokens: this reports `mx-exact` 30.0920 against `ablate_mx.py`'s
30.4450, which does quantize them -- so they plus `lm_head` are worth **0.35 ppl**, and every number
here is a lower bound by about that much.

**Threads.** `--threads` defaults to 32. Torch otherwise takes HALF the visible cores -- 128 of 256
on this box -- which is antisocial on a shared machine and made several timings in 13.10
unreproducible. The single-threaded numbers in 13.10b were explicitly pinned and are the reliable
ones; the "cache cliff" figures there were not, and are noisier than stated.

**Cost.** `bf16`/`mx-exact`/`mx-bf16acc` are minutes on a GPU at seqlen 2048. `rtl-exact` is the
expensive one -- it walks K sequentially with elementwise kernels per step -- so start at seqlen 128
or 512 with few windows and measure before committing to 2048.

### 13.12 MEASURED ON GPU: MXFP8 is essentially LOSSLESS at real context (2026-09-22)

`app/ppl_datapath.py` on a CUDA box, TinyLlama-1.1B-Chat, wikitext2, **seqlen 2048, 16 windows**:

| mode | ppl | d vs bf16 | relative |
|---|---|---|---|
| `bf16` (unquantized) | **7.8835** | -- | -- |
| `mx-exact` (MXFP8 operands, exact accumulate) | **7.9492** | +0.0657 | **+0.83%** |
| `mx-bf16acc` (+ bf16 cross-tile accumulation) | **7.9517** | +0.0682 | +0.87% |

**THIS OVERTURNS THE HEADLINE OF 13.9.** That section reported MXFP8 costing +1.84 ppl and the
datapath +2.42, for ~15% total -- measured at the captures' **32-token window**. At real context the
format costs **0.83%**, which is the near-lossless result the MXFP8 literature reports. The large
number was an artifact of the evaluation window (13.7b/c), not a property of the format.

The lesson generalizes: **quantization error measured at 32 tokens does not transfer to 2048.** At
seqlen 32 the model is far from its operating point (baseline 28.51 against 7.88), activations are
differently conditioned, and the relative cost of quantizing them is several times larger. Any
accuracy claim from this repo must state its seqlen.

**bf16 cross-tile accumulation is nearly free: +0.0025 ppl.** It is also now slightly WORSE than
exact accumulation, which is the expected ordering -- so 13.11's non-monotonicity (where
`mx-bf16acc` came out *better* than `mx-exact`) was a one-window small-sample artifact and should
not be read as a real effect.

**Cross-check.** The GPU's bf16 baseline 7.8835 against this box's CPU run 7.8821 (13.7c) -- two
machines, two torch builds, 0.02% apart.

Cost on GPU: `bf16` 1 s, `mx-exact` 12 s, `mx-bf16acc` 19 s for all 16 windows.

#### What is still unmeasured, and it is now THE question

`rtl-exact` -- the narrow per-lane accumulator (`[(4,4)]*8 + [(4,5)]*2 + [(4,6)]*5 + [(8,7)]*1`) and
the TRUNCATED e4m3 product -- has not been run at 2048. Everything above says the format and the
cross-tile step together cost under 1%; whether the LANES cost a little or a lot at real context is
exactly what the hardware team needs, and no number here predicts it. At seqlen 32 the lanes plus
product accounted for +2.42 of the +4.36, but 13.12 has just shown that proportions do not transfer.

```bash
python3 -m app.ppl_datapath --mode rtl-exact --seqlen 512  --windows 2   # time it first
python3 -m app.ppl_datapath --mode rtl-exact --seqlen 2048 --windows 16  # then commit
```

### 13.10c CORRECTION: the optimization was INERT until re-extraction, and hid three bugs

13.10 reported vectorizing `fp_quantize_rne` / `fp_add_exact` in
`gemmini-rocc-tests/fp8_matmul_model.py`. **The graded path does not use that file.** A GPU
`rtl-exact` run traced straight back into the scalar loop:

```
File ".../npu-exploration/app/mxmesh/fp8.py", line 114, in fp_quantize_rne
    out_flat = [fp_quantize_rne_scalar(float(v), exp_bits, man_bits) for v in flat]
```

`rtl_datapath._golden()` imports `app/mxarith.py` -> `app/mxmesh/fp8.py`, a copy **mechanically
extracted** from the upstream file by `tools/extract_model.py`, so the graded path has no runtime
dependency on the reference tree. `rtl_exact/README.md` still describes the older cross-tree import
("imported, not transcribed") and is out of date on this point. The drift guard is
`tests/selftest_extracted.py` instead, and it worked exactly as designed.

**It caught three real bugs that `test_vector_primitives.py` had missed**, because that test fed
only IN-FORMAT operands -- which is what `accumulate` supplies, but not what the primitives promise:

1. **bf16 fast path added before quantizing.** The scalar encodes each operand into the format
   first; adding raw values and rounding once is a different function. 685/4119 elements.
2. **Zero returns the other operand VERBATIM.** `if kind_x == "zero": return y` -- the RAW y, not
   its rounding. A value that merely underflows to zero does NOT take that path.
3. **The encode step does not saturate.** `_encode_exact_scalar_to_fields` has no overflow check and
   keeps an out-of-range exponent field, so `512 + -294.727` in (4,4) is **224**, not NaN. Using
   `fp_quantize_rne` (which overflows to inf) gave inf-inf = NaN on 150/4119 elements. Fixed with
   `overflow_to_inf=False`, plus a runtime in-range guard, because the int64 alignment bound assumed
   in-format operands and a 1e38 input into (4,4) needs a ~136-bit shift.

**The lesson for the gate.** An equivalence test that only probes the inputs the caller happens to
supply is not an equivalence test. `test_vector_primitives.py` passed all nine formats while three
divergences sat in the same functions; `selftest_extracted.py`'s edge-case probe found them in one
run.

**After re-extraction (`tools/extract_model.py ... app/mxmesh/fp8.py`), all gates pass:**

```
tests/selftest_extracted.py        11 checks, 0 failure(s)   (textual AND behavioural)
rtl_exact/verify_rtl_exact.py      PASS -- 65536/65536 identical, max abs diff 0
test_vector_primitives.py          PASSED on 9 formats
```

`verify_rtl_exact.py` now runs in **28.9 s against 75-88 s** before -- the first measurement where
the graded path actually got the speedup.

**A constraint on this region, worth knowing before editing it.** It is extracted verbatim into
`app/mxmesh/fp8.py` under a fixed preamble, so it **cannot reference module imports that the
preamble lacks** -- an `import os` for an env-var knob broke the extracted copy with a `NameError`
at import. `_FORCE_SCALAR_PRIMITIVES` and `_VEC_MIN_ELEMS` are plain module attributes for that
reason. **Re-extract after any change here, or the graded path silently keeps the old code.**

### 13.10d The window loop batched, and why the per-k form could never work on a GPU

After 13.10c the graded path really was vectorized -- and `rtl-exact` at seqlen 512 was still
hours, because the vectorization was never the binding constraint. `_simulate_atw_rtl` materializes
a full `[M][N]` outer product **per k** and walks k sequentially:

```
seqlen 512: 17920 k-steps/layer x 22 layers = ~394k steps, each ~80 small CUDA kernels on a
1M-element tile -> launch-overhead bound, O(M*N*K) MEMORY traffic where a matmul is O(M*N + M*K + K*N)
```

**The fix is structural: the WINDOWS are independent.** `S_red` is zeroed per 16-wide window and
only folded into `C` at the end, so nothing couples one window's reduction to another's. What is
sequential is the 16 LANES inside a window, whose accumulator precision varies. Batching `W` windows
turns K sequential steps into `window` of them on tensors W times larger -- exactly what a GPU
needs. `RTL_BATCH_BYTES` (default 2 GiB) sizes W; lower it if a run OOMs, it changes only the batch
size, never the result.

**The fold into C stays sequential and in order.** `cross_tile_accumulate` is bf16 and bf16 addition
is not associative, so reordering it would change the answer. A ragged K (not a whole number of
windows) falls back to `_simulate_atw_rtl_serial`, the original per-k form.

**Gated: `verify_rtl_exact.py` still PASSES -- 65536/65536 identical, max abs diff 0.**

**An optimization deliberately NOT taken.** `accumulate` calls `fp_quantize_rne` on both operands
before `fp_add_exact`, which then encodes them again -- seemingly a redundant double encode worth
2x. It is not redundant: products are e4m3 (max 448) while the accumulator lanes are e4m4
(max ~248), so that outer call implements the hardware's SATURATION. Removing it would silently
change the arithmetic.

**Progress reporting.** `ppl_datapath` now prints each projection as it completes
(`[rtl] 5/154 gate_proj (1,32,2048) 10.98s`), because a multi-hour run with no output is
indistinguishable from a hang -- which is exactly how the first GPU attempt looked. The per-
projection times also show where the cost is: `gate`/`up`/`down` (F = 5632) dominate at ~10 s each
on CPU against 0.4 s for `k_proj`/`v_proj`.

### 13.13 MEASURED ON GPU: the FULL hardware datapath costs ~1.9% at real context (2026-09-22)

`rtl-exact` -- MXFP8 operands, e4m3 TRUNCATED products, the per-lane accumulator schedule and bf16
cross-tile, the configuration `verify_rtl_exact.py` gates as bit-identical to hardware -- run on GPU
at **seqlen 2048**. Window 0, against the other three modes on the SAME window:

| mode | window 0 ppl | vs bf16 |
|---|---|---|
| `bf16` | 5.5547 | -- |
| `mx-exact` (format only) | 5.5626 | +0.14% |
| `mx-bf16acc` (+ bf16 cross-tile) | 5.6053 | +0.91% |
| **`rtl-exact` (the hardware)** | **5.6602** | **+1.90%** |

**The whole datapath costs about 2% of perplexity at real context.** Against 15% measured at the
captures' 32-token window (13.9), and it decomposes roughly into thirds: format ~0.1%, bf16
cross-tile ~0.8%, the narrow per-lane accumulator and truncated product ~1.0%.

**Caveat: ONE window.** The full 16-window run is ~29 h (below). The other three modes each tracked
their 16-window result closely on this window (bf16 5.5547 -> 7.8835 overall, and the mx modes
likewise), so ~8.03 total is a fair expectation -- but it is an expectation, not a measurement.

**This is the number the hardware team wanted**, and it says the aggressive accumulator schedule
(`[(4,4)]*8 + [(4,5)]*2 + [(4,6)]*5 + [(8,7)]*1`) costs ~1% perplexity, not the several points the
seqlen-32 measurement implied. 13.9's "+2.42 ppl from the datapath" was a small-window artifact in
the same way 13.12 showed the format's was.

#### Why it takes 1.8 h per window, and what is left to try

MEASURED: a flat **0.31 G element-steps/s** regardless of shape --

```
q_proj     8.6 G ->  27.9 s      gate_proj  23.6 G -> 76.3 s
k/v_proj   1.1 G ->   3.3 s      down_proj  23.6 G -> 76.4 s
```

-- so it is not shape-bound. `_rne_vec` and `_add_exact_vec` each issue 20-40 separate CUDA kernels,
every one reading and writing the whole tensor; a 50 M-element batched lane-step moves ~32 GB.

`RTL_COMPILE` (or `MXG_RTL_COMPILE=1`) wraps the three behaviours in `torch.compile` so inductor can
fuse those chains. OFF by default so the default path stays the one `verify_rtl_exact.py` has always
gated. If fusion does not close the gap, the remaining option is a Triton kernel with one thread per
output element walking k in registers -- the work is trivially parallel over (m, n) and only
sequential over k, so that is the shape of the real fix.

**`RTL_COMPILE` is wired but UNVERIFIED.** Compiling these functions on CPU did not finish in 40
minutes here, so it has never been gated. **Before trusting any number produced with it on, run
`MXG_RTL_COMPILE=1 python3 rtl_exact/verify_rtl_exact.py` on the target machine and require
65536/65536.** The default path (compile off) is unchanged and still passes.
