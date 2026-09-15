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

### 9.2 DEFERRED: the llama kernels belong in npu-exploration, not bareMetalC

**Decision (user, 2026-09-14): agreed in principle, deliberately not done yet.**

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

Not done now because it is a pure relocation touching two repos mid-task, and the kernels are
currently passing; moving them is worth doing as its own change with its own verification, not as a
side effect of adding the full-attention kernel.

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
