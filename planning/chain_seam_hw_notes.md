# The chain seam: what it costs, and what would remove it in hardware

**Created** 2026-09-02 · **Status** notes for a future RTL change; nothing here is implemented.

Context: fusing a matmul chain into ONE ELF (see `npu_exploration_bridge_plan.md`) means stage *i*'s
output has to become stage *i+1*'s A operand **on device**, with no host in the loop. The work the C
driver must do at that join is "the seam". This file records every part of it that exists only
because of a hardware or functional-model property, so it can be designed away later.

Every claim below cites the source it came from. Nothing is inferred.

Summary — the seam, in the order the driver performs it:

| # | Step | Why it exists | Cost today |
|---|---|---|---|
| 1 | drain codes `MX_READ_SMEM` -> DRAM, then `mvin` back | no smem->spad path in the functional model | M·N bytes out + back |
| 2 | transpose the scale bytes | requantizer WRITES `[M][N/32]`, mesh READS `[K/32][M]` | M·(N/32) byte moves |
| 3 | `MX_LOAD_SCALES` the transposed bytes back in | scales are produced on-device, exported, re-imported | M·(N/32) bytes |
| 4 | (`--seam rescale` only) LUT-shift every code, +6 every scale | requant output exponent is hardwired to fill the format | M·N table lookups |

Steps 2-4 are all consequences of ONE design choice (§1) plus TWO layout choices (§2, §3). Fix §1 and
step 4 disappears entirely; fix §2 and step 2 disappears; fix §3 and step 3 disappears. Step 1 is
already solved in RTL (§4) and is a functional-model gap only.

§6 and §7 are different from the rest of this file: not costs to design away, but **requantizer
defects**. §6 (a model-only bug) hides the very overflow §1 causes — read it before debugging any
chain that returns a scale code of 0. §7 is an accuracy bug in **both** oracles, found in radiance's
own software replacement for this block.

## Measured (2026-09-02, spike, 64x64x64 stages)

The emitter reports `METRIC seam_cycles_stage<i>` precisely so this cost is a number, not a guess:

| chain | seam | per-seam cycles | bytes the seam touches |
|---|---|---|---|
| `mlp2` / `mlp3` | `weight` | **454** | M x N/32 = 128 (scales only) |
| `mlp2` | `rescale` | **25 420** | + M x N = 4096 (every code) |

Two things to read off it:

* **`rescale` costs 56x what `weight` costs**, because the code shift touches 32x the data (one byte
  per element vs one per 32-element block). If §1a's bounded-safety argument wins and `rescale`
  becomes the default, that ratio is the price — and §1 removes it entirely.
* **454 cycles for 128 byte-moves is loop-bound, not bandwidth-bound**, and does not improve with
  source-level unrolling (gcc -O2 already unrolls the 2-trip block loop). It is host-core work, so it
  is equally real on spike and on RTL.

Do NOT compare these against the `cycles_stage<i>` numbers spike reports (~296 for a 64x64x64 MX
matmul). Those are host instructions retired, not mesh time — the RoCC matmul is a handful of
instructions to the host core. The seam-vs-matmul ratio is only meaningful against Verilator.

---

## 1. The requantizer's output exponent is hardwired to fill the element format

> **SUPERSEDED by §8 (2026-09-03).** The decision is made: the requantizer moves to MXQuant's e2e
> convention, which is `log2_pmax_floor = 0`. That is the fix this section asks for, and it removes
> steps 2-4 of the seam table and all of §1a along with it.

`MxRequantizer.scala:35`:

```scala
val log2_pmax_floor = MuxLookup(bits, 8.U)(Seq(
  FP4 -> 2.U, FP6 -> 4.U, FP8 -> 8.U, BF16 -> 16.U))
```

consumed at `:461`:

```scala
scale_exponent := max_biased_exp.zext.asSInt - 127.S - log2_pmax_floor.zext.asSInt
```

So an FP8 commit always normalizes each output block so its max lands at the **top** of e4m3 — codes
come back peaking at 448. That is standard OCP MX and correct in isolation.

But the mesh cannot accept codes that large as an operand. `ConfigsFP.scala:270-273`:

```scala
meshAccPrecisionList = Seq.fill(8) {MxFloat(4, 5, 4, true, false)} ++
  Seq.fill(2) {MxFloat(4, 6, 4, true, false)} ++
  Seq.fill(5) {MxFloat(4, 7, 4, true, false)} ++
  Seq.fill(1) {MxFloat(8, 8, 4, true, false)},
```

Exponent width **4** for 15 of the 16 rows, so the 16-deep column accumulator saturates near 2^8.
And the E8M0 block scales are applied only **after** that accumulation finishes — the accumulator
carries products of RAW CODES (`gemmini.cc:1163-1177` is the column pass; the scales do not appear
until `:1184`, and the multiply at `:1187` is in bf16). The accumulator's whole headroom is therefore
spent on code magnitude.

Two individually-correct designs that do not compose. Measured: feeding requant output straight into
a second GEMM gives 4096/4096 NaN (`app/mxquant.py:210`).

**The fix.** Make `log2_pmax_floor` a configured value rather than a format constant — a target
exponent field on the requant/mvout config (`gemmini_mxquant_config_mvout`, funct 26, has spare rs2
bits), so software can ask for output that peaks at 2^0 or 2^1 instead of 2^8. Then a chained commit
emits codes the mesh can consume directly, **the entire compensation problem disappears**, and with
it `--seam` as a concept. This is the single highest-value change in this file.

### 1a. RESOLVED 2026-09-02: both seams were over the bound; both fixed by one constant each

Worst case for the inner column pass is `16 * |A|max * |B|max <= 2^8 = 256`. As originally shipped,
**neither seam met it** — and the row this file previously recorded for `rescale` was wrong, because
it assumed a weight exponent the code did not use (`pipeline.py` used `+2`, not `0`):

| seam | A peak | B peak (was) | `16*A*B` | B peak (now) | `16*A*B` |
|---|---|---|---|---|---|
| `weight` (**default**) | 448 (raw requant output) | < 2^-3 (`-4`) | 896 — **3.5x over** | < 2^-5 (**`-6`**) | **224** |
| `rescale` | 448 / 2^6 = 7 | < 8 (`+2`) | 896 — **3.5x over** | < 2 (**`0`**) | **224** |

`weight` at -4 passed on sign cancellation, i.e. data-dependently — the same property
`app/mxquant.py:117` invokes to REJECT `TARGET_CODE_EXP = 2` ("it is data-dependent, which is not a
property to build on"). **It stopped holding at chain depth 4**: one column ceased cancelling, the
accumulator returned ±inf, and §6 below is what that turned into. Both constants are now derived
from the bound rather than measured (`grade/pipeline.py`).

Measured after the fix (mlp2..mlp8, all finite 4096/4096): the two seams are numerically a **wash** —
mlp8 gives 25.72% (`weight`) vs 26.29% (`rescale`) relative error vs fp32 — while `rescale` costs
**56x** more on device. So `weight` is the right default on both axes and the open decision is
closed. Note the prediction from operand precision alone (rescale keeps B at 2.336% error vs 3.145%,
since its reduction comes off the operand with headroom to spare) does NOT survive end to end.

Fixing §1 still makes the whole question moot, and would additionally restore the ~0.8% of weight
precision the `weight` seam's -6 gives up.

## 2. Scale write layout is the transpose of scale read layout

The requantizer writes one E8M0 byte per row per 32 output columns, row-major `[M][N/32]`:

* functional model, `gemmini.cc:1226` — `store<uint8_t>(scale_dram + m * N_blocks + bi, scale_code)`
* RTL, `MxRequantizer.scala:565` — `scale_mem_mvout_base_addr_act + (scale_write_addr_counter << ...)`

The A-side scale memory is read `[K/32][M]`:

* `gemmini.cc:1182` — `a_off = group * M_DIM + (i*DIM + r)`, i.e. `a_scales[group][row]`

Stage *i*'s N is stage *i+1*'s K, so the two are the same bytes in transposed order, and the driver
must physically transpose them. (The CODES need no such fixup — the drained buffer is already
contiguous `[M][N]` bytes, exactly the `[M][K]` layout `mvin` wants. Only the scales are wrong.)

**The fix.** Either a transpose-on-write bit in the requantizer's scale mvout path, or a stride/
transpose mode on `MX_LOAD_SCALES` (funct 27, whose rs2 currently carries only `len` and a 1-bit
`sel` — `gemmini.h:75`). Either removes step 2 outright.

Note the degenerate case: when `N == 32` the block count is 1 and `[M][1] == [1][M]`, so the
transpose is free. The emitter detects this and emits **no seam code at all** — the next stage reads
the requantizer's own buffer. Any wider intermediate needs the transpose.

Also checked and rejected: splitting the GEMM into 32-column strips so each strip's requant writes a
single block (making its scale write already `[1][M]`). Each strip's `LOOP_WS` has its own
`N_DIM = TJ*DIM`, so `smem_base + m*N_DIM` uses the strip width as the row stride and the CODES then
land strided instead of as a contiguous `[M][N]`. That trades a 128-byte scale fixup for an
M x N code fixup — strictly worse.

## 3. On-device scales make a pointless round trip through DRAM

The A-scales stage *i+1* needs were just computed on-device by the requantizer. They are exported to
DRAM (the address given to `gemmini_mxquant_config_mvout`), read back by the host, transposed, and
re-imported with `MX_LOAD_SCALES` into `mx_scale_a_mem`.

**The fix.** A path from the requantizer's scale output directly into `mx_scale_a_mem`, i.e. treat
"commit for chaining" as a distinct destination from "commit for export". Combined with §2 this makes
the scale half of the seam zero-cost.

## 4. Codes round-trip through DRAM — functional model only

`MX_READ_SMEM` (funct 28, `gemmini.h:80`) drains smem to DRAM; the driver then `mvin`s the same bytes
back into the scratchpad for the next stage. In the functional model there is no alternative — the
only smem exit is `mx_read_smem` (`gemmini.cc:1088`).

**Already fixed in RTL.** `ex_write_to_spad = true` (V1 in
`../planning/mxgemmini_rocket_standalone_plan.md`, sim-verified 2026-08-31) writes the
MxRequantizer's FP8 output straight to the internal scratchpad, reusable as an operand. So on the
Verilator path this step is already gone; only the spike path pays it. Worth adding an smem->spad
shortcut to `libgemmini` so the two oracles model the same dataflow.

## 5. `mx_smem` accumulates and is never cleared

`gemmini.cc:1190-1191`:

```cpp
float prev = bf16_to_f32(gemmini_state.mx_smem[idx]);
gemmini_state.mx_smem[idx] = f32_to_bf16_rne(bf16_accum_add(prev, scaled));
```

Nothing zeroes the region first. With one matmul per process this is invisible — `reset()`
(`gemmini.cc:63`) clears it at startup. In a fused multi-stage ELF, stage *i+1* would accumulate onto
stage *i*'s residue at the same `C_spad`.

**Worked around in the emitter**, not in hardware: each stage gets a disjoint `C_spad`. `mx_smem` is
`sp_matrices * DIM * 16 = 1024 * 16 * 16` u16 words (`gemmini.cc:63`, `gemmini.h:11`), so 64x64
stages 256 rows apart cost nothing against a 16384-row budget.

**Unverified:** whether the RTL has an overwrite-vs-accumulate control for this write. If it does,
this is a functional-model gap only, and the emitter's disjoint-`C_spad` trick can be dropped on the
Verilator path. Check before relying on it.

## 6. An accumulator overflow is laundered into a valid-looking scale code — functional model

> **PARTLY WRONG, see §8.4.** The claim below that "the RTL already does the right thing" is
> false. Masking non-finite values out of the block-max reduction means an `Inf` gets a scale from
> its finite siblings and then saturates to 448 — measured, one `+Inf` among `0.5`s comes back as
> **0.875**, a completely plausible value. The RTL is the *worst* of the three here, not the best.

Found 2026-09-02 while pushing chain depth to 4. **This one is a robustness bug, not a design
tradeoff**, and it cost real diagnosis time because it removes the evidence.

`gemmini.cc:1213-1222`, the FP8 requant scale path:

```cpp
uint8_t scale_code;
if (max_abs == 0.0f) {
  scale_code = 0;                                  // sentinel: an ALL-ZERO block
} else {
  int max_exp = (int)floorf(log2f(max_abs));       // :1217
  int s = (max_exp - 8) + 127;
  if (s < 0) s = 0;                                // :1220
  ...
```

When the mesh accumulator overflows, `max_abs` is `inf`. Then `log2f(inf)` is `inf`,
`(int)floorf(inf)` is `INT_MIN`, and the `s < 0` clamp maps it to **0** — the exact code the branch
above reserves for an all-zero block. Verified in isolation:

```
max_abs=inf -> max_exp=-2147483648  s=0  scale_code=0
```

The requantizer then packs `inf / 2^-127` as saturated 448-codes. So the output block reads back as
"scale 2^-127, codes at the top of e4m3" — a perfectly plausible tiny-value block — and every later
stage consumes it and produces more infs. Nothing anywhere reports an error.

Two things make it especially misleading: the sentinel is *ambiguous* (0 means both "all zero" and
"overflowed"), and the codes contradict the scale (an all-zero block cannot have 31/32 nonzero
codes), which is what eventually gave it away.

**The fix**, in the model: handle non-finite `max_abs` explicitly. E8M0 already has a NaN encoding —
code `0xFF` (`mx_fp_math.h:254`, and `e8m0_decode` maps it to NaN) — which is exactly the right
signal, propagates instead of hiding, and is free.

**CONFIRMED spike-only (2026-09-03).** The RTL already does the right thing: its block-max reduction
explicitly masks non-finite inputs out (`MxRequantizer.scala:440-442`):

```scala
val mag        = abs(e.asUInt)
val isNanOrInf = mag(14, 7).andR   // BF16: exp field all-ones → NaN or Inf
Mux(isNanOrInf, 0.U, mag)
```

so an infinite accumulator value cannot reach the exponent arithmetic at all. This is a
**functional-model-only defect**, and spike is the less careful of the two oracles here.

Same failure class as §14.1 of the bridge plan (non-finite input silently coded as zero): the
datapath's loudest failure turned into a silent wrong answer.

### 6a. Second spike/RTL divergence in the same lines: the all-zero-block sentinel

`MxRequantizer.scala:456-458` gives an all-zero (or subnormal-max) block scale code **127**, i.e.
scale 1.0. `gemmini.cc:1214-1215` gives it **0**, i.e. 2^-127. `app/mxquant.py:_shared_exponent`
independently chose 127, for the stated reason that it "stays exactly zero and introduces no
denormal scale" — so **spike is the outlier of the three**. Worth fixing in the model, and note it is
the same sentinel whose ambiguity made §6 hard to read: with 127 the overflow signature and the
all-zero signature would not have collided.

## 7. The scale exponent uses floor(log2), so ~19% of blocks clip their own max — RTL and spike

> **WRONG AS DIAGNOSED, see §8.** `floor(log2(amax)) - emax` is exactly what the OCP reference
> (`microxcaling/mx/mx_ops.py::_quantize_mx`) does — measured 100% scale-code agreement REF vs RTL
> vs spike over 12 000 blocks. `simt_quant`'s `ceil(log2(amax/448))` is the deviation, and adopting
> it would have been a regression against the reference. The clip rate on max-of-32 blocks is also
> ~56%, not 19% (that figure came from uniformly-drawn maxima). The real fix is §8: drop the `-
> emax` term entirely, which is a *third* convention and the one MXQuant's LLM evals actually use.

Found 2026-09-03 by reading `radiance-kernels/kernels/simt_quant` (idea reference only, §1.3), which
is a SIMT **software replacement for the tapeout-330 hardware requantizer**. Its exponent choice
differs from ours in one term, and the difference is a real accuracy bug on our side.

| | formula | resulting block-max code |
|---|---|---|
| ours (spike `gemmini.cc:1217-1218`; RTL `MxRequantizer.scala:461` via `max_biased_exp`) | `floor(log2(amax)) - log2_pmax_floor` | `[256, 512)` |
| simt_quant (`kernel.cpp:114`, `gen_quant.py:7`) | `ceil(log2(amax / 448))` | `[224, 448]` |

Both aim to put the block max at the top of e4m3. But ours derives the exponent from the **exponent
field alone** — `max_biased_exp = block_max_uint(14, 7)`, with no significand term — so a block whose
max significand exceeds 1.75 lands above 448 and **saturates**. Measured over 200k random block
maxima: the two agree on 80.7% of blocks, and on the remaining **19.3% ours silently clips the
largest element in the block**. simt_quant adds exactly the missing term
(`+ ((ab & 0x7FFFFF) > 0x600000 ? 1 : 0)`, where `0x600000` is the mantissa of 1.75) and never clips.

This is present in **both** oracles, RTL included, so it is a genuine hardware accuracy bug and not a
model gap. It costs up to one bit of code range on the other 80.7% of blocks, which is the right
trade: clipping the block max is the one error a block scale exists to prevent.

**It does NOT fix §1.** Verified: max |A| is 448 either way, so `16*|A|*|B|` is unchanged and the
accumulator bound is untouched. simt_quant implements the *same* normalize-to-format-max policy
§1 objects to — it replaces a broken requantizer, it does not redefine the target exponent.

---

## 8. DECIDED (2026-09-03): the requantizer targets MXQuant's e2e convention

The goal is to run realistic workloads whose quantization matches the **MXQuant repo**
(`MXQuant/`, in this directory). That repo contains *two different* block-scale conventions, and
the one its end-to-end LLM evaluations use is **not** the OCP one:

| | formula | block max lands in | `16*|A|max*|B|max` vs the 2^8 accumulator bound |
|---|---|---|---|
| `microxcaling/mx/mx_ops.py::_quantize_mx` (OCP Alg. 1) | `floor(log2 amax) - emax` | `[256, 512)` | 3.2e6 — **12 500x over** |
| `end_to_end_linear/mx_block_quant.py::_po2` (**the target**) | `floor(log2 amax)` | `[1, 2)` | 64 — **4x headroom** |
| RTL `MxRequantizer.scala:461` / spike `gemmini.cc:1243` (today) | `floor(log2 amax) - 8` | `[256, 512)` | 12 500x over |

The two differ by exactly `emax = 8` exponents. **Adopting the e2e convention fixes conformance
and the chaining overflow with one change**, because a max in `[1,2)` on both operands sits four
times under the accumulator bound. `--seam`, `WEIGHT_SEAM_TARGET_EXP`, `RESCALE_SEAM_TARGET_EXP`,
`CHAIN_EXP_SHIFT` and `_emit_seam`'s LUT shift all become dead code, and §1/§1a close.

Note `app/mxquant.py`'s `TARGET_CODE_EXP = 0` **already is** `_po2`. Its comment justifies it as a
workaround for the accumulator; it is in fact the conformant value.

### 8.1 The target spec, measured from `quantize_mx_block32` itself

| property | MXQuant e2e | RTL today | spike today |
|---|---|---|---|
| scale exponent | `floor(log2(max(amax, 2**-23)))`, **no `- emax`** | `max_biased_exp - 127 - 8` | `floor(log2 amax) - 8` |
| min scale | clamped at `2**-23` (fp32 eps) | unclamped | clamp `[0,254]` |
| all-zero block | scale `2**-23` -> code **104**; the `where(X==0,1,X)` guard on `mx_block_quant.py:137` is dead code | 127 | 0 |
| element rounding | **half away from zero** | RNE, via an E5M3 hop | half to even |
| saturate | +-448 | same | same |
| subnormals | allowed to `2**-9` | double-rounded; `exp<=5` truncates to 0 | correct |
| `-0.0` | -> `+0.0` | same | same |
| non-finite | `X = inf/nan`, elements -> NaN | `Inf` -> 448, block looks clean | -> 0 |
| subnormal-max block | flushed to 0 (eps clamp) | flushed (correct here) | recovered (wrong) |

Two of the notes above were backwards and are corrected in place: §7 (the exponent formula is
reference-conformant, `simt_quant` is the deviation) and §6 (the RTL launders non-finite input
worse than spike does). `microxcaling`'s `round="nearest"` is round-half-**away**-from-zero, not
RNE, so both oracles are non-conformant on ties — with 4 mantissa bits dropped, ties are ~1-in-16
elements, far more than the subnormal tail.

Measured on the real TinyLlama tiles under the target convention: **2.535%** of elements land in
the E4M3 subnormal tail, **0.000%** saturate, 0.211% are zero. So the accuracy risk of moving the
peak from 448 down to 2 (a shorter runway before the subnormal tail) does not materialise on real
data, and 2.5% is the exposure of the RTL's subnormal-path bug.

### 8.2 Step 1 — DONE 2026-09-03: the golden generator

`app/mxq_golden.py`. Wraps `quantize_mx_block32` (imported, never transcribed, so it cannot drift)
and converts its value output `(P, X)` into the hardware wire format: one E8M0 byte per block plus
one E4M3 byte per element. Both conversions are lossless by construction — `P` is already an exact
E4M3 value so it is encoded by an exact 256-entry **table lookup** rather than by rounding, and a
value outside the table raises instead of being silently rounded. `golden()` asserts
`decode(wire) == P * broadcast(X)` on every call.

Verified: every corner tile plus **5 046 272 real TinyLlama elements** (2464 tiles x both axes,
157 696 blocks) encode losslessly. `save()`/`load()` emit `.npz` vectors for the spike and RTL
benches. Layouts: `axis="row"` -> scales `[R][C/32]`, the requantizer's own write layout;
`axis="col"` -> `[R/32][C]`, the operand layout `stats_from_pairs.py` uses.

Run it with `python3 -m app.mxq_golden`. New venv deps: `packaging`, `qtorch`.

### 8.5 Step 4 — DONE 2026-09-03: the RTL test, on llama data

`software/gemmini-rocc-tests/gen_matmul_fp8_64x64_llama.py` regenerates
`include/matmul_fp8_64x64.h` with nothing synthetic in it:

* **operands** — four real logged TinyLlama 32x32 tiles (`layer0/mlp.gate_proj`, from MXQuant's
  own `data_evalrun_01`) stitched into 64x64, quantized by `quantize_mx_block32` via
  `app/mxq_golden.py`. Replaces `torch.randn` operands and `randint(-3,4)` power-of-two scales.
* **`C_out_bf16`** — those operands through `fp8_matmul_model.tiled_matmul_hwlike`, the bit-exact
  mesh model. Checked by `matmul_tiled_fp8_64x64.c`.
* **`C_out` / `C_scales_out`** — MXQuant's quantizer applied to that exact BF16 tile, in the
  requantizer's own `[M][N/32]` scale layout. Checked by `matmul_tiled_fp8_64x64_requant.c`.

That split isolates the requantizer: if `C_out_bf16` matches, then `C_out` matching means the
requantizer agrees with MXQuant with no accumulator modelling in the way.

The requant test now also **checks the block scales**, which it never did. Codes alone cannot
catch a wrong scale convention -- shifting every scale by a constant leaves the codes untouched,
which is exactly the `- emax` bug.

The old generator (`fp8_matmul_model.run` -> `matrix_mx_requantize`) hardcodes
`log2_pmax = emax = 8`, so it produces OCP-convention goldens and must not be used for this
header any more. Its `_po2` at `:834` is dead code.

**Verified on spike** (both PASS, codes and scales exact). ELFs for the RTL are built at
`build_mx_rocket/bareMetalC/matmul_tiled_fp8_64x64{,_requant}-baremetal`.

**TRAP, worth knowing:** `$RISCV/lib/libgemmini.so` is a *separate installed copy*. `make` in
`software/libgemmini` only updates the source-dir `.so`; `run_kernel.py` picks that up because
`compiler/targets/mx_gemmini_rocket/backend/runner.py:172` passes `--extlib=<source .so>`, but a
bare `spike --extension=gemmini` silently loads the STALE installed one. That is what made the
requant test fail with every scale off by exactly 8 on the first run. Either `make install` or
always pass `--extlib`.

### 8.3 Step 2 — DONE 2026-09-03: spike

- `gemmini.cc` requant post-pass — dropped the `- 8`; `max_abs` floored at `FLT_EPSILON`, which
  subsumes the all-zero sentinel (so code 0 no longer means both "all zero" and "overflowed")
- `gemmini.cc` — NaN/Inf tracked separately in the block-max loop (NaN loses every `>`
  comparison, so it was invisible); a non-finite block emits scale code `0xFF` and applies
  `nan`/`inf` as the divisor, which reproduces the reference's element codes byte-for-byte
- `mx_fp_math.h` — `round_half_to_even` -> `round_half_away`; non-finite -> `0x7F` instead of 0;
  underflow keeps its sign; `fp8_e4m3_decode` now reads `0x7F`/`0xFF` as NaN, not 480.0

Verified with a C++ harness that includes the real `mx_fp_math.h` and runs the block loop
verbatim: **0 code and 0 scale mismatches** against the golden on all 8 corner tiles and
2 523 136 real TinyLlama elements / 78 848 blocks. Single-stage `linear` output is bitwise
identical to before the change; 4/4 kernels still PASS.

### 8.4 Step 3 — DONE 2026-09-03: RTL

- `MxRequantizer.scala` — `log2_pmax_floor` -> 0; the scale block collapses to the block max's
  BF16 biased exponent floored at 104 (= `2^-23`), replacing the zero/subnormal sentinel; NaN and
  Inf flags OR-reduced out of the block-max reduction; `scale_e8m0 := 255` when non-finite, and
  the valid-code clamp lowered to 254 to reserve it
- `BF16ScalaRoundToTiny.scala` — new `BF16ToE4M3`: one rounding step, ties away from zero,
  subnormals to `2^-9`, saturating to 448. **`RoundAnyRawFNToRecFN` cannot target E4M3** — with
  `expWidth=4` hardfloat reserves exp field 15 for Inf/NaN, but E4M3 uses exp=15 mantissa 0..6 as
  normals up to 448, which is why the original went through E5M3 and double-rounded. So the §8.4
  plan of "point hardfloat at E4M3" does not work; direct logic does.
- `BF16ScalaRoundToTiny.scala` — `round_near_even` -> `round_near_maxMag` (FP6/FP4 paths);
  `input_exp === 0` now forces a signed zero; `mapToZero` in all three packers keeps its sign
- non-finite blocks are forced per element (NaN wins over Inf), because the datapath multiplies by
  a finite power of two and cannot get the reference's `/inf` -> 0 behaviour for free

Verified two ways: a line-for-line model of the edited Chisel gives **0 mismatches** against the
golden on the same corpus as spike, and the design **elaborates** (confirmed by the user).

---

## 9. DONE 2026-09-04: every `matmul_tiled_*` test runs on real TinyLlama data, all 27 PASS

§8.5 ported one header. This ports the whole family: **27/27 `matmul_tiled_*` tests pass on spike**,
every one of them against operands sliced from a real TinyLlama forward pass and quantized by
MXQuant. Spike was not modified.

### 9.1 The data: contiguous 512x512 pairs, not stitched 32x32 windows

`app/capture_llama_tiles.py` re-runs MXQuant's own capture (`log_pairs_from_eval.PairLogger`, same
hooks, same wikitext2 tokenization) at **`--N 512`**, giving 110 (A_square, W_square) pairs across
all 22 layers x 5 projections. Each test then slices one contiguous tile:

```
A = A_square[:M, :K]        real activations, tokens x in-features
B = W_square[:N, :K].T      real weights,     out x in-features -> [K][N]
```

The pair shares its in-feature offset `i0`, so `A @ B` is a genuine sub-block of that projection's
real output. This matters because `data_evalrun_01` was captured at `--N 32` with a **random**
offset per tile (`log_pairs_from_eval.py:183-195`): its 8 tiles per projection are independent
windows, so `gen_matmul_fp8_64x64_llama.py`'s 2x2 stitch produced a matrix whose every *value* was
real but whose *structure* was a collage — and it does not scale, since a 128x512 operand needs 64
tiles against the 9 a projection has. k_proj/v_proj are skipped: `out_features = 4*64 = 256 < 512`.

Deps added to the venv: `transformers`, `safetensors`, `datasets`, `accelerate`. One wrinkle —
`mxquant/datautils.py:10` asks for the bare id `wikitext`, which `datasets>=4` rejects; the capture
driver loads `Salesforce/wikitext` itself with identical tokenization rather than patching MXQuant.

### 9.2 The generators

| file | covers |
|---|---|
| `software/gemmini-rocc-tests/gen_matmul_llama.py` | all 7 FP8 + 3 FP4 headers |
| `software/gemmini-rocc-tests/llama_operands.py` | real operands + MX quant + k-means LUTs for FP6 |
| `lut_mapping_demo.py` (`MXGEMMINI_LLAMA=1`) | the 2 FP6 headers, via the module above |
| `software/gemmini-rocc-tests/verify_vs_pytorch.py` | the error ladder in §9.5 |

`gen_matmul_llama.py` supersedes `gen_matmul_fp8_64x64_llama.py` (which still works, on the old
stitched data). `app/mxq_golden.golden()` gained a **`pmax_shift`** argument for §9.4.

### 9.3 THE BUG THAT MATTERED: the tests were never rebuilt

`bareMetalC/Makefile:124` listed only `gemmini.h`, `gemmini_params.h`, `gemmini_testutils.h` as
prerequisites. The generated `matmul_*.h` were **not** tracked, so regenerating a golden left the
ELF untouched and the test re-ran the OLD data — **and passed**, because a stale header is
self-consistent with itself.

This is why five FP8 requant tests reported byte-identical mismatch counts (1022, 3068, 6142, 9212,
16374) before *and* after regeneration: they had never been rebuilt since §8.3 changed spike's FP8
requantizer. The moment the dependency was fixed they all passed with no other change. Fixed by
appending `$(wildcard $(abs_top_srcdir)/include/matmul_*.h)`.

Same failure class as the §8.5 `--extlib` trap, and worse: that one failed loudly, this one passed.

### 9.4 FP4/FP6 were never migrated to MXQuant's convention — in TWO ways

§8 moved the **FP8** requantizer to MXQuant's e2e convention. FP4 and FP6 kept the OCP behaviour,
and their goldens must reproduce it or they disagree with spike:

| | scale exponent | element rounding |
|---|---|---|
| FP8 | `floor(log2 amax)` (`gemmini.cc:1247`) | MXQuant `_quantize_elemwise`, ties away |
| FP6 | `floor(log2 amax) - 4` (`gemmini.cc:1366`) | `bf16_rne` -> `fp6` -> nearest LUT entry |
| FP4 | `floor(log2 amax) - 2` (`gemmini.cc:1478`) | `q_bf16_rne` then `hw_bf16_to_e2m1` (two-stage) |

The scale term is handled by `Format.out_pmax` + `golden(pmax_shift=)`. The **rounding** term was
found the hard way: with MXQuant's quantizer the FP4 requant test missed on 578/4096 elements, all
in the mantissa LSB (`got: c, exp: d`), because `mx_fp_math.h:402-404` requantizes via a two-stage
RNE->E3M1->E2M1 that breaks ties differently. `Format.out_requant="model"` routes FP4/FP6 output
requant to the hardware's own quantizer. **Operands** use MXQuant for every format — they are
inputs, so any valid quantization is legitimate.

**Measured — exactly where the hardware is and is not MXQuant.** Requantizing the same BF16 tile
with MXQuant vs with the hardware's quantizer:

| header | scale codes differing | element codes differing |
|---|---|---|
| all 7 FP8 | **0** | **0** — byte-for-byte identical |
| fp4 64x64 | 0 / 128 | 578 / 4096 (14.1%) |
| fp4 128x128 | 0 / 512 | 1802 / 16384 (11.0%) |
| fp4 128x128x512 | 0 / 512 | 1330 / 16384 (8.1%) |

So **FP8 is exactly MXQuant end to end**. FP4 agrees on every block scale (the `pmax_shift` term is
right) and disagrees only on element ties — and **every disagreement is exactly one step on the
E2M1 value ladder**, verified, never two. It is a tie-break difference, not a magnitude error. The
one change that would make FP4/FP6 exactly MXQuant too is giving their requantizers FP8's
single-step round-ties-away, i.e. finishing the §8 migration for the nibble formats.

### 9.5 FP6 is a LUT codebook, not MX element codes

The FP6 tests run `uselut=1`: operands are 4-bit indices into per-group 16-entry LUTs of FP6 codes,
one LUT per `2^G` rows of A / columns of B (`gemmini.cc:1321`, `lut_idx = (i*TM + m) >> G`; G=1).
MXQuant *does* have this — `prodacc_bundle/lut_quantization.py` on branch `chloe-branch-all`, a copy
of `microxcaling/mx/level2_scratch.py`: MX-quantize first so every value is a valid FP6 code, then
reduce that codebook to `num_signposts=16` per channel by k-means. `llama_operands.build_luts`
follows it, with deterministic quantile-init Lloyd instead of random init so headers reproduce.

### 9.6 Two test-side fixes (spike untouched)

* **`*_DRAMMvout` segfaulted spike itself.** `mx_loop_ws_spad` derives `smem_base = C_spad * DIM`
  (`gemmini.cc:1154`) and indexes `mx_smem` unchecked (`:1216`); those tests pass
  `acc_addr = 1<<31`, so `smem_base = 0x800000000`. **Spike models no accumulator destination for
  MX at all** — the internal scratchpad is its only MX output route. Guarded under `#ifdef
  SPIKE_SIM` to use the spad dest; the RTL path keeps `acc_addr`/`0xb8`.
* **`matmul_tiled_fp8` loaded no scales on spike.** Its scale-load was under `#ifndef SPIKE_SIM`, so
  the mesh ran against stale `mx_scale_{a,b}_mem` — the pre-existing 972 mismatches. Added the
  funct-27 path and pointed it at `matmul_fp8_32x32x32.h`; its old header declared `MATMUL_GK 16`
  for `K=32`, a 2-element scale group contradicting the block-32 convention. It now differs from
  `matmul_tiled_fp8_32x32x32` only in mvin layout, which incidentally settles that the two layouts
  (`B_in + j*DIM*M + k*DIM` vs `+ k*DIM*N + j*DIM`) agree against one `A @ B` golden.

The five FP8 requant tests that checked only codes now check `C_scales_out` too, as §8.5 argued they
must — a constant shift of every scale leaves the codes untouched.

### 9.7 Where spike sits relative to PyTorch

`verify_vs_pytorch.py`. Spike is **bit-exact** against both the mesh model and the requantizer (the
tests assert it, 0 mismatches, codes and scales). These are the gaps to plain fp32, on real data:

| header | operand quant | mesh arithmetic | end-to-end | output requant |
|---|---|---|---|---|
| fp8 32x32x32 | 3.70% | 4.91% | 5.69% | 2.67% |
| fp8 64x64 | 3.69% | 4.59% | 5.67% | 2.58% |
| fp8 96x32x32 | 3.67% | 3.83% | 4.74% | 2.63% |
| fp8 64x96x64 | 3.60% | 5.35% | 5.43% | 2.15% |
| fp8 96x96x64 | 3.17% | 3.47% | 4.77% | 2.71% |
| fp8 128x128 | 3.73% | 4.18% | 5.67% | 2.71% |
| fp8 128x128x256 | 3.63% | 4.45% | 5.27% | 2.69% |
| fp4 64x64 | 35.54% | 0.04% | 35.54% | 11.67% |
| fp4 128x128 | 37.77% | 0.00% | 37.77% | 11.78% |
| fp4 128x128x512 | 41.11% | 0.08% | 41.11% | 11.78% |

"operand quant" is MX itself (`A_hat @ B_hat` in fp32 vs `A @ B`); "mesh arithmetic" is the
hardware's truncating products and reduced-precision per-lane accumulators. FP8's two terms are
comparable; **FP4's mesh error collapses to ~0.05%** because E2M1's single mantissa bit makes the
operands far coarser than the mesh's own e4m3 arithmetic — the datapath is essentially exact
relative to its inputs, and all the loss is in the format.

### 9.8 Reproducing

```bash
cd generators/gemmini/npu-exploration && .venv/bin/python3 -m app.capture_llama_tiles   # once
cd ../software/gemmini-rocc-tests
export PATH=../../npu-exploration/.venv/bin:$PATH                 # ninja, for microxcaling
../../npu-exploration/.venv/bin/python3 gen_matmul_llama.py       # 7 fp8 + 3 fp4 headers
MXGEMMINI_LLAMA=1 ../../npu-exploration/.venv/bin/python3 lut_mapping_demo.py
MXGEMMINI_LLAMA=1 MXGEMMINI_M=128 MXGEMMINI_K=128 MXGEMMINI_N=128 \
  MXGEMMINI_LLAMA_LAYER=layer5 MXGEMMINI_HEADER_PATH=./include/matmul_data_mx_lut_hw.h \
  ../../npu-exploration/.venv/bin/python3 lut_mapping_demo.py
../../npu-exploration/.venv/bin/python3 verify_vs_pytorch.py
```

Then build + run each test per §6.1. **Always pass `--extlib`** (§8.5's trap).

---

## What "no seam at all" would look like

With §1, §2 and §3 fixed, a chained commit would: emit codes at a software-chosen target exponent,
write them to the scratchpad (§4, already done in RTL), and write their scales straight into the
A-side scale memory in the layout the mesh reads. The driver between two stages would then be a
`config_ex` and the next `LOOP_WS` — no data movement, no transpose, no LUT, no `--seam` flag, and
none of §1a's bounded-safety question.
