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

### 1a. Neither existing seam is bounded-safe; the default one is measured-safe only

Worst case for the inner column pass is `16 * |A|max * |B|max <= 2^8 = 256`:

| seam | A peak | B peak | `16*A*B` | verdict |
|---|---|---|---|---|
| `rescale` | 448 / 2^6 = 7 | < 2 (`TARGET_CODE_EXP = 0`) | 224 | within bound |
| `weight` (**default**) | 448 (raw requant output) | < 2^-3 (`WEIGHT_SEAM_TARGET_EXP = -4`) | 896 | **3.5x over** |

`weight` passes on sign cancellation, i.e. it is data-dependent. That is the same property
`app/mxquant.py:117` invokes to REJECT `TARGET_CODE_EXP = 2` ("it is data-dependent, which is not a
property to build on"). Applied consistently, it disqualifies `weight` as the default. Not changed
here — flagged for a decision. Fixing §1 makes the question moot.

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

---

## What "no seam at all" would look like

With §1, §2 and §3 fixed, a chained commit would: emit codes at a software-chosen target exponent,
write them to the scratchpad (§4, already done in RTL), and write their scales straight into the
A-side scale memory in the layout the mesh reads. The driver between two stages would then be a
`config_ex` and the next `LOOP_WS` — no data movement, no transpose, no LUT, no `--seam` flag, and
none of §1a's bounded-safety question.
