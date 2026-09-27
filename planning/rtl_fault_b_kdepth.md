# Fault B: the B-side scale window overflows at 2048 bytes (not "K=2048")

**Created** 2026-09-21. Companion to `rtl_mx_faults_handoff.md`, which covers both MX RTL faults;
Fault A (the requantizer's 2nd-block E8M0 scale) is **FIXED** as of `mxl7` passing. This file is
Fault B only.

> **RESOLVED 2026-09-21. `mxl4` passes.**
>
> **The fix:** `ScaleFactorMem.scala`'s `depth_sram` was hardcoded `64`, **ignoring the module's own
> `depth` parameter** — which is already 128 for DIM=16 (from the 16 KB `scale_mem` config) and 64
> for DIM=32. So the SRAM depth was there all along and the module just did not use it. Now
> `depth_sram = depth` (and `totalSizeBytes` with it). **No software change was needed**: the
> funct-27 loader writes linear byte offsets from 0, which stays consistent under a deeper SRAM.
>
> **New ceiling: 256 rows per window**, i.e. `N*K/512 <= 256` → **`N*K <= 131072`**. So `K=2048 @
> N=64` (256 rows) and `K=1024 @ N=128` (256) now fit; **`K=2048 @ N=128` needs 512 rows and still
> wraps, silently** — see §4.1.
>
> The analysis below is kept because the *mechanism* is what makes the symptom legible, and because
> §4.1 is still open. Note where it was wrong: it treated 2048 bytes as the hardware's real capacity
> and listed "deepen the scale SRAM" as the expensive honest fix. The capacity was never the
> constraint; a dropped parameter was.

**Status: FIXED. The mechanism below is confirmed; §4.1 (the silent wrap at the new ceiling) is
still open.**

## 1. The name "K=2048" is wrong, and that matters

The bisection found `mxl3` (K=1024) passing and `mxl4` (K=2048) failing, so the working label was
"a single `loop_ws` breaks past K=1024". Four quantities move together across that boundary and the
label picked the wrong one. The governing quantity is

```
    B-side scale rows required  =  J * G        where J = N/16 (output col tiles)
                                                      G = K/32 (E8M0 blocks along K)
                                =  (N/16) * (K/32)  =  N*K/512
                                =  the B-side scale window in BYTES / 16
```

and the hardware holds **128** such rows. `mxl4` needs 256. K is only implicated because N was held
at 64 throughout the first tier of the ladder.

This is why the "crosses 32 E8M0 groups exactly" observation in `llama_layer_hw_plan.md` §12.7 was a
red herring: at J=4, 32 groups is 128 rows, which is the capacity. The 32 was the capacity divided
by J, not a group-index width.

## 2. The mechanism, from `src/main/scala/gemmini/ScaleFactorMem.scala`

```scala
depth_sram   = 64                      // rows per bank
sramWidth    = 128                     // -> bytesPerBank = 16
numBanks     = 8                       // banks 0..3 WEIGHT scales, 4..7 ACTIVATION scales

// double buffering, from read_fire_banks: banks 0,1 = buffer A; banks 2,3 = buffer B
//                                        banks 4,5 = buffer A; banks 6,7 = buffer B

val row_addr_width = log2Ceil(2*depth_sram)          // = 7
val read_row_addr_w = Wire(UInt(row_addr_width.W))   // 7 bits -> 0..127
read_row_addr_w := (io.scaleMemCntl.loop_bound_j) * (counter_k_runtime >> kScaleShift.U)
                   + (counter_j_runtime)
val read_bank_idx_w      = read_row_addr_w(row_addr_width - 1)      // bit 6: 1 of 2 banks
val read_row_addr_w_real = read_row_addr_w(row_addr_width - 2, 0)   // bits 5:0: 64 rows
```

So per double-buffer half the READ path reaches **2 banks x 64 rows x 16 B = 2048 bytes**, addressed
by a 7-bit row index. `kScaleShift` is right: `tilesPerMxBlock = 32/(meshRows*tileRows)` = 2 at
DIM=16, so `counter_k >> 1` is the MX block index along K.

**Two limits coincide, because the address width was sized to the capacity:**

1. **Address truncation.** `read_row_addr_w` is 7 bits, so a required index of 255 becomes 127.
2. **Real capacity.** Only banks 0,1 (or 2,3) are read for a given `double_buffer_w_sel`, so 2048
   bytes is genuinely all that is readable per matmul.

**Why the output looks the way it does.** The WRITE path is wider than the read path:
`bank_idx_w_fp8 = write_addr_w(11,10)` is 2 bits, so a 4096-byte scale load happily fills banks
0,1,2,3. Reads with `double_buffer_w_sel = 0` then touch only banks 0,1, and the truncated address
wraps to row 0. **The second half of the reduction is therefore scaled by the first half's scale
bytes** — every output is a sum of correctly-scaled and wrongly-scaled partial products. That is
exactly the observed signature: right sign, right order of magnitude, 5-60% wrong
(`hw=0x4181`=16.125 vs golden `0x4172`=15.125), rather than the garbage a broken address walk gives.

It also explains why the failure is **bit-identical across two independently elaborated builds**
(Sep-15 and Sep-20, same 2036/2048, same values, same 87826 cycles): a deterministic addressing
truncation, not a race.

### The A side is fine, and only just

`read_row_addr_act := loop_bound_i * (counter_k >> kScaleShift) + counter_i_runtime`, needing
`I*G = (M/16)*(K/32)` rows. At `mxl4`'s M=32, K=2048 that is `2*64 = 128` rows — **exactly** the
capacity, so the A side does not overflow. With M=64 and K=2048 it would need 256 and would fail
the same way. Any fix must cover both windows.

## 3. The predicted pattern, and why it is a real test

`J*G` is the governing quantity, so **the group count alone and K alone both mispredict**. The
ladder already contains the two rungs that separate them:

| rung | M, K, N | J=N/16 | G=K/32 | rows needed `J*G` | B-scale bytes | fits 128? | predicted |
|---|---|---|---|---|---|---|---|
| `mxl0` | 64, 64, 64 | 4 | 2 | 8 | 128 B | yes | PASS — *observed* |
| `mxl3` | 32, 1024, 64 | 4 | 32 | **128** | 2048 B | **exactly** | PASS — *observed* |
| `mxl4` | 32, 2048, 64 | 4 | 64 | **256** | 4096 B | no | FAIL — *observed* |
| `mxl5` | 32, 2048, 64, 2 K-tiles | 4 | 32/call | 128 | 2048 B | exactly | PASS — *observed* |
| `mxl10` | 32, 1088, 64 | 4 | 34 | 136 | 2176 B | no | **FAIL** |
| `mxl11` | 32, 1536, 64 | 4 | 48 | 192 | 3072 B | no | **FAIL** |
| `mxl12` | 32, 2048, **32** | **2** | **64** | **128** | 2048 B | exactly | **PASS** |
| `mxl13` | 32, 1024, **128** | **8** | **32** | **256** | 4096 B | no | **FAIL** |
| `mxl14` | **64**, 1024, 64 | 4 | 32 | 128 | 2048 B | exactly | **PASS** |

**`mxl12` and `mxl13` are the decisive pair.** `mxl12` has **64 groups** — the same K=2048 depth as
the failing `mxl4` — and is predicted to PASS. `mxl13` has **32 groups**, the same as the passing
`mxl3`, and is predicted to FAIL. No hypothesis about group count, group-index width, or reduction
depth predicts that inversion; only the `J*G` product does.

`mxl10` is the sharpest quantitative check: at 136 rows it overflows by **8 rows out of 136**, so
only the last 2 of 34 groups are mis-scaled. It should fail by a *hair* — a small number of wrong
elements, or wrong in a small way — not the 2036/2048 that `mxl4` shows. If `mxl10` fails as badly
as `mxl4`, the capacity story is incomplete.

## 4.1 STILL OPEN: the new ceiling wraps just as silently

`depth_sram = depth` moves the limit from 128 rows to 256. It does not change the *failure mode*
past the limit: `read_row_addr_w` is `log2Ceil(2*depth_sram)` bits wide, so it still truncates
rather than signalling, and the upper K-groups still land on lower rows. `K=2048 @ N=128` needs 512
rows and will produce exactly the same right-sign, right-magnitude, quietly-wrong output that cost
this investigation two days.

**This is the same bug one parameter away, and the ladder cannot catch what nobody runs.** Two cheap
guards, either or both:

* **RTL:** assert/flag when `loop_bound_j * k_groups` (or the act-side equivalent) exceeds the
  readable rows. There is no silent-corruption-free reason to wrap.
* **Software:** a compile-time check in the kernels on `N*K <= 131072`. Note the current
  `LLAMA_KTILE_MAX` cap is expressed **in K**, which is only correct while `N <= 64` — see §4 item 2.

A ladder rung at `K=2048, N=128` (512 rows) would pin the new ceiling the way `mxl13` pins the old
one. It does not exist yet; `gen_mx_ladder.py`'s wide master already supports `N=128`.

## 4. Fixes, in increasing order of cost

1. **Report the limit instead of wrapping.** Cheapest and independently worth doing: nothing today
   tells software it has exceeded the scale window, and the failure is silent and plausible. A
   compile-time `require` cannot see runtime `J`/`K`, but the controller can raise an error or
   assert when `loop_bound_j * (k_blocks)` exceeds the readable rows. **A silent wrap is the actual
   defect here**; the capacity limit is a design parameter.
2. **Software: cap the per-`loop_ws` B-scale window** — already applied as
   `LLAMA_KTILE_MAX = 1024` in `llama_mlp.c` / `llama_attention.c`, and proven exact by `mxl5`. Note
   the cap is expressed in K, which is *only* correct while N ≤ 64. **The correct invariant is
   `N*K/512 <= 128`, i.e. `N*K <= 65536`.** A kernel that raises N must lower K to match; the
   current macro would not catch that. Worth generalizing even if the RTL is fixed, as a guard.
3. **Read both double-buffer halves** — 4 banks instead of 2 gives 4096 bytes and an 8-bit row
   index, at the cost of the double buffering the read path currently relies on.
4. **Deepen the scale SRAM** (`depth_sram`) and widen `row_addr_width` to match. The honest fix if
   deep-K single-call matmuls are wanted, and the one that also covers the A side at M=64, K=2048.

Whatever is chosen, **`mxl4` is the gate** and `mxl12`/`mxl13`/`mxl14` should keep passing.

## 5. What is already ruled out

From `rtl_mx_faults_handoff.md` §"Already ruled out", the ones that bear on this fault specifically:

* **non-square tile grids** — `mxl1` passes at `I=2, J=4`;
* **deep reductions as such** — `mxl2` (TK=16) and `mxl3` (TK=64) pass;
* **`ex_accumulate`** — honoured and bit-exact (`mxl5`), which closes §10.3's open question;
* **output-region reuse / smem clearing** — `mxl6` passes;
* **strided A mvin** — `mxl8` passes;
* **host fp32 glue** — `mxl9` passes on RTL;
* **`mxl4`'s spad map overrunning the scratchpad** — 12544 of 16384 rows, and the FPGA geometry was
  confirmed by the user. This was considered and dismissed; it would have been an alternative
  explanation for `mxl4` with the same "half the operands are wrong" character.

## 6. Reproducing

Commands, simulator caveats and the `gemmini_params.h` trap are all in
`rtl_mx_faults_handoff.md` §"Reproducing". In short:

```bash
cd /bwrcq/scratch/nicorakela/radiance-cy-dev/sims/vcs
E=/bwrcq/scratch/nicorakela/radiance-cy-dev/generators/gemmini/npu-exploration/out/baremetal/mx_rocket
D=/bwrcq/scratch/nicorakela/radiance-cy-dev/generators/testchipip/src/main/resources/dramsim2_ini
for t in mxl4 mxl10 mxl11 mxl12 mxl13 mxl14; do
  ./simv-chipyard.harness-MxGemminiRocketConfig-debug \
    +permissive +dramsim +dramsim_ini_dir=$D +max-cycles=10000000 \
    +vcs+initreg+1 +ntb_random_seed_automatic +loadmem=$E/$t \
    +permissive-off $E/$t </dev/null 2>/dev/null | grep -E '^(plan|mesh|mxl)'
done
```

Each rung prints `B-scales <n> B` in its `plan` line — that number over 2048 is the prediction.

### Confirming the mechanism on a waveform

If the pass/fail pattern holds, the wrap is directly observable and worth capturing once, because it
turns an arithmetic argument into a measurement. All the signals are `dontTouch`ed, so they survive
elaboration. Use the `waveform-signals` skill / NPI reader (see `generators/gemmini/CLAUDE.md`) and
watch, inside `ScalingFactorMem`:

* `read_row_addr_w` — should climb to 127 then **wrap to 0..3** rather than reaching 255;
* `read_row_addr_w_real` and `read_bank_idx_w` — the truncated halves;
* `counter_k_runtime` — should reach 127, confirming the k counter itself is fine.

Compare `mxl3` (never wraps) against `mxl4` (wraps at `counter_k_runtime = 64`). Per
`CLAUDE.md`'s pattern: scope the capture to this one module, and compare the ordered sequence of
`read_row_addr_w` values rather than absolute cycles.
