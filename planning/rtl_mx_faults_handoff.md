# Two MX RTL faults: findings and exact repro (handoff, 2026-09-21)

> **FIXES APPLIED 2026-09-21 (UNBUILT — verify on RTL):**
> - **Fault A** — `MxRequantizer.scala` scale coalescer AG. At `numOutBlocks==1` (DIM=16) the fp8
>   branch pinned `ag_bib := 0`, so `cur_b` stayed even and block index 1 was never written. Restored
>   the pre-widening `row->bib->g->sb` nesting for `numOutBlocks==1`; DIM=32 j-group path untouched.
> - **Fault B** — `ScaleFactorMem.scala` `depth_sram` was hardcoded `64`, ignoring `depth` (=128 for
>   DIM16 from the 16KB `scale_mem` config; =64 for DIM32). This capped a resident operand window at
>   2 banks*64 rows = 2048 B, so `read_row_addr_w = J*k_group+j` overflowed 7 bits past 128 rows and
>   wrapped upper K-groups onto lower rows. Now `depth_sram = depth` (and `totalSizeBytes`). No SW
>   change: the funct-27 loader writes linear byte offsets from 0, consistent under the deeper SRAM.
>   New ceiling = 256 rows/window (K=2048@N=64, K=1024@N=128 fit; K=2048@N=128=512 rows still won't).
>   Pre-regression Sep-15 simv backed up as `simv-chipyard.harness-MxGemminiRocketConfig-PREWIDEN`.
> - Gate A on `matmul_tiled_fp8_64x64_requant`, then mxl15/mxl7/mxl18. Gate B on mxl4 + mxl10-mxl14
>   (predicted: mxl12 PASS, mxl13/mxl14 the discriminators), then raise `LLAMA_KTILE_MAX` to `LLAMA_D`.


Written for a fresh session. Everything below was **measured**, not inferred; where something is a
hypothesis it says so. Background on the kernels is `llama_layer_hw_plan.md` §12; this file is
self-contained for debugging the RTL.

Both faults were found with the **MX bisection ladder**, `npu-exploration/baremetal/mxgemmini/`:
19 tiny self-checking ELFs, each one delta from a shape that passes, goldens from
`fp8_matmul_model.tiled_matmul_hwlike` (the same model that generates the llama headers).
**All 19 PASS on spike**, so any RTL failure is a real divergence from the functional model.

---

> **UPDATE 2026-09-21: Fault A is FIXED** (`mxl7` passes; the fix is an uncommitted change to
> `MxRequantizer.scala`, simv rebuilt 01:22). The section below is kept as the record of how it was
> found and what gates it. **Fault B is still open** and now has its own root-cause file:
> [`rtl_fault_b_kdepth.md`](rtl_fault_b_kdepth.md) — read that one for Fault B, not §"FAULT B"
> here, which only states the symptom.

## FAULT A — the requantizer never writes the 2nd MX block's E8M0 scale  [FIXED]

**A REGRESSION**, introduced by the mesh-widening commits. Fresh, narrow, and with a one-line
signature. Not present on the current FPGA bitstream, which is why `llama_attention` passes there —
**it will break the FPGA the moment the bitstream is rebuilt.**

### The signature

The ISA suite's own tests say it plainly. **Both** fp8 requant tests fail on the post-widening
simv, with byte-identical output:

```
$ matmul_tiled_fp8_64x64_requant      (Sep-20 simv)   $ matmul_tiled_fp8_64x64_chain    (Sep-20 simv)
Scale[0][1], Got: 0, Exp: 78                          C1_scale[0][1], Got: 0, Exp: 78
Scale[1][1], Got: 0, Exp: 79                          C1_scale[1][1], Got: 0, Exp: 79
Scale[2][1], Got: 0, Exp: 78                          C1_scale[2][1], Got: 0, Exp: 78
...                                                   ...
```

`matmul_tiled_fp8_64x64_chain` is the test the user confirmed **PASSES on the current FPGA
bitstream**. It fails here. That is the before/after in one line, on a test that predates this work
entirely.

**Block index `[1]` of every output row reads back as 0. Block index `[0]` is never reported
wrong.** The output is `[M][N/32]` = `[64][2]`, so with two 32-element MX blocks per row, only
block 0 gets a scale written.

**The FP8 codes are perfect** — `0/4096` in every rung. Only the scales are wrong. So the
requantizer computes correctly and fails to *write* the second block's exponent.

### Where to look

`46736b1~1..9eb04f3` changed `src/main/scala/gemmini/MxRequantizer.scala` by **236 lines**,
generalizing the scale from one wire to a vector:

```scala
- val scale_e8m0 = WireDefault(0.U(8.W))          // one block per row
+ val scale_e8m0_vec = Wire(Vec(numOutBlocks, UInt(8.W)))
+ val tilesPerMxBlock = scaleSize / (meshColumns*tileColumns)
```

At **DIM=16** one MX block spans **two** mesh tiles (`tilesPerMxBlock = 2`); at DIM=32 it spans one.
`Got: 0` is `WireDefault(0.U)` showing through, so the most likely shapes of the bug are: the write
enable / index only ever selects `blk = 0`, `numOutBlocks` elaborates to 1 on the DIM=16 path, or
the DRAM/window write loop iterates tiles where it should iterate blocks.

Also touched and worth a look: `ScaleFactorMem.scala` (61 lines), `Scratchpad.scala` (17),
`ExecuteController.scala` (11), `LoopMatmul.scala` (19).

### It is NOT `I`/`J`/`Kt` — it is the BLOCKS-PER-ROW count, and that is now measured

I built `mxl15`–`mxl18` to separate `I`, `J` and `Kt`, expecting a shape-specific fault. It is not
shape-specific in those terms. What separates pass from fail is **`GN = N/32`, the number of MX
blocks per output row**:

| rung | M, K, N | I, J, Kt | **GN** | scales wrong | Y |
|---|---|---|---|---|---|
| `mxl15` (control, = chain test's shape) | 64, 64, 64 | 4, 4, 4 | 2 | **82/128** | 4082/4096 |
| `mxl16` | 32, 64, 64 | 2, 4, 4 | 2 | **39/64** | 2041/2048 |
| `mxl17` | 64, 32, 64 | 4, 4, 2 | 2 | **76/128** | 4091/4096 |
| `mxl7` | 32, 32, 64 | 2, 4, 2 | 2 | **36/64** | 2047/2048 |
| **`mxl18`** | 64, 64, **32** | 4, **2**, 4 | **1** | **0/64 — PASS** | **0/4096 — PASS** |

**`GN=1` is completely clean and `GN=2` always fails**, across every combination of `I`, `J` and
`Kt`. `mxl18` also shows the *resident-window* write is fine at `GN=1` (Y is 0/4096), so this is
not two bugs: it is one per-block iteration that only ever covers block 0, in both the DRAM and the
resident paths.

`Y` fails in the `GN=2` rungs purely as a downstream consequence of reading those scales resident —
ignore its count until the scales are fixed.

**One loose end.** The counts are "half plus a few": 64 of 128 would be exactly the `[i][1]`
entries, and `mxl15` reports 82. So ~18 `[i][0]` entries are wrong too, and that is **not yet
explained**. Re-run `mxl7` (see below) to read the per-row dump and settle whether block 0 is also
disturbed or whether some of those rows are a second effect.

### Why nobody noticed

Commit `9eb04f3` is titled *"passing dim 32 **non requant** tests"*. The requant path was not
re-tested on the DIM=16 config, and that is exactly where this landed.

**Two sufficient regression tests already exist and both catch it** —
`matmul_tiled_fp8_64x64_requant` and `matmul_tiled_fp8_64x64_chain`. Nothing new needs writing to
gate the fix; they just need to be in the loop for DIM=16 after DIM=32 changes.

---

## FAULT B — one `loop_ws` with K=2048 computes the wrong reduction

**→ Root cause now identified: see [`rtl_fault_b_kdepth.md`](rtl_fault_b_kdepth.md).** In short, the
label "K=2048" is wrong: the governing quantity is the **B-side scale window**, `N*K/512` rows
against 128 readable, and `read_row_addr_w` in `ScaleFactorMem.scala` is 7 bits wide. The section
below records the symptom as originally measured.

**Pre-existing and real on the FPGA.** This is what originally broke the llama kernels. Currently
worked around, not fixed.

### What is known

```
mxl3   M=32 K=1024 N=64   TK=64    A scales 1024 B   B scales 2048 B   PASS
mxl4   M=32 K=2048 N=64   TK=128   A scales 2048 B   B scales 4096 B   FAIL 2036/2048
mxl5   the same 2048-deep matmul as 2 accumulating K-tiles of 1024     PASS, 0/2048
```

* Values come back with the **right sign and order of magnitude, 5–60% wrong**
  (`hw=0x4181`=16.125 vs golden `0x4172`=15.125; `hw=0xc0ce`=−6.44 vs `0xc082`=−4.06). A reduction
  that is not summing what it should — **not** a broken address walk, which would give garbage.
* **Reproduces bit-identically on the Sep-15 and Sep-20 builds** — same 2036 count, same diff
  values, same 87826 cycles. Deterministic and structural; not a race, not X-propagation.
* `mxl5` proves splitting the same reduction into two 1024-deep accumulating calls is **bit-exact**.
  So the fault is strictly *within one `loop_ws` call*, and cross-call accumulation is fine.

### Four things change across the mxl3→mxl4 cliff, and rungs exist to separate them

```
E8M0 groups along K     32  ->   64    <- crosses 32 EXACTLY. A 5-bit group index would wrap here,
K in elements         1024  -> 2048       and that would produce precisely this symptom.
A-side scale window   1024  -> 2048 B
B-side scale window   2048  -> 4096 B
```

| rung | shape | isolates | expected if the cause is… |
|---|---|---|---|
| `mxl10` | 32, **1088**, 64 | 34 groups — ONE past the last passing rung | groups → FAIL (and by a hair, which is itself evidence) |
| `mxl11` | 32, 1536, 64 | 48 groups — brackets the cliff | groups → FAIL |
| `mxl12` | 32, 2048, **32** | 64 groups, B window halved to 1024 B | B window → PASS; groups/A window → FAIL |
| `mxl13` | 32, 1024, **128** | 32 groups, B window at mxl4's 4096 B | B window → FAIL (only rung that can convict it) |
| `mxl14` | **64**, 1024, 64 | 32 groups, A window at mxl4's 2048 B | A window → FAIL |

`> 32 groups` and `A window > 1024 B` are the *same statement* at M=32 (window = groups × M bytes);
`mxl14` exists purely to separate them. **These five have never been run on RTL** — they are built
and waiting. Run them before opening any Scala: "it is a 5-bit group counter" and "the scale SRAM
is too shallow" need different fixes.

Likely files: `ScaleFactorMem.scala`, and the `gemmini_mx_load_scales` / group-index path.

---

## Reproducing

### The simulators, and a warning about them

```
sims/vcs/simv-chipyard.harness-MxGemminiRocketConfig         2026-09-15 01:45  PRE-widening
sims/vcs/simv-chipyard.harness-MxGemminiRocketConfig-debug   2026-09-20 22:57  POST-widening
```

**The `-debug` suffix is a coincidence, not the variable.** `-debug` merely means
waveform-capable; what matters is that the two binaries are **different elaborations**, five days
apart, straddling the mesh-widening commits. The whole Fault-A-is-a-regression conclusion rests on
the Sep-15 binary, and **re-elaborating that name destroys the only pre-regression artifact on
disk.** Copy it and its `.daidir` aside before building anything, or plan to re-elaborate at
`46736b1~1`.

Gemmini Scala commits in the window:

```
9eb04f3 2026-09-20  passing dim 32 non requant tests
9486934 2026-09-19  some progress with 32x32 mesh
46736b1 2026-09-17  widening mesh, adding configs. 32x32 mesh working for single throughput
```

### Run one ladder rung

```bash
cd /bwrcq/scratch/nicorakela/radiance-cy-dev/sims/vcs
E=/bwrcq/scratch/nicorakela/radiance-cy-dev/generators/gemmini/npu-exploration/out/baremetal/mx_rocket
D=/bwrcq/scratch/nicorakela/radiance-cy-dev/generators/testchipip/src/main/resources/dramsim2_ini

./simv-chipyard.harness-MxGemminiRocketConfig-debug \
  +permissive +dramsim +dramsim_ini_dir=$D +max-cycles=10000000 \
  +vcs+initreg+1 +ntb_random_seed_automatic +loadmem=$E/mxl15 \
  +permissive-off $E/mxl15 </dev/null 2>/dev/null
```

Swap `-debug` off for the pre-widening build. Add
`+fsdbfile=output/chipyard.harness.TestHarness.MxGemminiRocketConfig/mxl15.fsdb` for waveforms
(much slower). Every rung prints its own `DIM`, scratchpad geometry, spad map, per-stage mismatch
counts, and a PASS/FAIL line.

**The requant rungs now DUMP all `M*GN` scale bytes** plus two layout hypotheses. That dump was
added *after* the mxl7/mxl15 runs above, so **re-run `mxl7` to see the full per-row pattern** — it
should show `[i][1]` zeroed directly.

### The ISA regression test (the one to gate on)

```bash
B=/bwrcq/scratch/nicorakela/radiance-cy-dev/generators/gemmini/software/gemmini-rocc-tests/build_mx_rocket/bareMetalC
./simv-chipyard.harness-MxGemminiRocketConfig-debug \
  +permissive +dramsim +dramsim_ini_dir=$D +max-cycles=10000000 \
  +vcs+initreg+1 +ntb_random_seed_automatic \
  +loadmem=$B/matmul_tiled_fp8_64x64_requant-baremetal \
  +permissive-off $B/matmul_tiled_fp8_64x64_requant-baremetal </dev/null 2>/dev/null
```

### Rebuild the ladder, or add a rung

```bash
cd generators/gemmini/npu-exploration/baremetal/mxgemmini
make ladder        # both targets -> ../../out/baremetal/{spike,mx_rocket}/
make run-ladder    # on spike, in bisection order -- all 19 must PASS
```

A new rung is a line in `RUNGS` in `gen/gen_mx_ladder.py` plus a 6-line `src/mxlN.c` copied from a
neighbour and a name in the Makefile's `LADDER`. `gen/gen_mx_ladder.py mxlN` emits just its data.
**Always confirm a new rung passes on spike before reading anything into an RTL failure.**

### A trap that will cost an hour

`software/gemmini-rocc-tests/include/gemmini_params.h` is **shared with the ISA suite and gets
hand-flipped per bitstream** (`DIM 16 / BANK_ROWS 4096` ↔ `DIM 32 / BANK_ROWS 2048`). It flipped
mid-session and `mxl4` stopped *compiling*. The ladder and the llama kernels now pin both
(`LADDER_DIM`, `LADDER_BANK_ROWS`, `LLAMA_BANK_ROWS`) and `#warning` on a mismatch, so they no
longer follow it — but **anything in the ISA suite still does.** Check it before trusting an ISA
test result. Overrides: `-DLADDER_BANK_ROWS=N`, `-DLLAMA_BANK_ROWS=N`.

---

## Already ruled out — do not re-investigate

| hypothesis | verdict | evidence |
|---|---|---|
| non-square tile grid (`I≠J`) | **fine** | `mxl1` passes at `I=2, J=4` |
| deep reductions in general | **fine to K=1024** | `mxl2` (TK=16), `mxl3` (TK=64) pass |
| `ex_accumulate` (rs1 bit 0) ignored by MX RTL | **honoured, and bit-exact** | `mxl5`. This closes `llama_layer_hw_plan.md` §10.3's "UNVERIFIED ON RTL" |
| output region not cleared on `ex_accumulate=0` | **cleared correctly** | `mxl6`: two matmuls into one `C_spad`, both 0/2048 |
| strided A mvin (2048-byte row pitch) | **fine** | `mxl8` |
| host fp32 glue (`mx_host.h`, `expf`, RMSNorm, RoPE, softmax) | **fine** | `mxl9`: 0 mismatches on RTL |
| B-side spad tile ordering `(j*tK+k)` vs `(k*tJ+j)` | **both correct** | the ISA test swaps its loop variable names and is square, so the apparent transpose cancels; both agree with `loop_ws`'s `(k*J+j)` walk (`gemmini.cc:808`, `:1291`) |
| the requant scale write is transposed, or has a wrong row pitch | **neither** | the runtime hypothesis tests score 34/128 and 9/128 — it is a *missing* write, not a relayout |
| my ladder operands being pathological | **they are ordinary** | `mxl7`'s O scales span 128..129, no zeros, no saturation, peak code 0x40 |
| `mxl4`'s spad map overrunning the scratchpad | **it fits** | 12544 of 16384 rows; and the user confirmed the FPGA geometry |

---

## Current workaround in the kernels (remove when Fault B is fixed)

`LLAMA_KTILE_MAX = 1024` in `src/llama_mlp.c` and `src/llama_attention.c` caps the per-`loop_ws` K,
so a D=2048 projection runs as two accumulating K-tiles. Every graded number on spike is unchanged
(MLP 119815 ppm, attention 150192 ppm), because `mxl5` is bit-exact. **`llama_attention` passes on
the FPGA with this cap** — the end-to-end confirmation that Fault B is the fault that mattered.

`-DLLAMA_KTILE_MAX=2048` reproduces the failure. Raise it to `LLAMA_D` when Fault B is fixed.

**Still uncapped: `llama_attention_full`.** Its `mesh_matmul(LLAMA_M, LLAMA_D, nc, ...)` keeps the
full 2048 contraction in one call and has no K-tiling to cap; adding one means threading
`ex_accumulate` through its projection loop. It should still fail on RTL in the projections.

---

## Suggested order

1. **Fix Fault A** in `MxRequantizer.scala` — the diagnosis is already as tight as a ladder can make
   it (`GN=1` clean, `GN=2` always broken, codes perfect, both write paths affected), so this is
   ready for a code read rather than more experiments. Gate on `matmul_tiled_fp8_64x64_requant`
   (whose `Scale[i][1], Got: 0` is the clearest signal available), then `mxl15`, `mxl7`, `mxl18`.
   Urgent: fresh, three commits wide, and it blocks the next bitstream.
2. **Re-run `mxl7`** for the per-row scale dump, to settle the unexplained ~18 wrong `[i][0]`
   entries in `mxl15`. Cheap, and it may turn out to be part of the same fix.
3. **`mxl10`–`mxl14`** on RTL to name Fault B's cause, then fix it and remove the K cap. These have
   never been run; they are built and waiting.
