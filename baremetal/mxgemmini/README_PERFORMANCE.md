# MxGemmini performance: fast kernels and how to program them

This covers which kernels in this directory are the **fast path** and what they achieved on RTL. It also
explains how to program MxGemmini for high utilization, using TinyLlama as the worked example.

Target: `MxGemminiRocketConfig` (DIM 16, fp8 E4M3 single × single, E8M0 block scales per 32 elements, BF16
or FP8-requant output). Every result was measured on RTL (VCS or FireSim, noted where it matters) and
cross-checked bit-exact on Spike. Paths below are relative to this directory; `../../../` is the gemmini
repo root.

**Contents:**
1. Utilization benchmarks
2. Llama kernels: fast path vs baseline
3. Building and running
4. The machine in numbers
5. Native DRAM loops
6. Hiding load latency across a layer
7. Why M = 32 tops out near 70%
8. Residency
9. Scale loading and ordering
10. What moved the numbers
11. Measuring and debugging
12. Pitfalls
13. Next steps

---

## 1. Utilization benchmarks: start here

One large matmul, random fp8 data from a DRAM-cold blob (`gen/gen_mx_bench.py` → `data/mx_bench.{bin,h}`).

| kernel | shape (M × K × N) | what it measures | RTL result |
|---|---|---|---|
| `mx_bench_matmul` | 128 × 4096 × 1024 | **latency hiding**: compute-bound (~4.2 B/c demand) | **99.3%** (2,111,015 / 2,097,152 ideal) |
| `mx_bench_matmul_m32` | 32 × 4096 × 1024 | llama-shaped, memory-bound (~8.9 B/c demand) | **72.1%** (726,406 / 524,288), FireSim |
| `mx_bench_matmul_m32_blk` | same, B pre-blocked | whether the DRAM layout of the weights matters | 71.6%: it doesn't |

Each benchmark prints three passes:
- **pass 0:** a load-only 4 MB stream, the platform's DMA ceiling;
- **pass 1:** `PERF … util=`, traffic and B/c, plus `PERF_HW`, the mesh busy/idle split;
- **pass 2:** `PERF_LITTLE`, with reads in flight, latency per 64 B, and the % of cycles with the request
  table full (`xact_full`), the bus refusing requests (`tl_wait`), or the mvin tracker full (`ld_cmd_full`).

Read `PERF_LITTLE` like this:
- `xact_full` high: Gemmini's 32-request cap is the limit.
- `tl_wait` high: the memory system is saturated.
- `ld_cmd_full` high with `xact_full` low: too few mvins in flight.

`CHECKSUM` must match Spike:

| kernel | Spike checksum |
|---|---|
| M = 128 | `sum=4391224426 xor=5a410dab13820ef0` |
| M = 32, both variants | `sum=1094338120 xor=2623395b490a2320` |

## 2. Llama kernels: fast path vs baseline

Each native kernel is the same source as its baseline with `-DMLP_NATIVE`, `-DATTN_NATIVE` or
`-DLAYER_NATIVE` set by a small shim in `src/`. Cycles are RTL mesh cycles; lower is better.

| fast path | baseline | shape | native | baseline | ideal |
|---|---|---|---|---|---|
| **`llama_layer_full_native_gh`** | `llama_layer_full` | one full layer: 32 heads, F = 5632, host stages golden | **7,833,998 (70%)** | — | 5,521,408 |
| `llama_layer_full_native` | `llama_layer_full` | same, host stages computed | same mesh schedule | — | 5,521,408 |
| `llama_mlp_native` | `llama_mlp` | M32 D2048 F64 | 81,781 (60%) | — | 49,152 |
| `llama_attention_native` | `llama_attention` | M32 D2048, 1 head | 109,387 (60%) | — | 66,048 |
| `llama_mlp_small_native` | `llama_mlp_small` | M32 D256 F64 | **9,905** | 20,771 (**2.1×**) | 6,144 |
| `llama_mlp_tiny_native` | `llama_mlp_tiny_db` | M32 D64 F64 | **3,373** | 6,807 (**2.0×**) | 1,536 |
| `llama_attention_tiny_native` | `llama_attention_tiny` | M32 D64, 1 head | **5,934** | 10,158 (**1.7×**) | 2,560 |
| `llama_attention_small_native` | `llama_attention_small` | M32 D256, 1 head | not yet run on RTL | | 8,704 |

The ISA tests in `../../../software/gemmini-rocc-tests/bareMetalC/` show the same progression on a
128×128×128 fp8 matmul (ideal 8192):
- scratchpad flow: 10,639 compute-only (77%); 9,077 with 4 manual chunks and `acc_banks = 2` (90%);
- native loops (`matmul_tiled_fp8_128x128_dramloop_ls`): 10,649 loop cycles, 11,060 total, with the output
  store included.

Notes:
- **`_gh`** (`-DLLAMA_GOLDEN_HOST`) replaces RMSNorm, RoPE, softmax, SwiGLU and residual 1 with their goldens
  from the blob. The RTL run is then mostly mesh time, and every mesh stage is still checked. The full-layer
  kernels print a per-phase table (QKV, S, O, o_proj, G/U, down) with measured cycles, ideal cycles and utilization.
- **Attention** keeps `P@V` as the resident scratchpad-requant loop (section 8). O and its scales feed o_proj in
  place in the D ≤ 2048 kernels; in the full layer they are concatenated across heads for one K = 2048 o_proj.
- **Other scratchpad-flow variants** (`llama_mlp_tiny`, `_tiny_t1`, `_tiny_db`, `llama_attention_tiny_t1`,
  the `_full` kernels, the ladder) are kept as baselines and for debugging.
- `llama_mlp_tiny_native_ua` is an alignment A/B test only.

## 3. Building and running

```bash
export RISCV=/bwrcq/scratch/nicorakela/radiance-cy-dev/.conda-env/riscv-tools PATH=$RISCV/bin:$PATH
O=$(realpath ../../out/baremetal)                       # targets must be absolute paths
make $O/mx_rocket/mx_bench_matmul $O/spike/mx_bench_matmul
make $O/mx_rocket/llama_layer_full_native_gh
python3 gen/gen_mx_bench.py                             # regenerate the benchmark blob

# Spike with the local MX model (its "cycles" are instruction counts, not timing)
(cd ../../../software/libgemmini && make libgemmini.so)
spike --extlib=../../../software/libgemmini/libgemmini.so --extension=gemmini $O/spike/mx_bench_matmul
```

- **Hardware:** `MxGemminiRocketConfig` (DIM 16, `acc_banks = 2`, 512-bit system bus) with the loop-managed
  scales, the cross-loop WAR hold in `LoopMatmul`, and the narrow-Put fix.
- **ISA tests:** build them in `../../../software/gemmini-rocc-tests` with `./build_mx_rocket.sh bareMetalC`
  and `./build_spike.sh bareMetalC`. Touch the `.c` file after a header change, because header dependencies
  aren't tracked.

---

## 4. The machine in numbers

| resource | size / rate | consequence |
|---|---|---|
| mesh | 16×16; one 16×16×16 tile-op per cycle | ideal cycles = M·N·K / 256 |
| scratchpad | 16384 rows × 16 B, two halves of 8192 rows | a loop's A and B each live in one half |
| accumulator | 64 KB = 512 rows, two 256-row halves (`acc_banks = 2`) | E4M3-single C uses M·N/64 rows, so N ≤ 512 per loop at M = 32 |
| scale memory | per operand (act, wgt): two 4 KB halves | act slice = (K/32)·M B, wgt slice = (K/32)·N B per loop |
| DRAM (64-bit memory bus, 8 B/c peak) | **~6 B/c sustained** (FireSim, mixed read/write); 7.5 B/c on a 16 KB cold burst | the ceiling that matters |
| L2-warm mvin / scale load | ~15.2 B/c / ~6 B/c (cold ~3) | spad write max is 16 B/c (one row per cycle) |
| BF16 store to DRAM | ~5.6 B/c | |
| system bus | 512-bit (`WithSystemBusWidth(512)`), DMA `max_in_flight_mem_reqs = 32` | |

**Roofline for llama prefill.** A 256 B weight tile feeds M cycles of compute, so weights need 256/M B/c:
**8 B/c at M = 32, above the ~6 B/c the memory system sustains**. Decode (M = 1) sits far below it. A BF16
output needs 512/K B/c (8 at K = 64). Once per-loop overheads are hidden, llama is memory-bound, and end to
end the host's fp32 glue dominates (~460× the mesh on the tiny MLP).

## 5. Native DRAM loops (the fast path)

One `gemmini_loop_ws_mx` is one matmul: A and B stream from DRAM, C (BF16) goes straight back to DRAM, and the
loop loads its own E8M0 scale slices. There are no manual mvins, mvouts or scale loads, no packing, and no
fences between loops.

```c
gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, /*BF16*/3, 0);
gemmini_extended3_config_ld(A_pitch, MVIN_SCALE_IDENTITY, false, 0);   // A row pitch (bytes)
gemmini_extended3_config_ld(B_pitch, MVIN_SCALE_IDENTITY, false, 1);   // B row pitch
gemmini_config_st(C_pitch * sizeof(uint16_t));
gemmini_loop_ws_mx(I, J, K,                              // tiles: M/16, N/16, K/16
                   A, B, C, A_pitch, B_pitch, C_pitch,   // elements
                   A_sc, B_sc, A_sc_pitch, B_sc_pitch,   // [K/32][M] and [K/32][N] E8M0 rows, pitch in bytes
                   /*ex_accumulate*/ false, a_spad_id, b_spad_id);
gemmini_fence();                                         // only where the host reads C
```

Per loop, the hardware:
1. Issues two 2-D `MX_LOAD_SCALES` into the scale half for the loop's slot. Slots alternate 0, 1, 0, 1 over
   consecutive `LOOP_WS`.
2. Issues a `CONFIG_SCALE_MEM` with bounds equal to this loop's I/J/K.
3. Runs operand loads, compute and the acc → DRAM store pipelined.

Loop c's store overlaps loop c+1's compute because acc halves alternate automatically. That needs
`acc_banks = 2`; otherwise stores serialize behind the next loop.

**Chaining rules (what keeps the mesh fed):**
- **Issue independent loops back to back.** Fence only where the host reads a result.
- **Reuse A.** Pass `A = NULL` for loops that share A with the previous one (U after G; K and V after Q;
  N-chunks). A stays in `a_spad_id`'s region.
  - If the A-scale slice is also identical (same address, pitch, I, K) and the loop lands in the slot that
    last loaded it, the A-scale load is skipped.
  - Any non-loop command (config, manual scale load, mvin) clears that record.
- **Double-buffer B.** Alternate `b_spad_id` 1/2. `a_spad_id = 1` puts A at row 0; `b_spad_id = 1 / 2` puts B
  at the top of half 0 / 1.
- **N-chunk** when M·N/64 exceeds 256 acc rows (N > 512 at M = 32): C at `C + c*NC`, B at `B + c*NC`, B scales
  at `&B_sc[0][c*NC]` with pitch N, and `A = NULL` after chunk 0.
- **Split K** across consecutive loops of one chunk: `ex_accumulate` on every K-tile after the first, and C
  only on the last K-tile. A chunk's K-tiles must be consecutive loops, so they share one acc half.

**Limits.** E4M3-single × single, BF16 out; no padding, D or transpose. Per loop: A ≤ 8192 rows, B ≤ 8192 rows,
C ≤ 256 acc rows, each scale slice ≤ 4 KB. Scale arrays must be 8 B-aligned (the loader asserts), and DMA
buffers should be `aligned(64)`.

## 6. Hiding load latency across a whole layer

`mxn_matmul` ([`include/mx_native.h`](include/mx_native.h)) runs every weight matmul as N-chunks × K-tiles of
native loops. Each loop's A and B sit in one scratchpad half, and halves alternate:

```c
for (c = 0; c < N / NC; c++)                 // N-chunk: one 256-row acc half
  for (t = 0; t < K / Kt; t++) {             // K-tile: accumulate, C only on the last
    int h = 1 + (mxn_loops++ & 1);           // global counter: halves alternate across calls too
    gemmini_loop_ws_mx(M/16, NC/16, Kt/16, A + t*Kt, B + t*Kt*ldb + c*NC,
                       t == K/Kt - 1 ? C + c*NC : NULL, lda, ldb, ldc,
                       &A_sc[t*Kt/32][0], &B_sc[t*Kt/32][c*NC], M, N, /*accumulate*/ t > 0, h, h);
  }
```

What gets hidden:
- **The next loop's A and B** load into the other half while the current loop computes.
- **The next loop's scales** load into its own scale half.
- **A chunk's C store** overlaps the next chunk's compute, because acc halves alternate.
- **Reuse of a half:** loop n+2 reuses loop n's half only after loop n's slot frees.
- **Across matmuls:** consecutive matmuls with no host step between them (Q → K → V, G → U) overlap too.

**Tile choice** is automatic (`mxn_pick_nc` / `mxn_pick_kt`):
1. The largest NC ≤ 512 whose C fits an acc half (M·NC/64 ≤ 256 rows).
2. Then the largest Kt ≤ 512 whose A and B fit one half together and whose scale slices fit 4 KB.

At M = 32 (2·Kt + Kt·NC/16 ≤ 8192 rows) that gives **Kt = 128, NC = 512**: 8192 compute cycles against 68 KB of
loads per loop, balanced at ~8.3 B/c, with a 6% A re-read. Layer loops:

| layer matmul | K-tiles × N-chunks |
|---|---|
| Q, o_proj | 16 × 4 |
| K, V | 8 × 1 (Kt = 256, NC = 256) |
| G, U | 16 × 11 |
| down | 44 × 4 |

`mxn_matmul_ex` also accepts a pre-blocked B (each loop's tile contiguous, pitch NC).

Still exposed:
- the first loop of each chain;
- `config_ld` / `config_st` between matmuls;
- a ~35-cycle scale-config drain per loop;
- host seams (fences);
- the 32 tiny per-head S and O matmuls.

**Never reload into a region the other in-flight loop is using.** With explicit spad ids, the newer loop's
mvins reach the reservation station ahead of the older loop's remaining computes, which then read the new
data. `LoopMatmul` now holds such a load until the older loop has issued all its computes. The result is
correct but serialized: a K-tiled 128³ test that reloads A in place ran 13,254 cycles vs 10,649. Alternating
halves never triggers the hold. Upstream Gemmini never hit this: its software uses per-slot halves (spad
id 0), and explicit ids only with NULL pointers.

## 7. Why M = 32 tops out near 70%

The compute-bound benchmark runs at **99.3%**, so the schedule hides load latency. At M = 32 the same
schedule reaches ~72% (single matmul) and 70% (full layer). The gap is the **memory system**, not Gemmini:

| FireSim load counters, llama-shaped matmul | value | meaning |
|---|---|---|
| `tl_wait` | **80%** | the DMA has a request ready and the bus refuses it |
| `xact_full` | 1% | the 32-request cap is almost never hit |
| reads in flight / latency | 26.8 / 283 cycles | Little's law: 26.8 × 64 B ÷ 283 = 6.05 B/c, the measured rate |
| pre-blocked weights | 71.6% vs 72.1% | no change, so not DRAM row locality |

The path behind Gemmini (L2 → 64-bit memory bus → DRAM) sustains ~6 B/c, roughly 76% of the bus's 8 B/c peak,
against the ~8.9 B/c a fed mesh needs at M = 32. The ways past it:
- **fp4 (E2M1) weights** halve the weight traffic, bringing M = 32 demand to ~4.5 B/c.
- **More tokens per weight byte.** M = 128 runs at 99.3%.
- **A wider or second memory channel.**

## 8. Residency: keep activations and scales on chip

At a seam with no host op (attention's `P@V → o_proj`), run the producer as a scratchpad loop with FP8 requant
(`gemmini_mxquant_config_mvout_resident` + `LOOP_WS_REQUANT_TILED`). Its codes land in the spad in operand-A
tiled layout, and its E8M0 bytes in the act-scale window. Consume both in place with a native loop:

```c
gemmini_mx_load_scales_2d(&Wo_sc[0][c*NC], NC, K/32, N, /*dest*/ (c & 1) << 12, /*wgt*/ 1);
gemmini_mxquant_config_mvout_wait(sink, I, NC/16, K/16, /*act half*/ 0, /*wgt half*/ c & 1, 1);
gemmini_loop_ws(I, NC/16, K/16, 0,0,0, NULL /*A resident at row 0*/, Wo + c*NC, NULL, Y + c*NC,
                K, N, 0, N, false,false,false,false,false, NO_ACTIVATION, /*a_spad_id*/ 1, 1 + (c & 1), false);
```

- The resident producer must write O at spad row 0, which is `a_spad_id = 1`'s A region.
- Manual scale loads aren't ordered against queued computes. `_wait` (rs2 bit 16) makes the config wait for
  the load.
- Reusing a scale half for chunk c+2 is safe only because each chunk issues far more ex commands (2·I·J·K)
  than the 16 reservation-station ex slots.

## 9. Scale loading and ordering

| instruction | encoding | ordering |
|---|---|---|
| `MX_LOAD_SCALES` (27) / `gemmini_mx_load_scales_2d(dram, row_bytes, rows, pitch, dest, sel)` | rs1[39:0] addr, [63:40] pitch; rs2[31:0] bytes/row, [32] sel (0 act, 1 wgt), [45:33] dest (half = bit 12), [53:46] rows, [54] gated | bypasses the RS: needs a fence, a `_wait` config, or loop management |
| `CONFIG_SCALE_MEM` (26) / `gemmini_mxquant_config_mvout[_wait]` | rs1: I/J/K bounds, [60] act half, [61] wgt half; rs2[16] wait for loads, [17] loop-managed, [18] act scales reused | executes only after the mesh drains (~35-cycle gap) |
| `LOOP_WS_CONFIG_SCALES` (31) / `LOOP_WS_CONFIG_SCALE_STRIDES` (32) | A/B scale bases, and pitches, for the next `LOOP_WS` | hardware per-half state FREE → LOADED → INUSE → FREE: a load into a half waits until its previous user's config is superseded |

The loader keeps 8 Gets in flight behind a 4-deep queue and runs loads one at a time. A loop's first config
waits ~200 cycles for its scales; after that, scale loads are fully hidden. The old scratchpad-flow tricks are
unnecessary with loop-managed scales: packing scales per chunk, or one config with `k_bound = NCHUNKS·K` so the
counters run on.

## 10. What moved the numbers

In order:
1. **64 B-row mvins** (4 tiles per mvin): −11%.
2. **An 8-slot pipelined scale loader**, and 32 DMA requests in flight.
3. **`acc_banks = 2`**, so a store overlaps the next compute.
4. **DRAM loops:** no mvout pass, no per-matmul fence, weights double-buffered. About 2× on llama.
5. **Loop-managed scales:** no fence or packing, hidden after the first loop.
6. **Per-loop halves across a layer (`mxn_matmul`):** 99.3% when compute-bound.

## 11. Measuring and debugging

- **Counters.** `../../../software/gemmini-rocc-tests/include/mx_perf.h`: `MX_PERF_MARK` (fence + rdcycle) phases,
  plus HW counter sets for compute (`EXE_ACTIVE`, `MAIN_EX`/`ST_EX`/`LD_EX` …) and loads (`LOAD_DMA_WAIT`,
  `RDMA_XACT_FULL`, in-flight via Little's law). Read external counters live; snapshots mis-read them.
  `mx_bench_matmul.c` shows both sets in use.
- **ISA tests** (`../../../software/gemmini-rocc-tests/bareMetalC/`):
  `matmul_tiled_fp8_128x128_dramloop{,_nc,_nc4,_nc_wait,_ls,_ls4,_kt}`, and `mx_mem_bw` for short 16 KB bursts
  (latency-dominated, not a sustained ceiling).
- **Waveforms** (`../../../waveform-debug/npi/`, through the NPI reader only):
  - `mlp_native_phases.py`: per-stage DMA / mesh / store timeline;
  - `loop_gaps.py`: mesh gaps vs DMA;
  - `lds_timeline.py`: scale loads and configs;
  - `loop_reqs.py`: per-loop acc and store requests;
  - `sfout_puts.py`: requant scale Puts.

## 12. Pitfalls

- **Platforms:** VCS and FireSim use different DRAM models, so bandwidth numbers don't transfer between them.
  Compare on one platform.
- **Layout noise:** array placement alone moves DRAM-bound kernels by ±3%.
- **Narrow Puts** must be lane-shifted into the beat. This broke requant DRAM scales when the bus went from 256
  to 512 bits; it is fixed.
- **Stale builds:** rebuild clean before debugging a "failure".
- **Alignment:** regenerated headers lose alignment unless the generator emits it. `gen_llama_layer.py` now does.
- **Spike timing:** Spike "cycles" are instruction counts. Use Spike for bit-exactness, never timing.

## 13. Next steps

- Get past the ~6 B/c memory ceiling at M = 32: fp4 (E2M1) weights with E4M3 activations (the asymmetric MX
  modes already exist), or more tokens per pass.
- Run the 22-layer model on Spike for perplexity. `llama_layer_full_native` already passes bit-exact on RTL
  at 70%.
- FP8-requant output from native loops.
- Overlap consecutive scale loads, and a drain-free scale config (each ~1% at llama sizes).
- Move the host glue onto the accelerator.
