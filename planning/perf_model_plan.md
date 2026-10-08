# perf_model_plan — a timing model inside the spike functional model

Status: **PLAN, nothing implemented** (2026-10-04). Steps are confirmed one at a time with the user.

## 1. Decisions (user, 2026-10-04)

* Build a new performance model **inside libgemmini** (`../software/libgemmini/gemmini.cc`), driven by
  the same RoCC instruction stream the functional model executes. It replaces
  `models/perf/perf.py`'s call into `MxGemmini-workspace/ppa/perf/perf_model.py` as the source of the
  `PERF` line; that model's measured constants and timelines are kept as calibration input (§6).
* **Three modes**, chosen at run time, same `.so`:

  | mode | math (bit-exact) | timing | use |
  |---|---|---|---|
  | `func` | yes | no | today's behaviour, VERDICT grading |
  | `perf` | **no** | yes | fast cycle estimates; must be the fast path |
  | `both` | yes | yes | one run gives VERDICT and cycles |

* `perf` **speed is the top requirement**.
* Model the **VPU** (`VPU_EXEC`) and `SPAD_REQUANT`.
* A **fixed CPU-cost estimate** for the host (Rocket), not tuned per experiment, so "without the VPU"
  baselines (nonlinears on the CPU) are stable comparisons.
* Accuracy target: **≤5 %** total cycles vs VCS on the accelerator side. The CPU estimate is expected
  at ~10–15 % and is labelled as an estimate, not a hardware claim.

## 2. The hook

All Gemmini commands enter through `CUSTOMFN(XCUSTOM_ACC)` (`gemmini.cc:2456`) already decoded.

```cpp
reg_t gemmini_t::CUSTOMFN(XCUSTOM_ACC)(rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  if (mode != FUNC) timing.on_cmd(insn.funct, xs1, xs2, host_cycle());   // timing sees every command
  if (mode == PERF) return timing.value_for(insn.funct, xs1);            // counters/fence answered by the model
  ... existing dispatch, unchanged ...
}
```

* Mode from an environment variable (`GEMMINI_MODE=func|perf|both`, default `func`), read once at
  `reset()`. The runner (`compiler/targets/mx_gemmini_rocket/backend/runner.py:204`) sets it; no rebuild.
* The timing model is a separate class (`gemmini_timing.{h,cc}`) that owns **no data, only time**:
  it never reads `gemmini_state.spad`/`mx_smem`. That makes `both` ≡ `perf` in cycles by
  construction, and gives a free self-check: **`perf` and `both` must report identical cycles**.
* It needs the config state (loop bounds, formats, LUT/scale settings) that `config`/`loop_ws_config_*`
  write. Those handlers are cheap and stay running in `perf` mode; only the heavy work is skipped.

## 3. What makes `perf` fast

| cost in `func` today | `perf` mode |
|---|---|
| per-MAC `fp_quantize_rne`/`fp_add_exact` in `mx_loop_ws_spad` (O(M·N·K)) | skipped |
| mvin/mvout byte-by-byte MMU loads/stores | skipped (timing only needs sizes + addresses) |
| requant post-passes, LUT decode, VPU math | skipped |
| `LOOP_WS` timing | one event per tile step: O(I·J·K) tile steps (a 1024³ GEMM at DIM 16 ≈ 262 K steps ≈ tens of ms); closed-form fast-forward of the steady state only if profiling shows it is needed |
| host code | **still interpreted by spike** — control flow and addresses need it. This is the floor: `perf` mode costs about spike's own speed on the host instructions |

Caveat: host code that reads accelerator results (self-checks, printing mismatches) sees garbage in
`perf` mode and may branch differently. Timing runs build with the self-check off (a `-D` flag);
anything else that reads results is listed in the plan when found.

## 4. What the timing model contains (RTL source of each latency)

| structure | models | RTL |
|---|---|---|
| command path | RoCC queue depth, backpressure to the CPU, `fence` waits for idle | `Controller.scala` |
| reservation station | load / execute / store / vec queues, address dependencies, issue width | `ReservationStation.scala` |
| loop unrollers | `LOOP_WS` (DRAM and spad MX variants) → mvin/preload/compute/mvout in RTL order; scale loads + reuse hit | `LoopMatmul.scala` |
| DMA | latency + bytes/cycle, per-request overhead | `DMA.scala`, `LoadController.scala`, `StoreController.scala` |
| mesh | fill + 16 cycles/block, PE mode (single/quad/dual) | `ExecuteController.scala`, `MxConfigFragments.scala` |
| requantizer | drain rate + latency, overlap with the next tile | `MxRequantizer.scala` |
| LUT / scale loads | `MX_LOAD_LUT`, `MX_LOAD_SCALES` word-serial DMA | `Scratchpad.scala` |
| VPU | one row/cycle, +1 on same-bank src1/src2, reductions, `stall`, `vpu_units` | `vpu/Vpu.scala` |
| spad requant | vec issue path | `SpadRequant.scala` |

## 5. Host CPU estimate

Fixed per-class costs for in-order Rocket (int, FP add/mul, `fdiv`/`fsqrt`, load/store, branch) plus a
simple cache-miss penalty, accumulated as a running `host_cycle()` so the CPU and Gemmini share one
timeline (command issue time, queue-full stalls, fences). Needs a cheap per-retired-instruction counter
in spike's core — the open design point (patch the execute loop vs commit-log post-processing, which is
too slow for `perf`). Checked once against VCS `llama_attention_small` (host-dominated).

## 6. Calibration and validation

* Ground truth: `sims/vcs/mx_vcs_logs_*` (VCS, many configs/tests); per-stage boundaries measured
  with the NPI/FSDB tooling (`waveform-debug/npi/`).
* Starting constants from `MxGemmini-workspace/ppa/perf/perf_model.py` `CAL` (measured on Radiance):
  16 cycles/compute block, +25 fill/tile, requant 27 outputs/cycle + 50, `MX_LOAD_LUT` 100 + 49/8-byte
  word. Its Radiance-host terms (80 cycles/GPU-issued instruction, 1800 setup, GPU-written scales) do
  **not** transfer to Rocket. Its waveform timelines (`perf/tl_*`) are a cross-check of mesh/DMA/LUT
  phases.
* Gate: ≤5 % total cycles on the `run_mx_vcs.sh` regression, with a held-out subset not used for fitting.

## 7. Steps (each confirmed before it starts)

**Decision (user, 2026-10-04): the timing code lives outside `gemmini.cc` as much as possible.**
It is `libgemmini/gemmini_perf.{h,cc}`; `gemmini.cc` carries only the hook line in `CUSTOMFN` and
`perf.reset()` in `gemmini_t::reset()`; `gemmini.h` only the include and the `perf` member.
The perf-file funct list mirrors `gemmini.h`'s private `*_funct` constants (not reachable from outside
the class) -- keep both in step when a command is added. Build: `Makefile` (all three .so) and
`models/spike/build_spike.py` (`SOURCES` + compile line) compile `gemmini_perf.cc`.

Step 1 status: implemented 2026-10-04, see §8.

1. Mode switch + hook skeleton: `GEMMINI_MODE`, `perf` skips the math, timing class returns 0.
   Gate: `func` unchanged (VERDICT on all kernels), `perf` runs every kernel without crashing.
2. Measure `perf`-mode wall time vs `func` on the llama kernels; decide on the host counter (§5).
3. Command path + reservation station + DMA + mesh for plain `LOOP_WS`. Calibrate on the fp8 ISA tests.
4. MX extras: scales, LUTs, requantizer, resident chains.
5. VPU + spad requant.
6. Host CPU estimate.
7. `models/perf/perf.py` reads the new timeline; energy still from `models/ppa` at the achieved utilization.
8. Validation sweep vs VCS; record per-test error here.

## 8. Step 1 results (2026-10-04)

Built into a scratch `.so` (`MX_LIBGEMMINI`), in-tree `libgemmini.so` untouched. All 65 ELFs in
`out/baremetal/mx_rocket`, run three ways (stock `.so`, new `.so` in `func`, new `.so` in `perf`):

* **`func` is byte-identical to the stock model on all 65** (stdout `cmp`). `llama_model_m64` hits the
  1800 s timeout in both, identically.
* **`perf` runs every ELF to completion.** 62/65 then exit 1 at the kernel's own self-check (garbage
  outputs, as expected); the 3 without a self-check (`mx_bench_matmul*`) exit 0.
* **Speed (answers Step 2's question early): the Gemmini math, not the host, is the functional cost.**

  | ELF | func | perf | × |
  |---|---|---|---|
  | `llama_model` | 1437 s | 14.0 s | 103 |
  | `llama_model_l2` | 210 s | 3.1 s | 69 |
  | `llama_layer_full` | 65.6 s | 0.55 s | 119 |
  | `llama_mlp_full` | 50.9 s | 0.37 s | 137 |
  | `mx_bench_matmul` | 48.7 s | 0.036 s | 1370 |
  | `llama_model_m64` | >1800 s (timeout) | 30 s | >60 |

  So `perf` mode's floor is spike interpreting the host: ~2 G host instructions in 14 s for `llama_model`.

* **Finding: host work is data-dependent, so `perf` mode changes the host instruction count.** The
  kernels' own counters (spike `rdcycle` = retired instructions) agree on the mesh (identical) but not
  on the host, because the host runs on zeros/garbage instead of real activations:

  | ELF | host, func | host, perf | Δ |
  |---|---|---|---|
  | `llama_layer_full` | 82.25 M | 76.40 M | −7.1 % |
  | `llama_model` | 2093.9 M | 1902.7 M | −9.1 % |
  | `llama_mlp_full` | 50.37 M | 45.97 M | −8.7 % |
  | `llama_attention_full` | 28.45 M | 27.01 M | −5.1 % |
  | `attn_vpu` (host fp32 softmax) | 264.7 K | 172.4 K | −35 % |

  libm (`expf`, etc.) and the host glue take data-dependent paths. The accelerator timeline is unaffected
  (no data dependence); the **host estimate in `perf` mode is not trustworthy as-is**. Decision pending (§9).

## 9. Host accuracy in `perf` mode — DEFERRED (user, 2026-10-04)

The §8 finding (host instruction count 5–35 % low in `perf` mode, data-dependent libm/glue on garbage
outputs) is real but **host cycle accuracy is the least relevant component; revisit after the
accelerator model.** Until then `perf`-mode host numbers are labelled approximate; use `both` for host
studies (incl. the no-VPU baseline).

Parked design (user chose it as the direction, implementation deferred): fixed per-op host costs.
* Marker = `COUNTER_OP` (funct 126) read of counter 0 with `rd`, payload in `rs2`
  (`0x9E7F` tag [63:48] | begin/end [47] | op [46:40] | n [39:0]). RTL-safe: COUNTER_OP is handled in
  `Controller.scala` before the reservation station (which asserts on unknown functs,
  `ReservationStation.scala:362`), and neither `CounterFile.scala` nor spike reads `rs2`.
* Markers inside the `mx_host.h` functions (both copies) + the hand-rolled host math in `attn_vpu.c`,
  `attn_flash.c`, `llama_model.c`, `llama_attention.c`, `mx_ladder.c`.
* `both` logs (op, n, instret) per region -> fit `a + b·n` -> generated `gemmini_host_costs.h`;
  `perf` substitutes table cost for the marked regions. Gate: within 2 % of `both`.

## 10. Calibration set: functional vs performant kernels (user, 2026-10-04)

Most VCS runs are **functional validations, not the performant way to drive the hardware** (fences
after every scale load / matmul / drain, ~3 per matmul, nothing overlapped). The most performant
kernels are the **flash-attention VB kernels** (`attn_flash_llama_vb`, VCS Oct 4 on
MxE4M3VpuGemminiRocketConfig: serial 215,530 vs pipelined 94,718 cycles).

How the two are used:
* **Serial/fenced runs calibrate per-stage costs.** Each phase is isolated by fences, so its cycles
  are one unit's latency (mesh, DMA, VPU, requant). `attn_flash_llama_vb`'s serial pass gives its own
  per-stage split (loads 39,758, QK 51,640, softmax 53,759, requant 17,572, PV 39,330, O update 10,542,
  final 2,872).
* **Pipelined runs validate overlap.** These are the cases a simple additive model gets wrong. The
  model must reproduce `attn_flash_llama_vb` / `attn_flash_llama` pipelined (94,718 / 98,808) from the
  same constants, with no per-kernel fitting. They are the headline gate for ≤5 %.

## 11. Research inputs for Step 3 (2026-10-04, three survey agents)

**Command mix (what to model first).** The dominant pattern (P1, `mx_mesh.h:99-125` and copies, both
emitters, ~77/88 ISA tests): `config_ld` + I·Kt mvin A, 2× MX_LOAD_SCALES, **fence**, `config_ld` +
Kt·J mvin B, `config_ex`, `config_st`, CONFIG_SCALE_MEM, BOUNDS/SPAD_AB/LOOP_WS(rs2 0x200|0x38),
**fence**, drain `config_st` + M·N/128 mvout, **fence** (~3 fences per matmul). Others: P2 resident chain
(LOOP_WS 0x400, `_resident` scale config), P3 native DRAM LOOP_WS with loop-managed scales
(`mx_native.h:173-208`), P3b DRAM loop + software scales + `_wait`, P4 VPU/SPAD_REQUANT chains with no
fences (`attn_vpu.c`, `attn_flash.c`). Never used by kernels: preload/compute (4/5/6), loop_conv,
MVOUT_SPAD, SPAD_C, MVIN3, MX_LUT_DISABLE; MX_LOAD_LUT only in ISA tests + `mxgemm_emit`.

**RTL timing facts (MxGemminiRocketConfig, hand-derived ±1–2 cyc, not waveform-verified):**
* Front end: 2-entry queues RoCC router → raw_cmd_q → LoopConv → LoopMatmul → unrolled_cmd → RS
  alloc 1/cycle (~5–6 cyc). Controller-handled (never in RS): FLUSH, COUNTER_OP, CLKGATE,
  MX_LUT_DISABLE, MX_LOAD_SCALES (4-entry start_q), MX_LOAD_LUT (blocks the stream while busy).
* `gemmini_fence()` is the RISC-V `fence`: Rocket stalls until `rocc.busy` drops (RS, spad incl. write
  acks, scale/LUT loaders, VPU, requant flush). Spike's funct 127 is not the RTL mechanism.
* RS: ld 8 / ex 16 / st 4 (VPU cfg: ld 32, vec 16); full queue blocks allocation of all queues (HOL);
  ld/ex in order + pipelined, st/vec serialised (next after previous completes); cross-queue deps =
  address-interval overlap, cleared on completion. CONFIG_LD/ST complete on issue; CONFIG_EX and
  CONFIG_SCALE_MEM wait for no matmul in flight (a mesh drain).
* LoopMatmul: 2 slots; k outer, j, i inner; ≤1 cmd/cycle total; throttles ld ≤8, ex ≤16, st ≤4
  outstanding; LdA = I·⌈Kt/4⌉ mvins, LdB = Kt·⌈J/4⌉ (4 blocks per mvin); StC = 2·I·⌈J/4⌉.
* Load: every Get 64 B, 1 Get/cycle, ≤32 in flight; BeatMerger writes one 16 B row/cycle → **16 B/cycle
  mvin ceiling** (= one mesh block per 16 cycles). DRAM/L2 latency not derivable — calibrate.
  MX_LOAD_SCALES: 8 Gets in flight, retires 8 B/cycle, +1/row. MX_LOAD_LUT: one 8 B Get at a time.
* Mesh: 16 cycles per request in every PE mode, back-to-back with no bubble; first output ≈ +33,
  last ≈ +48, acc commit ≈ +51. Quad modes = more MACs per request, same cycles.
* Store: 1 acc read/cycle; completes when its reads issue (early). Requantizer 1 beat/cycle, 2-cycle
  pipe; scale flush Puts at end. FP8→spad ~64 cyc / 16×64 tile, BF16→spad ~128.
* VPU: 1 row/cycle, +1 on same-bank src2, fixed ~5-cycle latency for all ops, 2 units.
  SPAD_REQUANT ~4 cycles/32-elem block, waits for requant quiet.

**Ground truth (FP8 E4M3, DIM 16, 500 MHz):** ~40 recent (Sep 27–Oct 4) VCS runs in
`sims/vcs/output/<cfg>/` with in-program `rdcycle` phase numbers, FSDB beside each (NPI-readable).
Old `mx_vcs_logs_*` are pass/fail only. Measured: 16 cyc/tile-op (128³ ex 8,244 vs 8,192), ~350–390
fixed per run, DMA A-warm 15.2 B/c / cold 7.5 B/c (latency ~242), scale load cold 171 cyc/512 B,
mvout ~1 row/cycle, VPU 512 rows in 534, SPAD_REQUANT 128 blocks in 611. Gaps: no non-E4M3 timing,
no LUT load on Rocket, no K/M sweeps, base ISA matmuls not re-run since Sep 9; ~20 % DRAM noise seen
early → check run-to-run variance before holding ±5 %.

**Spike can expose modelled time to the program.** `processor_t::decode_insn` searches extension
(custom) instructions **before** base ones (`riscv-isa-sim/riscv/processor.cc:663`). In `perf`/`both`
the extension registers overrides for `fence` (host time = max(host, accelerator idle)) and `rdcycle`
(`read_cycles()` in `gemmini_testutils.h:273`; returns modelled time). Then every kernel's own
`PERF`/`cycles` printout is a modelled number, directly comparable to its VCS log — no new reporting.
`func` mode registers neither, so it stays byte-identical.

## 12. Step 3 as built: event-driven core (user decision 2026-10-05)

**Decision (user): the forward-only (timestamp) model is replaced by an event-driven core.** Forward-only is
exact only for structures arbitrated in program order (RS deps, in-order queues, capacities, throttles, front-end
backpressure, fences). It cannot be exact for priority-arbitrated shared resources, where a later command wins an
earlier cycle: accumulator bank port (mesh writes beat store reads), scratchpad bank ports (VPU > ex > mvin >
requant > SPAD_REQUANT), and the memory system shared by DMA reader / writer / scale / LUT loaders. Those are
exactly the overlap / double-buffering cases (the user's priority). The first forward version showed it: stores
overlapped compute (`st_ex` 47 in RTL) and compute came out −25 %.

**Layout** (`software/libgemmini/perf/`, one class per hardware block, namespace `gperf`):

| file | class | models |
|---|---|---|
| `config.h/.cc` | `config_t` | every parameter, one X-macro line each; presets `mx_rocket`, `e4m3_vpu`; `GEMMINI_PERF_CONFIG`, `GEMMINI_PERF_SET="sec.name=v,..."`, `GEMMINI_PERF_DUMP_CONFIG` |
| `event_queue.h` | `event_queue_t` | the clock: events in cycle order, FIFO within a cycle, cancellable |
| `port.h/.cc` | `port_t` | one-requester-per-cycle resource, priority + preemption, cycles granted only as time reaches them |
| `memory_system.h/.cc` | `memory_system_t` | bus request channel (port), LRU L2 with fill-pending lines, DRAM channel (port) |
| `dma.h/.cc` | `dma_reader_t`, `dma_writer_t` | StreamReader + BeatMerger (Gets per line, in-flight cap, in-order merge into bank write ports), StreamWriter |
| `scratchpad.h`, `accumulator.h` | `scratchpad_t`, `accumulator_t` | per-bank read/write ports with RTL priorities; acc bank port (mesh > store) = the double buffer |
| `reservation_station.h/.cc` | `reservation_station_t` | 4 queues, dependency counts + dependents released on completion, per-queue issue rules |
| `loop_matmul.h/.cc` | `loop_matmul_t` | LdA/LdB/Ex/StC/StCSpad unrollers, state-checked gates and throttles, 2 slots, acc-base alternation |
| `load_unit`, `execute_unit`, `store_unit` | | LoadController, ExecuteController + mesh, StoreController + requant path |
| `scale_loader`, `lut_loader.h` | | MX_LOAD_SCALES / MX_LOAD_LUT loaders (outside the RS) |
| `vpu`, `spad_requant.h` | | VPU units, SPAD_REQUANT (uncalibrated, step 5) |
| `frontend.h`, `host.h` | | RoCC path; host clock = instret × cpi + stalls |
| `model.h/.cc` | `model_t` | decode, the per-cycle command-stream arbiter, host coupling, report, trace |

**Host coupling.** Before each RoCC command, the simulation runs every event before the core's current cycle; a
fence runs it to idle. `fence` and `rdcycle` are taken over in perf/both (extension instructions match before the
base ISA, `riscv-isa-sim/riscv/processor.cc:663`), so a kernel's own printed cycles are the model's.
**Spike bumps `minstret` only at the end of a 5000-instruction batch** (`execute.cc:373`); the model's host clock
was off by up to 5000 cycles inside a batch. Fixed the way spike's CSR reads do it: the RoCC / fence / rdcycle
handlers return `PC_SERIALIZE_BEFORE` once (ends the batch, exact minstret) and run again. The RoCC handler is
wrapped from `gemmini_perf_t::add_instructions`, so `gemmini.cc` is unchanged by this.

**Checks (2026-10-05):** `func` byte-identical to a HEAD build on 8 ELFs (mxl3, llama_mlp_tiny, attn_vpu,
llama_attention_tiny, attn_flash_mid, llama_two_mm, llama_layer_full, mx_bench_matmul_m32). `perf` and `both`
print identical cycles; `both` passes the kernel's functional check. Speed (perf mode): llama_layer_full 2.1 s
(func 65.6 s), llama_model_l2 9.4 s (func 210 s, 54 M events).

**Build fix:** the Makefile now compiles `perf/*.cc` (an in-tree `make` before this produced a `.so` with undefined
`gperf::` symbols that spike could not load; rebuilt). `build_spike.py` copies and compiles `perf/` (`source_files()`), and links `../gemmini-rocc-tests` beside the copied sources -- per-recipe builds had been broken since the VPU commit made `gemmini.cc` include `../gemmini-rocc-tests/include/vpu_ref.h` by relative path.

**First comparison, uncalibrated defaults** (VCS `PERF` line vs model, ISA ELFs `build_mx_rocket`):

| test | load | compute | mvout |
|---|---|---|---|
| fp8_64x64 VCS | 3712 | 1927 | 2837 |
| fp8_64x64 model | 4128 (+11 %) | 1699 (−12 %) | 936 (−67 %) |
| fp8_128x128 VCS | 5110 | 10839 | 5610 |
| fp8_128x128 model | 5320 (+4 %) | 10469 (−3.4 %) | 2753 (−51 %) |

mvout is the write path: the RTL's ~2.7 cycles per 16-byte Put with 32 in flight implies ~90 cycles per Put ack
(model default 40), and the 64×64 case is slower still (cold lines). Calibration is the next step.

**Folder layout (user request, 2026-10-05).** `perf/` is grouped by role; includes are relative to each file:
`params/` (config), `sim/` (event_queue, port, types -- the engine), `control/` (frontend, loop_matmul,
reservation_station), `compute/` (execute_unit, vpu, spad_requant), `lsu/` (load_unit, store_unit, dma,
scale_loader, lut_loader), `memory/` (scratchpad, accumulator, memory_system), and `model.*` + `host.h` at the top.
Makefile globs `perf/*.cc perf/*/*.cc`; `build_spike.py` copies `perf/` recursively. Re-verified: func identical to
HEAD, perf == both cycles unchanged by the move, recipe build OK.

**Final validation (user, 2026-10-05): `attn_flash_llama_vb` is THE validation test and the user's most important
one.** Never used for calibration; the model must get as close to it as possible from constants measured on other
tests. Its error is reported separately, as the headline number.

## 13. Calibration log

### 13.1 Write path (2026-10-05), from `matmul_tiled_fp8_128x128` FSDB + commit trace
Script: `waveform-debug/mxgemmini_debug_scripts/perf_write_ack.py` (Put/ack pairing by TileLink source).
* **Commit trace vs FSDB clock: FSDB cycle = commit-trace cycle + 509** for this run (aligned on the ~14-cycle counter
  reads: log 60425/60438/60452 ↔ FSDB busy pulses 60934/60947/60961). An earlier read of the FSDB with the log's
  cycle numbers mislocated the mvout phase and briefly suggested a 500-cycle store startup and a fence that does not
  wait; both were artifacts of the 509-cycle offset. The fence does wait for `io.busy`.
* The StreamWriter issues **one 64-byte Put (byte-masked) per 64 B line a row touches** — the model's existing rule.
  The VCS binary's `C_hw` was 8 B off a 16 B boundary, so 5 Puts per 4 rows: 2560 Puts for 2048 rows.
* **The L2 accepts a Put every 2nd cycle** (2411 of 2559 issue gaps are 2; `a_ready` low 2675 cycles) →
  new `mem.put_cycles = 2`.
* **Put → ack: median 44** (p10 21, p90 74, max 110; first touch of a line 46 vs re-touch 44: no cold penalty);
  ~25 Puts in flight, under the 32 cap → `mem.write_ack_latency = 44`.
* Phase = 294 setup + 2560 × 2.05 + 66 tail = 5610 (VCS). Setup is mostly counter reads: **a RoCC instruction
  with rd stalls the core ~10 cycles** → new `host.rocc_resp_cycles = 4` (after the command is taken; xd read from
  the instruction in the RoCC wrapper, `gemmini.cc` unchanged).
* **Binary mismatch:** the ISA ELFs were rebuilt 2026-10-03; VCS kept only `.dump`. Current ELF has `C_hw`
  16-aligned → 2048 Puts; VCS-equivalent for it ≈ 294 + 2048 × 2.05 + 66 ≈ 4560.

| 128×128 phase | VCS | model before | model now |
|---|---|---|---|
| load | 5110 | 5320 | 5509 (+7.8 %, load path not calibrated yet) |
| compute | 10839 | 10469 | 10649 (−1.8 %) |
| mvout | 5610 (≈4560 for the current binary) | 2753 | 4318 (−5.3 % vs 4560) |

### 13.2 Load path + native DRAM loop (2026-10-05)
**Exact-binary set** (current ELF disassembly == the VCS `.dump`): `llama_mlp_tiny_native_ua`, `llama_mlp_tiny_db`,
`llama_mlp_small`, `matmul_tiled_fp8_128x128_dramloop`, `mx_mem_bw`; the other `dramloop_*` differ only by a swapped
register pair in host check code (same layout). The others (e.g. `llama_mlp_tiny_native`, `*_attention_*_native`,
`mx_bench_matmul`) were rebuilt after their VCS run.

**Native loop (`gemmini_loop_ws_mx`) now modelled exactly per the RTL:** LOOP_WS_CONFIG_SCALES/STRIDES; LdS
unroller (top priority, gated 2-D MX_LOAD_SCALES A then B into half = slot, A skipped when A = NULL and the slot
holds the same slice; reuse record invalidated by any non-loop command, LoopMatmul.scala:1503-1554); the loop's
managed CONFIG_SCALE_MEM first in Ex (waits drain + scale_ready, claims its halves); StC as 2 chunks per
(i, j-group). Scale loader: four half states FREE/LOADED/INUSE, gated head-of-line blocking, landed/ready per half
(Controller.scala:596-721); raw CONFIG_SCALE_MEM decodes act/wgt half + wait/managed/reuse bits.

**Measured (FSDB via NPI; scripts `waveform-debug/mxgemmini_debug_scripts/perf_tl_latency.py`, `perf_rtl_events.py`):**
* DRAM channel: one 64 B line per 8 cycles. Unloaded DRAM latency is small (L2 DRAM-side median 23 in the MLP
  down loop); the ~200-250 seen in `dramloop` is queueing (32 reads in flight × 8 cycles). `mem.dram_latency` 200 → 20.
* L2 hit 12 (`mx_mem_bw` B_warm_16B); scale-loader client hit ~10, **scale Gets are 8 B** (64 Gets / 512 B), and the
  scale loader is **not on the DMA's crossbar port** (own client; shares L2/DRAM) → `mem.client_hit_latency`,
  `scale.get_bytes`, separate `client_bus` port.
* Writes: PutFull (whole line) accepted every cycle, ack mean 115 (dramloop C); PutPartial every 2nd cycle, ack 46.
* A TileLink client holds a request until the bus accepts it (reader/writer now wait for the grant).
* A store to DRAM is back-pressured by its write queues: released when all but `st.write_slack` (8) Puts are on the
  bus; ~13-cycle command → first Put pipeline (`st.pipe_latency` 12), hidden by the slack in steady state.
* L2: a request to a line whose fill is outstanding takes its own L2 pass after the fill.

**Replay mode** (`tools/rtl_replay.py` + `GEMMINI_PERF_REPLAY`): the VCS commit trace gives the retire cycle of every
Gemmini command / fence / rdcycle; the model uses those as arrival times (no host model in the comparison) and
reports, per fence that follows Gemmini work, RTL retire vs model idle. Fences are matched by the number of Gemmini
commands before them (spike and the RTL run differ in boot/printf fences). The FSDB cycle = commit-trace cycle + ~508.

**Found, not yet modelled — the CPU's caches.** (1) Data the CPU just wrote (e.g. `h_codes` before the MLP down loop)
is warm in the L2 / owned by the L1: RTL reads hit, the model's lines are cold (MLP down loop +191 cycles of 1255).
(2) Writes to lines the L1 owns are slower (`mx_mem_bw` mvout: `out_buf` is CPU-zeroed .bss; model −27 %).
(3) The CPU's own I/D-cache misses (45-50 cycles each) — hidden by replay mode. Needs spike's memtracer
(observe host stores) — a speed cost, decision pending with the user.

| test (perf mode, exact binary) | VCS | model | |
|---|---|---|---|
| dramloop compute | 11899 | 11560 | −2.8 % |
| dramloop_nc / nc4 / nc_2d loops | 10655 / 10727 / 10630 | 10415 / 10550 / 10415 | −2.3 / −1.7 / −2.0 % |
| dramloop_kt loops | 13254 | 12592 | −5.0 % |
| dramloop_ls / ls4 / nc_wait loops | 10649 / 10900 / 10536 | 13334 / 11550 / 11084 | **+25.2** / +6.0 / +5.2 % (loop-managed scales: open) |
| 128x128 mvout (≈4560 for the current binary) | 5610 | 4331 | ≈ −5 % |
| mx_mem_bw A cold / A warm / B warm | 2179 / 1076 / 1080 | 2105 / 1069 / 1069 | −3.4 / −0.7 / −1.0 % |
| mx_mem_bw B cold 16 B / scale cold / scale warm / mvout | 3373 / 171 / 85 / 2917 | 2735 / 260 / 113 / 2128 | −19 / +52 / +33 / −27 % |
| MLP native (replay): G,U loops end / Y loop end | — | +89 / +191 cycles | of 2088 / 1255 |

### 13.3 CPU-store tracking + scale loader + regression tool (2026-10-05)
**Decision (user): host-memory tracking ON by default, with a switch** (`GEMMINI_PERF_SET="mem.host_tracking=0"`).
Spike's memtracer, registered from the extension's `reset(processor_t&)` (before boot code runs; `gemmini.h`
2-line override), traces **stores only** — loads/fetches keep spike's fast TLB path. Stores issued while a
Gemmini command executes (the functional model's own writes in `both`) are ignored. Model: the CPU's L1
(`mem.host_l1_kib` 16, LRU) owns lines it wrote; the inclusive L2 holds them (dirty). A Gemmini access to an
L1-owned line pays `mem.probe_cycles` / `mem.probe_bus_cycles` (placeholders, uncalibrated) and takes it out of the L1.
**Cost: llama_layer_full 2.10 → 2.38 s (+13 %), llama_model_l2 8.21 → 8.39 s (+2 %).**
Effect (MLP native, replay): down loop +191 → −9 cycles; G,U loops +89 → −239 (now early: open).

**Other fixes in this round (all from FSDB evidence):**
* Scale loader: Gets are line-sized (largest aligned ≤ 64 B): 8 requests for mx_mem_bw's 512 B (FSDB: valid high
  8 cycles). The test's "reqs=64" counts 8-byte words; an earlier reading of it as 64 Gets was wrong.
  → `dramloop_ls` +25 % → −1 %, `ls4` +6 → +0.5 %, `nc_wait` +5 → −1 %.
* The fence waits for Gemmini's busy signal (RS completions, Put acks, scale/LUT loads, requant spad writes), not
  for L2/DRAM background work: `model_t::busy_until()`.
* L2: dirty lines (Gemmini or CPU writes) cost a DRAM slot when evicted; a partial write to an absent line costs a
  DRAM fill slot in the background (the Put is not delayed: first-touch ack 46 vs 44).
* Store backpressure counts Puts (`st.write_slack` = 8 Puts still waiting for the bus), not cycles.

**Regression tool:** `python3 tools/perf_regress.py [--set ...] [--only ...]` — every exact-binary test, model vs
VCS per phase, plus MLP replay fences; whole suite ~1.5 s.

Current (perf mode): dramloop compute −2.8 %; nc/nc4/nc_2d/ls/ls4/nc_wait loops −2.3/−1.7/−2.0/−1.0/+0.5/−1.1 %;
kt −7.6 %; 128x128 compute −1.7 %, mvout ≈ −5 % vs the binary-adjusted VCS; mx_mem_bw A cold/warm, B warm,
scale warm within 3.4 %; scale cold −38 %, B cold 16 B −19 %, mvout_16B −27 % (open). Phases dominated by
host instructions (e.g. 55-cycle scale phases, the fenced MLP tests) need the host model or replay.

### 13.4 LoopMatmul load arbitration (2026-10-05)
* **A/B arbiter (WeightedArbiter static weight, LoopMatmul.scala:1170-1184):** B's very first load goes first
  (`inB_k == 0 && inB_j == 0`), then B whenever A's k is ahead, else A. When the A and B unrollers serve different
  loops, the one on the head loop is **forced** — and an idle unroller keeps the id of the loop it served last.
  So in `dramloop_kt` (4 K-tile loops, all A in half 0) loop 2's B loads waited ~2100 cycles: A pointed at the
  head loop (forced), and loop 2's A was `ld_blocked` (newer loop's A rows overlap the head's until the head has
  issued all its computes, :1387). An A = NULL loop still "starts" its LdA unroller with zero commands, so the
  unroller passes through it (`unroller_loop()`). dramloop_kt −7.6 % → −1.6 %; replay −1161 → −157 cycles.
* **Lone preload** (`mesh.lone_preload_cycles` 5): new weights reaching an idle mesh are their own request.
  Real (dramloop FSDB: requests 5 cycles apart) but small.
* **Open — end-of-loop mesh stalls:** in the plain dramloop (replay −260 = −2.2 %) the RTL mesh pauses 128 / 89 / 89
  cycles once the C stores start (~cycle 9500 of the phase); the model's mesh runs on. A store/mesh interaction on
  the accumulator not yet modelled. MLP G,U loops: replay −240 (store timing: RTL first Put 1330 vs model 991).

### 13.5 Store/mesh interaction: the RS's coarse accumulator ranges (2026-10-05)
**Found (dramloop FSDB):** at the end of the loop the mesh idles ~110 cycles at a time while the C stores run.
LoopMatmul emits only stores; `ex_utilization` is pinned at its limit 16 while the ExecuteController's queue is empty
and the accumulator is not written: 16 ex commands sit in the RS waiting on the stores. Cause: the RS's address
ranges are coarser than the data — a preload's C range is `c_rows` (DIM) rows and an mvout's is
`(blocks-1)*DIM + rows` (ReservationStation.scala:250-299), but a single-throughput C tile occupies DIM/4
accumulator rows. So a store's range reaches into the next row-block's tiles, and the preloads writing them wait
(WAR) for the store. The model now uses the RTL's ranges, not the exact ones.

Effect: 128x128 compute −1.6 → −0.3 %; dramloop_nc / nc_2d / ls / nc_wait loops → −1.9 / −1.6 / −0.6 / −0.7 %;
MLP native replay G,U −240 → −147 cycles, Y +6 → +1; mlp_tiny_db G,U phase −26 → −13 %. Plain dramloop: the
stalls now appear (last tile 9977 vs RTL 9926); replay −2.6 %, the rest is the store tail.

**Open (small): L2 write throughput.** RTL Put spacing at the end of dramloop is ~7 cycles (acks slow down as the L2
write path fills: p90 242 vs mean 115); the model's fixed ack latency gives bursts. Worth ~1-2 % here.

### 13.6 VPU config (e4m3_vpu preset, 2026-10-05) — in progress
Built: RS vec queue out of order with the RTL deps (row conflicts, shared read banks, SR after SR, SR after a
quantized store); VPU timing (EXPSUM dst2 + 1 bubble/row, latency 5, unit busy until drained, 2 units);
LoopMatmul vec_bypass (`vec_pass` / `cfg_pass` / `ld_pass`, `model.cc` `may_pass`). `check_stuck()` prints a
deadlock diagnostic at fence/finish.

Bug found: a CONFIG_SCALE_MEM that bypassed the loops bumped the loops' ex-outstanding count (`note_passed_ex`,
RTL :1440-1446) but its RS entry was untagged, so it never gave the slot back. After 16 passes the ex throttle
closed forever -> SPAD_REQUANT stuck at the front-end head, model ran 16/32 loops of `attn_flash_llama_2h`
(and reported an impossible 130% util). Fix: tag the passed command `LC_EX` (`passing_` flag in `model_t`).

Exact-binary results after the fix (perf mode, model vs VCS):
| test | phase | VCS | model | err |
|---|---|---|---|---|
| attn_flash_llama_2h | pipelined | 184258 | 183736 | -0.3% |
| attn_flash_llama_2h_fused | pipelined | 156357 | 156677 | +0.2% |
| attn_flash_llama_vb_fused | pipelined | 82341 | 82261 | -0.1% |
| chain_pipelined | fenced / program / pipelined | 6470 / 5851 / 5285 | 6311 / 5027 / 4985 | -2.5 / -14.1 / -5.7% |
| attn_vpu_fa | attn_vpu | 47510 | 43252 | -9.0% |
| dramloop_relu | loops / total | 10793 / 11186 | 10586 / 10801 | -1.9 / -3.4% |
Open: chain_pipelined `program` and attn_vpu_fa (both short, host-heavy — check host share before tuning);
dramloop_relu `scales` 2 vs 55 (cold scale fetch, same open item as mx_mem_bw scale_cold).
Validation (`attn_flash_llama_vb`): no matching binary yet — the ELF was rebuilt after the VCS run.

#### 13.6.1 Host vs Gemmini share (replay), and a bus-width finding (2026-10-05)
* Replay (RTL commit-trace host timing): **attn_vpu_fa** Gemmini idles within +452 cycles of the RTL (~1%) — its
  -9% is the deferred host model. **chain_pipelined**: a constant ~+222 per fenced phase, from `check()`'s 4 KB of
  16 B-row spad mvouts (256 PutPartials).
* **The FSDBs come from two system-bus builds.** `perf_bus_width.py` (spad.widget present = a TLWidthWidget splits
  Gemmini's 64 B DMA beats for a narrower bus): every MxGemminiRocketConfig FSDB up to `llama_mlp_tiny` (09-28 11:19),
  incl. **`matmul_tiled_fp8_128x128` (the 13.1 write-path calibration)**, has it; from `mx_mem_bw` (09-28 14:27) on
  (all dramloops, MLP native/db/small, attention, mx_bench) and the VPU config: none (512-bit, current RTL).
* So 13.1's "L2 takes a Put every 2nd cycle" (`mem.put_cycles = 2`) is the old build's width adapter (two 32 B beats
  per Put; `a_ready` at `spad.auto_id_out` is always 1). On the 512-bit build the bus accepts a PutPartial every cycle:
  - chain_pipelined check (L2-warm 4 KB buffer): 256 Puts in 306 cycles, a_ready never low, ack mean 30.
  - mx_mem_bw mvout_16B (cold lines): 1024 Puts at 2.7 cyc/Put, a_ready never low, ack ~85 → bound by the 32 in
    flight / ack latency, not by the bus.
  Scripts: `perf_put_line_gap.py`, `perf_bus_width.py`, `perf_put_bursts.py` (waveform-debug/mxgemmini_debug_scripts).

#### 13.6.2 Write path re-fit on the 512-bit build (user-approved step, 2026-10-05)
Done:
1. `mem.bus_bytes` (64; 32 = the pre-09-28 MX build) replaces `mem.put_cycles` / `mem.full_put_cycles`: a Put is
   always a line-sized TileLink message (DMA.scala:426-430: with 64 B beats the only write size is 64 B, PutFull iff
   the mask covers the line), so it takes line / bus_bytes bus cycles.
2. Partial Put ack: `mem.write_ack_latency` 46 → 29 (chain_pipelined, L2-warm). A partial Put to a line the L2 lacks
   now waits for the line's fill (as a read miss does). New `mem.write_fill_cycles = 11`: the L2 starts a
   partial-write fill every 11.0 cycles (mx_mem_bw mvout_16B FSDB, L2 DRAM side: 257 AcquireBlocks, 2.8 outstanding,
   27-33 cycles each) vs 8 for a read miss.
3. `perf_regress.py`: the 128x128 row runs with the old build's bus (`mem.bus_bytes=32, mem.write_ack_latency=46`).
4. Bug found on the way: **StC chunk geometry**. RTL (LoopMatmul.scala:573-577, 714-720): a chunk is mbl/2 j tiles
   (32 BF16 cols = whole 64 B lines), and a j group of gb tiles stores gb·numChunks/4 chunks (≥1). The model always
   cut a group into 2 halves, so a 2-tile group (dramloop_nc4 / ls4: J = 2 per loop) became 2 chunks of 32 B partial
   Puts (1024 Puts + 512 write fills) where the RTL sends 512 whole-line PutFulls. Fixed (`stc_tiles/stc_chunks/
   stc_per_i` in loop_matmul.h).

| test / phase | VCS | before | after |
|---|---|---|---|
| mx_mem_bw mvout_16B | 2917 | 2132 (-26.9%) | 2900 (-0.6%) |
| chain_pipelined check() fences (replay) | — | +222 each | +54 each (+0.6%) |
| 128x128 total (old-bus row) | 21559 | — | 21494 (-0.3%) |
| dramloop_nc4 loops | 10727 | +2.2% | 11298 (+5.3%) |
| dramloop_ls4 loops | 10900 | +0.5% | 11477 (+5.3%) |
| attn_flash_llama_2h / 2h_fused / vb_fused | 184258 / 156357 / 82341 | -0.3 / +0.2 / -0.1% | -0.3 / +0.2 / -0.1% |
Other rows unchanged.

nc4 / ls4 were right for the wrong reasons before (wrong chunking hid behind cheap partial Puts). What is left:
the **PutFull ack is a fixed 115** (dramloop mean), but in nc4 the RTL's PutFull acks are bimodal (bursts: median
12-62, mean 81-152), and with 32 Puts in flight a flat 115 makes the store too slow. Candidate mechanism (not yet
verified): the L2's MSHRs are shared by read misses, write fills and PutFull misses, so a PutFull acks fast when
the L2 is idle and waits behind load fills otherwise; it would also explain the read-miss (~4 outstanding) and
write-fill (~2.8 outstanding) rates as one limit instead of two fitted slot lengths.

#### 13.6.3 L2 / probe model (user-approved "MSHR" step, 2026-10-06)
Parameters are now tagged in `config.h`: **[knob]** = a hardware parameter (bus width, L1 geometry/replacement,
...), **[measured]** = a latency/rate read off an FSDB for behaviour not modelled structurally, **[approximation]**.
Findings (scripts: `perf_line_timeline.py` [SLOW=, PROBESTATS=], `perf_put_bursts.py`, `perf_tl_latency.py`):
* **MSHR hypothesis rejected**: the L2 has 42 MSHRs (`ListBuffer_QueuedRequest_q126_e42`; 2 + memCycles 40 /
  blockBeats 1) -- never the limit. Cold reads run 31 fills deep at the DRAM side (DRAM-bandwidth-bound, 8 cyc/line:
  the model's existing mechanism is right). A "shared L2 data array" (fill 8 + 1/access) was also tried and rejected
  (A_cold_64B +11%).
* **Same-line Puts serialize**: warm line (chain FSDB) acks +22/+30/+38/+46 -> `l2_put_serial_cycles 8`,
  `write_ack_latency 22`. After a write-allocate fill the spacing is ~16 (noisy) -> `l2_put_serial_filled_cycles`
  [measured, mechanism unexplained].
* **The slow PutFull acks are L1 probes** (dramloop FSDB): 305 of 512 C lines are probed out of the CPU's L1
  (ProbeAckData: the CPU memset C_hw), and all 295 slow acks are probed lines. Probes issue one per 7 cycles
  (292/304 gaps), probe -> ProbeAckData 19 (301/301), -> Put ack +7; unprobed PutFull acks in 10. New params
  `probe_issue_latency 5, probe_cycles 7 (serial L1 port), probe_latency 19, probe_put_ack_latency 7,
  full_write_ack_latency 10` (was 115 = the probe-inflated mean); placeholders `probe_cycles 20 / probe_bus 2` gone.
* **CPU L1**: WithNHugeCores = 64 sets x 8 ways (32 KB, was 16 KB fully-assoc), random replacement
  (`host_l1_ways`, `host_l1_random`). Approximation per user ("the cache may be approximate"): the L1 starts full of
  unseen lines (`host_l1_start_full`) -> dramloop 331 probes vs RTL 305. CPU-load tracking was tried (no effect on
  this, removed).
* **Bug fixed: the CPU-store tracer attached on the first RoCC instruction** (Spike registers `--extension` after
  the processor reset, so `extension_t::reset(processor_t&)` never reached us) and missed every store before it
  (e.g. dramloop's `memset(C_hw)`). Now attached from `get_instructions(processor)` via
  `gemmini_perf_t::on_register` (one line in gemmini.cc). func/both outputs unchanged (chain_pipelined, dramloop).

| test / phase | VCS | 13.6.2 | now |
|---|---|---|---|
| dramloop compute | 11899 | -4.2% | -0.4% |
| dramloop_nc / kt / nc_2d / ls / nc_wait loops | | -1.9 / -2.2 / -1.6 / -0.6 / -0.7% | +0.0 / -1.8 / +0.2 / +1.2 / +1.2% |
| dramloop_nc4 / ls4 loops | 10727 / 10900 | +5.3 / +5.3% | +5.8 / +5.8% |
| mx_mem_bw mvout_16B | 2917 | -0.6% (fitted write_fill_cycles) | -26.0% (open: cold partial-write latency) |
| llama_mlp_tiny_native_ua fence 1 / 2 (replay) | | -147 / +1 cyc | -258 / -109 cyc |
| attn_flash_llama_2h / 2h_fused / vb_fused | | -0.3 / +0.2 / -0.1% | -0.8 / -0.5 / -0.7% |
Open: nc4/ls4 +5.8% (store side, 4 small loops), cold partial-write path (mvout_16B; RTL's first ack after a
fill is later under load, 45-91), scale_cold / B_cold_16B (unchanged).

### 13.7 Fixing the remaining kernels (user: "lets try to fix these kernels", 2026-10-06)
* **chain_pipelined**: replay (RTL host timing) puts Gemmini's end of `program` at −4 and `pipelined` at −7 cycles,
  `fenced` +125 (+1.9 %). The perf-mode −14.1 / −5.7 % is the deferred host model. Individual fenced stages are
  ±100 cycles (requant +7 %, the small loop_ws_spad matmul +14..29 %).
* **dramloop_nc4 / ls4 (+5.8 % → −1.7 / −1.6 %)** — RTL-exact loop-store RS range. Per-loop mesh: RTL 16 cyc/tile
  steady, model 4 bubbles of 82 at tiles 120-123 of every loop (store ↔ preload ping-pong). RTL RS stream
  (`perf_rs_trace.py`) shows the same packed acc layout (preload C = i·16 + j·4) and the store allocated before the
  next preload, yet no stall: LoopMatmulStC (`LoopMatmul.scala:602-611`) gives every chunk of (i, j-group) the group's
  acc address and packed cols = tiles·DIM/4 ≤ DIM (one mat), so the RS range is 16 rows from the group base — tile
  row i only. The model used the chunk's tile address and (blocks−1)·DIM + rows = 32 rows (into row i+1). Fixed:
  `loop_cmd_t::rs_span` for L_STC; raw mvouts keep the rs2-derived range. The 13.5 store/mesh stalls come from the
  preloads' DIM-row C range and are unaffected (dramloop compute −0.1 %).
* **`mem.l2_fill_secondary_cycles` (16, approximation)** replaces `l2_put_serial_filled_cycles`: requests queued on
  a fill (Gets and partial Puts) resume one per 16 cycles after it lands. B_cold_16B −18.9 → −13.2 %, mvout_16B
  −26 %, nothing else moves. Not tuned further (user: cache may be approximate).
* **MLP native_ua loops G,U**: perf-mode phase 1542 vs RTL 2088 (−26 %), Y 991 vs 1255 (−21 %); replay: Gemmini
  idle 405 cycles early. The binary matches the VCS .dump. Ordinal Get comparison: the model pulls ahead at Gets
  #48-64 and #112-144 (RTL 6 cyc/Get there). There the L2's DRAM side shows latency ~100 (min 97) with only ~6
  outstanding and D beats at 100-156 per 200 cycles (no write-backs on C) — elsewhere it saturates at 1 beat/cycle
  with latency 20-30. **The RTL's memory is SimDRAM + DRAMSim2 (DDR3: banks, open rows, refresh;
  `mm_dramsim2.cc`)**: latency depends on the address pattern; the model's DRAM is a fixed pipe (8 cyc/line + 20).
  Scripts: `perf_l2_outer.py`, `perf_rs_trace.py`, `perf_line_timeline.py` (now with Gets).

### 13.8 Optional open-row DRAM model (user: "additional, not as important", 2026-10-06)
`mem.dram_model = 1` adds DRAMSim2-like banks (testchipip `dramsim2_ini`: DDR3-1333, 8 banks, open page, scheme2 →
bank = line % 8, row = line / 2048; at 500 MHz tRP+tRCD = 15 cycles, tRC = 26). Off by default: it moves every
regression row by < 1 % and MLP G,U only 1558 → 1573 (RTL 2088), so the MLP gap is **not** row misses. Not pursued
further (refresh, scheduler reordering not modelled). The MLP G,U / Y gap (replay: Gemmini idle ~375-400 cycles
early, ~18 % of the 2088-cycle phase) stays open: the RTL's L2→DRAM latency is ~100 with ~6 outstanding in two
stretches of the G,U loads; cause not identified.

### 13.9 llama_layer_e2e vs FireSim, and a projection reproducer (2026-10-06)
FireSim build (firesim-cy, recipe `alveo_u250_firesim_mx_gemmini_rocket_singlecore_no_nic`): MxGemminiFiresimRocketConfig
= MxE4M3VpuGemminiRocketConfig with a **local 256-bit system bus**, every clock 1 GHz (one domain), FASED LatencyPipe
(read latency 30, maxReads 16), no LLC, 64-bit memory bus. New preset **`firesim`** (= e4m3_vpu + bus_bytes 32,
dram_latency 30, new knob `mem.dram_max_reads` 16). It moves the e2e phases < 4 %, so FireSim's memory is not the
cause. Model vs FireSim (firesim preset): attention −1.9 %, projections +7.7..+13.2 % (model mesh util 84-90 % vs
FireSim 94-99 %; demand ~4-6 B/c, not memory-bound), VPU/SR phases −19..−24 %, rope −36 % (CPU), layer +3.5 %.
Sensitivity (not adopted): host.cpi 2 fixes rope only; dram 16 B/c fixes projections, latency 60 the VPU phases.

**Reproducer for the projection gap**: `baremetal/mxgemmini/src/mx_bench_matmul_m64_proj.c` (mx_bench_matmul with
BENCH_M 64, K 1024, N 512, BENCH_SKIP_STREAM, BENCH_ONE_PASS; K/N/skip macros added to mx_bench_matmul.c, defaults
unchanged; Makefile entries added). Same mxn_matmul tile as the e2e projections (64 x 256 x 256 loops, 8 chained).
Model: 141,653 cycles (util 92.5 %), ideal 131,072. Waiting on a VCS run with waveform on MxE4M3VpuGemminiRocketConfig.

### 13.10 Projection reproducer on VCS → two LoopMatmul fixes (2026-10-06)
`mx_bench_matmul_m64_proj` on MxE4M3VpuGemminiRocketConfig: **135,370 cycles (96.8 %)**, checksum = Spike; ELF = .dump.
Model before: 141,653 (+4.6 %), replay +6773. RTL mesh: 16,404 cyc / 1024-tile loop with one 52-cycle bubble at each
boundary; model: 500-1500-cycle stalls at boundaries and one 1522-cycle stall at tile 6 of loop 0.
1. **Load-unroller loop id** (`unroller_loop`): an unroller takes the next loop's request as soon as it is idle
   (io.req.fire sets loop_id), unless that request is withheld by `ld_blocked` (`ldA/ldB.io.req.valid`,
   LoopMatmul.scala:1475-1476 / 1496-1497); the arbiter forces only when the A and B unrollers are on different loops
   (:1173-1176). The model updated the id only on emitting, so both unrollers stayed on the finished head loop and
   blocked each other's first load of the next loop until the head retired: the next loop's loads serialized behind
   the head's compute (8.5k idle DMA cycles per boundary). dramloop_kt: its loop1→2 stall is now 491 vs RTL ~557
   (old rule 1159 — the old −1.7 % was that overshoot cancelling a cold-start shortfall; kt is now −6.1 %, the rest
   being the cold first loads/scales: −286 at start-up, −159 at loop 0→1).
2. **Managed CONFIG_SCALE_MEM is the Ex unroller's first command** (LoopMatmulExecute: req → state cfg, :505), so a
   loop's scale config cannot leave before every older loop's computes have: the model let loop 1's config enter
   the in-order ex queue early, where it waited for its scales to land with loop 0's computes stuck behind it.
Result: m64_proj **133,879 (−1.1 %)**, replay −1001; model boundary bubbles 61 vs RTL 52. Regression otherwise
unchanged (dramloop_ls −0.0 %, ls4 −1.1 %), flash kernels −0.8 / −0.5 / −0.7 %.
Script: `waveform-debug/mxgemmini_debug_scripts/perf_rtl_events.py` (whole-matmul window), ordinal comparison.

### 13.11 CPU L1 caches from spike's cache simulator (user-approved, optional; 2026-10-06)
`mem.host_tracking = 2` (default stays 1): `perf/host/host_cache.h` subclasses spike's installed `cache_sim_t`
(libriscv.so; LFSR random replacement like Rocket) for the CPU's L1 D$ and (host.icache) I$, geometry
`host.l1d_sets/ways`, `host.l1i_sets/ways` (spike --dc/--ic S:W:B terms; WithNHugeCores 64 x 8 x 64 B). The memtracer
also forwards loads (and fetches) at level 2. Each L1 miss stalls the CPU clock (blocking D$, nMSHRs = 0):
`host.l1_miss_l2_hit` 10 / `host.l1_miss_dram` 43 cycles — **measured** from the VCS commit traces with the new
`tools/commit_stalls.py` (load/store retire gaps cluster at 10-12 and 43-45 in chain_pipelined, attn_vpu_fa,
llama_mlp_tiny_db). Probes now come from the simulated L1 (Put → probe to N; Get → probe only a dirty line, to B);
level 1 keeps the approximate store-only L1. No spike source change; spike quirk handled: a FETCH trace passes the
end address, not a length. Speed: small kernels unchanged; llama_layer_e2e 3.4 → 7.0 s.
Effect: llama_mlp_tiny_db ld_scales+Xn+Wg 605 → 799 (VCS 884), G,U 2121 → 2164 (2438), loop Y 1161 → 1204 (1151);
e2e rope 412k → 584k (FireSim, old binary, 641k). chain_pipelined program / attn_vpu_fa barely move (5037 / 42719 vs
VCS 5851 / 47510): their CPU time is not cache misses.
What the commit traces show is left for the CPU model (next, not done): **RoCC issue** — chain_pipelined's 612
commands retire 7.3 cycles apart on average, a cluster at 12 when not back to back (attn_vpu_fa: mean 100, mostly
Gemmini backpressure, which the model has); **FP latency** — llama_mlp_tiny_db FP ops +2.48 cyc each (82k gaps of
exactly 4 = dependent FP ops, 25-26 = fdiv/fsqrt), ~430k cycles; **branches** +0.44..0.63 per branch.

### 13.12 CPU pipeline model (user-approved "RoCC / FP" step, 2026-10-06)
Measured first (VCS commit traces, new `tools/commit_latency.py` + `tools/commit_stalls.py`):
* **RoCC issue is ~free**: back-to-back commands retire at (instructions between) + 0..1 cycles; longer gaps are
  Gemmini backpressure, which the model has. So the e2e VPU/SR phases' −20 % is not CPU time (it is the cold small
  mvin / 16 B mvout path, = mx_mem_bw mvout_16B / B_cold_16B).
* **Producer → dependent latency** (consumer = next instruction): int load 2, FP load 4, fadd/fmul/fma 4, fmv/fcvt
  3-4, fsqrt 27-28, int div 66-67 (5-12 early-out). **Branches** (extra cycles after): taken 0.67-0.74, not taken
  1.05-1.74 (0 or a 3-cycle mispredict), jal 0.4-1.0, jalr 0.25-0.56.
Built: `perf/host/host_core.{h,cc}` — RV64GC decoder (per PC, cached) + register scoreboard + unpipelined fdiv/div
units + branch penalties by outcome (exact: the next fetched PC), fed by the level-2 fetch trace; instruction bits
read once per PC through the core's public MMU `load<uint16_t>` (spike unchanged). Params `host.lat_*`, `host.br_*`,
`host.jal/jalr`, `host.core_model` (lat_fdiv, lat_mul [assumed]). Enabled with mem.host_tracking 2 (+ host.icache).
Result (CPU-side compute time the kernels print): llama_mlp_tiny_db 1,483,681 VCS: level 1 −49 % → level 2 −15 %;
llama_mlp_small 3,224,669: −44 % → −7.9 %. Defaults (level 1) unchanged; regression unchanged.
Open: the remaining −8..−15 % (BHT/BTB, store buffer, mul latency not measured); chain_pipelined program /
attn_vpu_fa regions are command-issue bound, unchanged.

### 13.13 Element-wise reproducer for the e2e VPU/SR phases (2026-10-06)
`baremetal/mxgemmini/src/llama_e2e_elemwise.c`: includes llama_layer_e2e.c with its `main` renamed (the user's file
is not edited) and runs its `rms_phase` (VPU + SPAD_REQUANT) and `residual_phase` (VPU) on the layer's blob
(Makefile: KERNELS, BLOB_KERNELS, blob_llama_e2e_elemwise := llama_layer_e2e, deps). Spike: rmsnorm1 hash xn1
54258df321e5caae (= the FireSim full layer's), xn1_sr e9f0db92c5a5b802, hout 0a5aa1858366b822.
Model (perf): rmsnorm1 81,444 (e4m3_vpu) / 84,320 (firesim); residual 108,007 / 108,855 (tracking 1), 125.5k at
tracking 2. (The full-layer model's 109k rmsnorm1 was on the 15:06 ELF, older than the current rms_phase source
(16:47); the user rebuilt llama_layer_e2e at 17:20.)
Waiting on: VCS with waveform (MxE4M3VpuGemminiRocketConfig) and ideally FireSim of this ELF.
VCS result (2026-10-06): rmsnorm1 129,344, residual 112,078; hashes = Spike. Model 104,071 / 88,136 (−19.5 / −21.4 %).
* Replay (RTL host timing): −0.4 / +3.0 %. But the CPU spends 210k of 241k cycles waiting at RoCC (Gemmini's RS
  is full 99 % of the time: head command a VPU op 43 %, mvout 32 %, mvin 22 % — `perf_rs_block.py`), so Gemmini is
  the bound and replay masks per-command speed. Per command (`perf_rs_cmds.py`, RTL vs model perf trace): **VPU
  exact** (median 263 both), SPAD_REQUANT −4.5 %, **mvin 120 vs 97 cyc/cmd, mvout 199 vs 158**.
* Cause: **Get latency** (latency-bound: ~8 in flight). RTL mean 87.5 vs model 26.8. A cold Get (74 lines):
  Get → L2 AcquireBlock 6, → last DRAM beat 41, → data 7 (first touch 56); the 3 other Gets of the line (one 64 B
  Get per 16 B row) are served +31, +12, +8 after the first. Added [measured] `mem.l2_miss_detect` 6,
  `mem.l2_fill_to_data` 7, `mem.l2_fill_secondary_first` 38 / `_next` 10 (replaces l2_fill_secondary_cycles).
  Model now 107,499 / 90,242 (−16.9 / −19.5 %), mean Get latency 30.1. Remaining: the RTL L2 re-fetches more
  (~6.3k read fills vs model 4,160: set-assoc random 512 KB L2 under ~640 KB of traffic vs the model's fully
  associative LRU). Cold microbenchmarks improved: B_cold_16B −13.2 → −5.6 %, mvout_16B −26 → −15.8 %,
  scale_cold −38 → −30 %; dramloop / m64_proj unchanged; exact flash attn_flash_llama7b_fused +2.3 % (unchanged).
* **Binary drift**: attn_flash_llama_2h (ELF 10-05 22:27), _vb_fused and _2h_fused (ELF 10-06 17:53) no longer match
  their VCS runs. `tools/perf_regress.py` now includes the VPU-config tests and checks each ELF against its VCS
  .dump (objdump -D), reporting "binary differs" instead of an error figure.
* **Concurrent edit (not this session)**: perf/control/loop_matmul.{cc,h} changed at 18:14 after the user's commit
  92fae92 (StCSpad chunk geometry, sts_per_i / sts_chunks, terminal ex_ahead). It moves chain_pipelined (fenced
  6311 → 6021). Left untouched pending the user.

### 13.14 Handoff `perf_model_handoff_spad_store.md` (2026-10-06/07)
1. Kept the handoff's L_STSPAD change (LoopMatmulStCSpad geometry + group RS range) and the `rs.packed_exact` removal.
2. **Pending-bank gate implemented** (`rs.vec_pending_gate`, default 1): `store_unit_t` counts, per scratchpad bank,
   acc/spad -> scratchpad stores from start to their last write-port beat (`pending_banks()`); a vector entry
   (`rs_cmd_t::sp_banks` = banks of a/b/c/d) may not issue while they intersect (ReservationStation.scala:574-578);
   Q_VEC is re-kicked when a count drops. Effect small: +0.1..0.8k cycles.
3. **The handoff's diagnosis did not hold**: inside the RTL's 727-cycle mesh stalls (2h_fused FSDB, 177806+) the
   pending-bank mask is 0 (`perf_stall_probe.py`); the stall is a dependency chain ending in a K/V block's mvins.
   Per-command RTL vs model (`perf_rs_cmds.py` over the pass): **loop stores to the scratchpad (f23) take 67 cycles in
   the RTL (median, p90 79) vs 19 in the model**: acc reads for requant_to_spad are accepted 1 in 4 cycles (a read
   waits for the previous result's two bank-row writes: dma_resp_ready, Scratchpad.scala:1127-1131; 24 reads, ready
   1 in 4, mesh idle; `perf_acc_reads.py`). New [measured] `st.spad_read_interval` 4 -> model store median 67.
   7b_fused RTL stores: median 63 too.
Exact binaries (VCS / model): 2h_fused (handoff copy, -DATTN_STATS_OUT=1) 156,475 / 159,441 (+1.9 %);
attn_flash_llama7b_fused 144,008 / 156,308 (+8.5 %, was +2.3 %); chain_pipelined fenced +0.7 / program −9.5 /
pipelined −0.6 % (were −6.9 / −19.7 / −11.9 %); dramloop / mx_mem_bw / elemwise unchanged.
Open: with store durations now right, both flash totals overshoot by ~3k (2h_fused) / ~12k (7b_fused): the RTL
hides part of the store time where the model does not. In 7b_fused the model's mvins (median 1408 vs RTL 1024)
and computes (79 vs 64) are slower — next: which dependency serializes loads/computes behind the stores.

#### 13.14.1 attn_flash_llama7b_fused comparison (2026-10-07)
* Per 512-tile key block: model +1,026 cycles, from two clusters of mesh gaps (277 + 141 + 134 + 61) at tiles ~115
  and ~371; RTL one 52-cycle gap there. Model: preloads wait for in-flight loop stores (DBG: preload C [0x114,0x124)
  vs store source [0x120,0x130) of the next row group). RTL: no wait.
* Cause: **RTL fix `b0dd6cb` (2026-10-05)** — with packed MX acc (`io.mx_packed_acc`), a preload's RS range starts at
  its DIM-row group (`ReservationStation.scala:302-307`): "a range that starts mid-group would claim the next
  group's rows: false WAR on its stores". New knob `rs.packed_preload_align` (`model_t::preload_c_span`, loop + raw
  preloads): 0 = older RTL (the MxGemminiRocketConfig sim of 09-29, where 13.5's spill stalls are real), 1 in the
  e4m3_vpu / firesim presets (VPU sim rebuilt 10-06 11:41; FireSim build of 10-05 21:06).
* Result: **attn_flash_llama7b_fused 144,008 VCS -> 141,892 (−1.5 %)** (was +8.5 %); **2h_fused 156,475 ->
  149,849 (−4.2 %)** (was +1.9 %). Regression otherwise unchanged.
* 2h_fused's remaining −4.2 %: per block the model matches (+18) except every 2nd block, where the RTL has a ~700-cycle
  mesh stall (the 727-cycle stalls): preload <- store f23 (waits 4.1k cycles) <- a 2,311-cycle VPU op completing.
  The vector queue issues in a different order in the model (same op types, durations within 5 cycles), so that
  chain does not form. Next if needed: compare the vector-queue issue order / deps.
* MxGemmini-workspace (Amanda Shi, `ppa/opt`): an analytic PPA model — DMA 3.13 B/c (128-bit bus), DDR3 peak
  21.3 B/c, DRAM energy presets, and DRAM *bytes / power* from DRAMSim2 epoch stats for validation. No cycle-level
  DRAM timing; useful to check this model's L2 -> DRAM traffic, not its latency.

#### 13.14.2 Per-chunk destination of loop scratchpad stores (2026-10-07)
2h_fused's alternate-block ~700-cycle stall is a chain: preload <- loop store f23 (dst 0x2b84, waits ~4.1k) <- a
2,311-cycle VPU op (src/dst 0x2000-0x2800, src2 0x2c00-0x2c80). The RTL store's range is exact
(ReservationStation.scala:312-325: BF16 (rows-1)*step + 8 = 0x2b84-0x2c04, overlapping the VPU op's src2, WAR);
the model gave every L_STSPAD the whole loop output (0x2000-0x2800) as its destination, so that dependency could
not form (and others did). Now `make()` computes LoopMatmulStCSpad's per-chunk dst (:854-918; BF16 / FP8 flat /
FP8 tiled via LOOP_WS rs2 bit 10 -> `loop_t::tiled`) and the RS range. `perf_cmd_bits.py` reads full 64-bit RS
operands (NpiWave.val truncates); the model trace now carries each command's row ranges.
Result: **2h_fused 156,475 -> 153,321 (−2.0 %)**, every key block within ±18 cycles of the RTL (alternate-block
stall 704 vs 743); remaining difference at the pass start-up. 7b_fused −1.5 % (unchanged).
