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
