# MX-Gemmini inside cyclotron (Radiance) — integration plan

Started 2026-10-13. Goal: model the whole Radiance SoC by running libgemmini (functional + perf, `both`) inside
cyclotron (`generators/radiance/cyclotron`), behind the RTL's MMIO interface, instead of cyclotron's fixed-latency
`TensorQueue`. The Spike/Rocket path stays as it is; both hosts live side by side, chosen by parameter.

Work branch: libgemmini `cyclotron-integration` (checkout: `cyclotron/libgemmini`, from `c17f961`).

## User decisions (2026-10-13)
- Functional Gemmini is wanted (`both`: the math of gemmini.cc + the timing model), not timing only.
- A "host" layer in the Gemmini model: Gemmini lives in the Radiance SoC (cyclotron, MMIO) or under Rocket (spike,
  RoCC); parametrizable.
- Scale memory, LUTs, requantizer: the latest Radiance kernels do NOT write them through MMIO windows; they trigger
  DMAs (MX_LOAD_SCALES incl. gated rs2[54], MX_LOAD_LUT, SPAD_REQUANT reading P from shared memory). Model those;
  the SF (0x88000/0x8A000) and requantizer (0x40000) windows are not modelled (warn if touched).
- Calibration: fa_mxfp8 (taken as `fa_mxfp8_sr`, the DMA-style kernel — to confirm). The user runs more VCS
  sims for validation. Waveforms: /scratch/nicorakela/radiance-hbm-cy/sims/vcs/output/
  chipyard.harness.TestHarness.RadianceE4M3MxGemminiSingleClusterTraceConfig (non-HBM; FireSim twin
  FireSimE4M3MxGemminiRadianceConfig).

## Facts (exploration 2026-10-13)
- MMIO (GemminiTile.scala:360-390), GPU-local 0x84000: INST +0x00 (W, issues; stalls until cmd.ready — no FIFO,
  raw_cmd_q 2 inside), READY +0x08, RS1 +0x10, RS2 +0x18 (64-bit, single per cluster), BUSY +0x20,
  OCCUPANCY +0x28 (LOOP_WS accepted − unrolled), RETIRED +0x38 (has_loop_retire_counter). Kernels: `sw.shared`
  stores (radiance-kernels lib/include/nightly/mx.h). CISC (+0x30) and CSR 0xacc unused by Muon kernels.
- Scratchpad = cluster SMEM (use_tl_ext_mem + use_shared_ext_mem, 4 banks, 128 KiB = 8192 rows x 16 B);
  accumulator internal 32 KiB (256 rows); DMA over CSBUS -> sbus -> L2 -> DRAM, 32 B requests (CacheBlockBytes).
  Gemmini config: e4m3SingleNoLutMxFPConfig + has_spad_requant + has_loop_retire_counter, DIM 16,
  spad_read_delay 4, acc_latency 3 (radiance Configs.scala:360-460).
- cyclotron: cycle-stepped, functional effect at timing-accept; reject -> replay. TensorQueue only counts bytes,
  triggers on GLOBAL stores (kernels use shared ones), never blocks a warp. No functional Gemmini. SMEM timing per
  core. No build.rs / C++ yet.
- gemmini.cc's Spike coupling: DRAM via read_from_dram / write_to_dram + 6 direct mmu stores, dprintf, the
  extension_t / rocc front; spad = internal array; geometry compile time (gemmini_params.h).
  perf/: only host/host_cache.h (spike cache_sim_t).

## Steps
- **S1 host abstraction** — `gemmini_host.h`: `gemmini_host_t` (functional DRAM load/store as the DMA sees it,
  commit logging) + `host.kind` (0 rocket, 1 radiance). Spike's becomes `rocket_host_t`; gemmini.cc reaches
  memory only through it. No behaviour change (perf_regress bit-identical, a `both` run identical).
- S2 Radiance build: gemmini.cc / perf without Spike, Radiance geometry, spad backed by the host's shared memory,
  external clock + MMIO command path in the model, C API (create / mmio_write / mmio_read / advance / busy +
  memory callbacks).
- S3 `radiance` timing preset (+ shared-memory-as-scratchpad knobs, Radiance memory path).
- S4 cyclotron: build.rs (cc, cargo feature), GemminiDevice on the 0x84000 shared window, `[gemmini]` config
  (model off|queue|gperf, mode perf|both, preset, overrides), sim waits for Gemmini idle.
- S5 memory coupling (own L2/DRAM first; later cyclotron's L2 + cluster-level SMEM banks).
- S6 calibration on fa_mxfp8(_sr): RTL INST-write replay first, then end-to-end kernel_cycles.
- Risk: functional effect at issue (spike `both`) vs Muon sharing the SMEM — check with goldens; fallback: apply
  each command's data effect at its modelled completion.

## Log
- 2026-10-13 **S1 done** (uncommitted, libgemmini `cyclotron-integration`): `gemmini_host.h` (interface, kind,
  load/store/store8/store16, log_commits), `rocket_host.h` (spike mmu, same access sizes), gemmini.cc memory +
  dprintf through `host_` (set_processor creates the rocket host; `p` kept for the spike front: on_cmd,
  illegal_instruction), `host.kind` knob (unused until S2/S3), Makefile `ROCC_TESTS` (vpu_ref.h now `-I`; the
  cyclotron checkout has no sibling gemmini-rocc-tests). Check vs HEAD build: perf_regress output identical, FP4
  109,526 identical, `both` runs of matmul_tiled_fp8_64x64_requant / 128x128 / chain_pipelined byte-identical
  (PASSED).
- 2026-10-13 **S2 done** (uncommitted, libgemmini `cyclotron-integration`). Radiance build without spike:
  `make libgemmini_radiance.a` (-DGEMMINI_NO_SPIKE -DGEMMINI_RADIANCE: BANK_ROWS 2048, ACC_ROWS 256).
  * gemmini.h/.cc: spike parts under `#ifndef GEMMINI_NO_SPIKE` (extension_t base, registration, perf member,
    `p`, illegal_instruction -> abort); `set_host()` + `execute(funct, rs1, rs2)` for other hosts.
  * `gemmini_spad.h` spad_t: row views over own storage (rocket, same semantics as the vector of rows) or the
    host's shared memory (radiance: row r at byte 16 r — RTL bank i at i*rows*16, Controller.scala:226-228;
    no resize/clear; row address wraps). 4 `auto &row = spad[..]` -> `auto row`.
  * `gemmini_host_t::shared_scratchpad()`, `command()` hook; `gemmini_host_log.h` (GEMMINI_FUNC_LOG=<file> on spike:
    C/L/S records). `gemmini_functs.h`: funct enum + moves_or_computes_data (moved out of gemmini_perf.cc).
  * perf: `model_t` radiance API — advance_to, offer (INST: refused while front end full), cmd_ready, busy
    (!idle || busy_until >= t), loops_pending (LOOP_WS accepted − unrolled), loops_retired (in order, all RS
    entries of the loop done: rs_cmd_t::loop + on_loop_done, LoopMatmul loops_unrolled), next_event.
    config_t::make(preset, overrides) (from_env uses it). host_cache.h stub without spike (host_tracking 2 aborts).
  * `gemmini_radiance.h/.cc`: C API (new/free/write/read/advance/busy/next_event/finish), radiance_host_t,
    registers INST/READY/RS1/RS2/BUSY/OCCUPANCY/RETIRED (others warn once).
  * Checks: tests/radiance_replay.cc replays spike func logs through the C API with the spike geometry —
    matmul_tiled_fp8_64x64_requant (78 cmds, 8,448 loads, 4,224 stores), 128x128 (222 / 33,792 / 32,768),
    chain_pipelined (612 / 69,760 / 67,584): **byte-identical**. tests/radiance_capi_test.cc: gemm_mx-style
    MXFP8 64x64x64 spad loop through MMIO, A/B resident in SMEM, C (BF16) in SMEM vs float within 3.6 % of
    sum|terms| (mesh E4M4..BF16 accumulation); BUSY drops, OCCUPANCY 0->1->0, RETIRED 1. Spike path bit-identical
    (perf_regress, FP4 109,526, both runs).
  * **Found: spad-mode LOOP_WS DRAM loads not modelled** (both models). RTL LoopMatmul loads A/B from the
    sticky LOOP_WS_CONFIG_ADDRS_AB addresses in spad mode too (ldA.io.req.valid has no spad_only term,
    LoopMatmul.scala:1509; gemm_mx relies on it). gemmini.cc mx_loop_ws_spad never loads; perf loop_matmul
    `total[L_LDA/B] = (!spad_only && ...)`. Proposed S2b.
- 2026-10-13 S2 committed `f655b65`. **S2b done** (uncommitted): spad-mode LOOP_WS loads its own A/B (RTL LdA/LdB
  run in either mode, issue only when the DRAM address != 0, skipped by rs2[3]/[4]; LoopMatmul.scala:64-68,
  93, 178-182, 1509). gemmini.cc `mx_loop_spad_loads(rs2)` (mvin id 0 / id 1 into the SPAD_AB tile layout, before
  mx_loop_ws_spad; same formulas as mx_loop_ws_dram); perf loop_matmul.cc L_LDA/L_LDB totals no longer exclude
  spad_only (addresses are already sticky in both models, as RTL). fa_mxfp8_sr depends on it (kernel.cpp
  138-191: Q, K, V blocks loaded by spad loops). capi test now the gemm_mx pattern (A/B from DRAM): C within
  0.036 of sum|terms| (same as resident), timing loads 8,192 B, loop done 2,541 (was 2,050 with no loads).
  Rocket: perf_regress, FP4, both runs identical; fresh spike logs replay IDENTICAL.
- 2026-10-13 S2b committed `158738a`. **S3 done** (uncommitted): `preset_radiance()` (config.cc; GEMMINI_PERF_CONFIG
  =radiance; the C API's default). RTL facts: e4m3SingleNoLut = standaloneMxFPConfig pipeline, the same as
  MxGemminiRocketConfig (spad_read_delay 4, acc_latency 3, mesh_output_delay 1, tile_latency 0, acc_banks 2, 32
  in flight) -> mx_rocket latencies kept. Changed: host.kind 1; mem.host_tracking 0 (no coherent CPU L1);
  frontend.depth 6 (INST drives io.cmd: raw_cmd_q + LoopConv + LoopMatmul input queues, 2 each; the RoCC 10
  included 2 core-side queues), frontend.latency 3 [estimate]; spad.bank_rows 2048 (128 KiB smem); acc 256 rows
  (bank_rows 128, total_rows 256); mem.line_bytes / bus_bytes / dma.get_bytes / put_bytes / scale.get_bytes 32
  (WithCacheBlockBytes(32), WithSystemBusWidth(256)); rs.packed_preload_align 1, rs.scale_cfg_waits_loads 0 (simv
  2026-10-07 11:37 is after b0dd6cb and 42c9871, before d3e6df6 FP4-only). L2 512 KiB x 8, DRAM 8 B/cycle
  (WithEdgeDataBits(64); GPU memory = the same ExtMem channel, WithExtGPUMem) = defaults.
  Deferred to S6 (need waveform evidence): the scratchpad's TileLink path to shared memory (extra read latency,
  bank contention with Muon), frontend.latency.
  capi test: 256 x 32 B Gets, loop 2,533; new back-pressure burst (8 INST behind a LOOP_WS) refused ~2,200
  cycles with READY 0 (non-loop commands wait for the loop's unrolling, as mx.h describes).
- 2026-10-13/14 S3 committed `8486020`. **S4 done** (uncommitted). cyclotron (`cargo build --release --features
  gemmini`; default build unchanged, pure Rust): build.rs (cc 1.2, already in Cargo.lock via rusqlite) compiles
  libgemmini's radiance build (GEMMINI_ROCC_TESTS or ../../gemmini/software/gemmini-rocc-tests for vpu_ref.h);
  src/gemmini.rs (GemminiConfig [gemmini]: enabled, mode both|perf, preset, overrides, mmio_base 0x84000 / 0x1000,
  report; GemminiDevice over the C API, DMA callbacks into gmem (addr & 0xFFFFFFFF), scratchpad = the cluster's
  SMEM FlatMemory pointer (FlatMemory::as_mut_ptr/len)); Cluster owns one device (cycle = cluster tick), every warp
  holds it; warp.rs: backend_timed replays an INST store while !READY (set_resource_wait_until + replay),
  mem_response routes window accesses to the device before taking the gmem lock (DMA callbacks take it);
  Sim: finish -> gemmini_report_cluster<N>.txt in the perf log dir; ClusterConfig.gemmini_config,
  Sim::new_with_gemmini (new_with_timing unchanged). Test plumbing: [sim] smem_init / smem_dump (nightly printBuf
  LAUNCH/GO magics at 0x80000 + slot*8, verify slots 16-23).
  Fixes found on the way: (1) **cyclotron Neutrino: barrier IDs 8-15 were decoded as generic jobs** (task < 8;
  RTL MuonCore barrierBits = 4 -> 16 IDs; fa_mxfp8_sr BAR_N 9 / BAR_ND 10 deadlocked) -> NUM_BARRIER_IDS 16 in
  neutrino.rs + scoreboard.rs. (2) libgemmini perf: an empty LOOP_WS (rs2 0x3f0: nothing to do) never entered
  active_ so never counted unrolled -> RETIRED stuck; now every loop enters and finished loops pop from the head
  (pop_done). (3) C API catches C++ exceptions (reports the command, aborts). GEMMINI_RADIANCE_TRACE=1 prints
  every MMIO access; finish report adds LOOP_WS pending / retired.
  Result: fa_mxfp8_sr (build/1sm radiance.elf) runs to completion — functional 332,074 cycles, --timing 526,414
  (Gemmini 728 cmds, 140 loops all retired, busy until 332,732; RTL VCS: kernel_cycles 349,281, Muon 485,293 —
  not comparable yet, S6). **GPU verify FAILS: exact 10 / 32,768, err 100 %** (same in both modes: a functional
  gap, not timing). gemm_mx (v3 default) stops at its first acc mvout: functional MVOUT has no MX accumulator
  path (chunk rs2[54] lands in the row count; MX loops never leave C in `accumulator`).
  Rocket: perf_regress, FP4, both runs identical; capi PASSED.
- 2026-10-14 S4 committed: libgemmini `f66d9b8`, cyclotron `0e84506` (new branch `gemmini-integration`).
  **S4b** (uncommitted, libgemmini): functional MX loop vs the Radiance kernels' patterns.
  * RTL MX accumulator for single-format E4M3 loops: `mx_acc` (MX_ACC_ROWS 512 rocket / 256 radiance, rows of 64
    BF16; a 16-row group = 16 x 64 output block, tile (i,j) at c_start + (i*ceil4(J)+j)*4 rows: LoopMatmulExecute
    c_addr / LoopMatmulStCSpad src_addr), compute and store cursors (move by MX_ACC_ROWS/2 when the unit issues for a
    loop with inc_acc / a C DRAM address: LoopMatmul.scala:1654, 1755); rs2[6] skips computes, rs2[7] stores; the
    store half is staged into mx_smem so the requant / deposit code is unchanged. Single x quad and the FP4/FP6
    branches keep the per-loop buffer.
  * A loop skipping computes and stores writes nothing (every format): before CONFIG_EX the K-load loops took a
    quad branch, ignored the skips and deposited zeros at C_spad 0 over K^T.
  * MVOUT from the MX accumulator (mx_scales_used): rs2 chunk[55:54] rows[53:48]; 32-col BF16 half per row (gemm_mx).
  * GEMMINI_RADIANCE_SNAP=<dir>: scratchpad after each LOOP_WS / SPAD_REQUANT (debug).
  Results: **gemm_mx (512^3 MXFP8, nightly) PASSES bit-exact in cyclotron** (exact 262,144 / 262,144, err 0;
  648,844 cycles functional). fa_mxfp8_sr: S_0 = Q K_0^T **bit-exact** vs gen_data.py --dump (8192/8192; data
  regenerated in scratch, identical to the repo's); P after the SIMT softmax is wrong except row 0 (mref too) ->
  a cyclotron Muon SIMT issue (lanes != 0 of the softmax warps), not Gemmini. Rocket: both runs identical, spike
  logs replay IDENTICAL, capi PASSED.
- 2026-10-14 S4b committed `0b988ca`. **Softmax/SIMT + scale-gating fixes** (uncommitted):
  * cyclotron FPU (src/muon/execute.rs): BF16 ("h", fmt 10 = BF16 on Muon, FPPipe.scala:80-81) compares
    feq/flt/fle.h, fsgnj/fsgnjn/fsgnjx.h, fcvt.h.s / fcvt.s.h (F2F, src = rs2[2:0]; f32->bf16 rounding by f3, dynamic
    = RNE: frm not modelled; results sign-extended, signExtendFp16Lanes) were missing; an unknown FP op silently
    left rd unwritten -> now reported once on stderr. fa_mxfp8_sr's max pass (flt.h + vx_split) used a stale rd:
    every lane took the `mv`, only row 0 came out right. ISA suite: 86 pass / 16 fail vs main 84 / 18 — the fix
    passes rv32uzfh-p-fcmp and -move, same 16 failures otherwise.
  * libgemmini functional: gated MX_LOAD_SCALES (rs2[54]) wait for their half to be FREE, in order (loads behind
    wait): queue + half states FREE/LOADED/INUSE as perf scale_loader (a managed CONFIG_SCALE_MEM takes the halves
    it selects, any config frees an INUSE half it does not select). Functional-at-issue had K_{g+3}'s scales
    overwrite half 0 before QK_{g+2} -> S_3.. wrong. Loop-managed (DRAM loop) loads stay immediate.
  * debug: GEMMINI_RADIANCE_SNAP also writes .acc / .sca / .scb / .cur (gemmini_t::debug_dump).
  Method: gen_data.py --dump intermediates (scratch copy) vs per-stage snapshots: S (QK store), P (SMEM before
  SPAD_REQUANT), Pc (PA tiles), V tiles, accumulator after each PV vs golden.mx_matmul partial sums, O store.
  **Result: fa_mxfp8_sr PASSES bit-exact** (exact 32,768 / 32,768) in functional (322,047 cycles) and --timing
  (468,726; Gemmini busy until 326,694). gemm_mx still exact. Rocket: both runs identical, replays IDENTICAL.
- 2026-10-14 committed: libgemmini `d5e5e8e`, cyclotron `d838760`. **S6 started** (calibration, fa_mxfp8_sr).
  Ground truth (VCS 2026-10-07, non-HBM SingleClusterTrace): START 11,183 / 11,182, END 360,200 / 360,463 (mcycle),
  kernel_cycles 349,281; Muon core 0 177,177 insts / 485,293 cycles.
  * Binary: the run's ELF was rebuilt since (text 0x5088 vs 0x4fa0 now). New `tools/radiance_elf_from_dump.py`
    rebuilds it from the run's .soc.dump (objdump -D bytes; bounds from the .log copy_elf lines; 2-byte tails that
    objdump cannot print come from --fill = the regenerated data regions, every printed byte checked against
    them); self-test on the current ELF identical for all 7 segments. -> s6/fa_mxfp8_sr.vcs1007.elf (scratch).
    In cyclotron it passes bit-exact (32,768 / 32,768).
  * RTL Gemmini command timeline: `waveform-debug/mxgemmini_debug_scripts/perf_radiance_cmds.py` (io_cmd valid &
    ready edges + rs1/rs2, busy edges; gemmini at ...tile_prci_domain_2.element_reset_domain_radiance_gemmini_tile_3,
    clock 2000 ps, edge0 1000): 728 commands, identical rs1/rs2 to the model's GEMMINI_RADIANCE_TRACE.
  * RTL Muon trace: `sims/vcs/fa_mxfp8_sr.soc.sqlite` (same run): inst (cycle per instruction, core cycles), smem /
    dmem (req / resp cycles). cyclotron fixes so its own trace compares: mcycle CSR (= the core cycle, warp.rs
    frontend), trace `cycle` / `req_cycle` filled (Tracer::stamp_cycle, MemTraceLine.req_cycle).
  * First numbers (model = cyclotron --timing + radiance preset, same binary): kernel_cycles 336,981 (−3.5 %);
    Gemmini first command -> last busy drop: RTL 308,497, model 321,433 (+4.2 %) -> compensating errors.
    RTL latencies: smem load 6, store 5; Gemmini MMIO st 3.5 / ld 3; dmem load mean 131 (p50 96, p90 265), store 282.
    Producer, one command: RTL 4-13 cycles between dependent instructions (lw.shared -> use 13, jal 9, beqz 12, ret
    9), 55 after the INST store; cyclotron 1-3 (no register dependence model: per-warp latency is only the FU queues).
- 2026-10-14 **S6a / S6b (Muon timing) done** (uncommitted).
  * `tools/radiance_trace_compare.py`: `latency` (per class: cycles to the warp's next instruction, dependent /
    independent; architectural registers from the ELF encoding -- the RTL trace's rs ids are physical, after
    renaming), `align` (per warp, k-th barrier / Gemmini register store vs START). RTL: ALU 4 (indep 2), FP32 7,
    BF16 6, lw.shared->use 13, lw.global 16, fexp 13, div 38 (blocks), branch/jump/SIMT/CSR -> next 9, fence 10.
  * cyclotron `src/muon/hazards.rs` (new; [hazards] in the timing config, off by default): per-warp RAW
    scoreboard, redirect after control flow, blocking divide, fence.s = ordering only (cyclotron sent an L1 FLUSH
    for it: f3 = 2 hit the `else flush_l1` branch; 200+ cycles vs RTL 10). Touches: core_graph.rs (config field),
    gmem/mod.rs + core.rs (timing model owns WarpHazards), muon/core.rs (stall + replay before backend_timed,
    record after issue), warp.rs (fence.s sends no flush when enabled). Off: run identical (429,662), ISA 86/16.
  * cyclotron's execute path blocks a warp until each instruction completes (issue_execute pending_execute) -> ~3
    cycles per instruction, dependent or not. With hazards: `config/timing/radiance/execute.toml` (units 0 latency,
    wide: never block); hazard values = RTL - 1 (the replay costs one cycle): `config/timing/radiance/hazards.toml`.
    Result: every class matches the RTL (dependent and independent). `config/radiance_e4m3_gemmini.toml`: the SoC.
  * fa_mxfp8_sr: kernel_cycles 340,797 vs 349,281 (−2.4 %); softmax compute per block 5,017 vs 5,104 (was 6,659).
    Left: (a) prologue −18.5k: the RTL L0d has 1 MSHR and allocates on writes (Rocket DCacheParams default), the
    producer's 42 cold stack stores wait ~900 each; cyclotron never fills on stores (cluster.rs:1049) -> needs a
    write-allocate policy option; (b) main loop: the softmax warps wait longer for the producer / Gemmini (BAR_P
    3,313 vs 2,421, BAR_S 720 vs 512 per block) -> S6c (Gemmini).
- 2026-10-14 mtests traces (msoftmax, regdep, mshort, scatgat, rb_base; sims/vcs/*.sqlite + output/mtests): HBM simv,
  multi-image launch ELFs (GPU code copied to 0x10000000 at launch) -> PCs not mappable statically; user will
  regenerate them for validation later. `radiance_trace_compare.py` now reads ELF64 / CPU-fused ELFs too.
  **gmem (Radiance):** cyclotron `GmemPolicyConfig.write_allocate` (default false; cluster.rs fills on stores when
  set) + `config/timing/radiance/gmem.toml` (L0d 4 KiB = 64 sets as L0dCacheConfig, 1 MSHR as DCacheParams default,
  write_allocate) in config/radiance_e4m3_gemmini.toml. Prologue fixed (first Gemmini store 23,906 vs RTL 24,468,
  was 6k), but kernel_cycles 403,536 (+15.5 %): the O-normalize warps' global stores / loads (and the final verify,
  82.9k RTL vs 142.2k) are slower than the RTL -> the gmem latencies / miss handling need their own calibration:
  RTL trace has dmem req/resp per lane; cyclotron's trace has no resp cycle yet (next: record gmem completions).
- 2026-10-14 **S6 gmem calibration** (cyclotron gmem completions recorded: `gmem_req` table, uncommitted; L0 MSHR
  1 -> 8 in radiance/gmem.toml, uncommitted). FSDB (NPI): the L0d is `l0d.tlnbdCache.nbdCache` with **4 MSHRs**.
  * RTL levels by address replay (scratch s6/rtl_levels.py: L0 64x64 B direct-mapped per core, L1 64 KiB, L2 512 KiB;
    the replay agrees with the model's own hit flags on its trace): unloaded (min / p10) L0 hit 6 (9 multi-line),
    L1 17-19, L2 ~34, DRAM p10 96 / p25 112. Model: L0 13 (floor 10 with tag/data 0 -- coalescer + links), L1 23, L2
    36, **DRAM 243**. New user runs (mt_new2 / mt_fa / silufuse, HBM simv, killed at 14:02 exit=255, ~40-107k cycles,
    runtime-assembled code -> PCs unmappable): used for latency only; L0 6/9 confirmed, prologue stores p50 440-570.
  * Prototyped in a scratch copy (s6/cyproto), fa_mxfp8_sr (RTL kernel_cycles 349,281):
    - DRAM base 200 -> 60 (model DRAM 103), L0 MSHRs 4: model DRAM stores p90 1,569 -> still queueing.
    - **in-order global stores** (RTL LSU.scala:550-643: a warp's stores issue in order and the queue head advances
      only on the memory response; loads wait for older stores): hazards.rs `store_in_order` + hooks in gmem/issue.rs
      (block a warp's gmem request while it has a store out) and gmem/completions.rs (store answered). Store DRAM
      p90 110. 333.3k.
    - **vx_join** is never issued in the RTL (0 in its trace, cyclotron 11,095): costs +3 on the preceding redirect.
      hazards.rs `Join` class + `join` hold (default 8 = old behavior; radiance 3). 329.1k; warp 0's 476-cycle
      command intervals +342 -> +46.
    - icache `hit_rate = 0.98` (probabilistic, ~80-cycle misses) -> 1.0: the RTL loop has no icache misses
      (long prologue gaps are not at line crossings). 326.8k.
  * Remaining: (a) prologue -18k: the 16 warp stacks are 128 KiB apart -> same set at L0 / L1 / L2; each RTL stack
    store evicts a dirty line and refills (~630 cycles, 4 in flight per core); cyclotron's writebacks are by fixed
    rate (l1/l2_writeback_rate), not by dirty state; (b) main loop +15k: the intervals where Muon waits on Gemmini
    (+600-800 per block) -> S6c.
- 2026-10-15 **The 4 fixes landed in cyclotron** (user-approved; as prototyped): radiance/gmem.toml DRAM 60 + L0 MSHRs
  4; hazards.rs `store_in_order` (+ hooks gmem/issue.rs, gmem/completions.rs); hazards.rs `Join` class + `join` hold
  (default 8 = old behavior, radiance 3); new radiance/icache.toml (hit_rate 1.0) used by config/radiance_e4m3_gemmini.toml.
  fa_mxfp8_sr on the SoC config: 387,298 cycles, kernel_cycles 326,766 (−6.4 %), exact 32,768 / 32,768; identical in
  4/4 runs. cargo test: 192 pass. Non-radiance configs (no [hazards]): same results as HEAD; note the default
  config is NOT deterministic run to run (fa_vcs.toml: 429,662 or 429,669, at HEAD too -- pre-existing).
  Docs: docs/radiance_perf_model.md (design + charts); project log ~/project-logs/radiance-perf-model.md.
- 2026-10-15 **S6c Gemmini-side calibration** (fa_mxfp8_sr; RTL = FSDB via NPI; model = GEMMINI_PERF_TRACE).
  * New RTL extractors (waveform-debug/mxgemmini_debug_scripts): perf_radiance_retire.py (gemmini.retired changes),
    perf_radiance_rs.py (RS alloc / issue per queue / completions), perf_radiance_tl.py <port> (TileLink A/D fires).
    Fixed an off-by-one in their handshake sampling (and perf_radiance_cmds.py): a fire is an edge whose PRECEDING
    cycle has valid & ready (lo < edge <= hi); counts were right, cycles / sampled fields could be one off.
    RTL gemmini.io_cmd_bits_inst_funct is 0 on every command (funct is not carried there); pair commands by index.
  * libgemmini trace additions (uncommitted, trace-only): `ev,loopdone,<cycle>,<loop>` (a loop's last RS entry done,
    model.cc), `ev,get|sget|lget,<cycle>,<line addr>` (address added) and `ev,data|cdata,<cycle>,<addr>` (response,
    memory_system.cc). Timing unchanged (387,298 / 365,257 identical with and without the trace).
  * Counts match: 728 cmds (rs1/rs2 identical), 140 loops, RS ex 32,874 / st 776 / vec 32; data Gets 33,792
    (gemmini.auto_spad_id_out; auto_tl_out = scale loader 1,056 = model sget; spad_writer_out = 12,415 Puts).
    RS ld: RTL 2,114 vs model 1,058 (each model loop_lda/ldb = 2 RTL entries) -- not yet explained, small effect.
  * Found 1: acc -> scratchpad stores (loop_stspad, 776): RTL reads the accumulator 1 row/cycle (acc_mems_0
    read_req ready every cycle; next store 19 cycles later, 520 of 776); model st.spad_read_interval 4 (Rocket
    requant-to-FP8 measurement) -> 67 cycles each. With 1: steady-state RS drift flat (was +17.7k over the loop),
    store gaps 19. kernel_cycles 326,766 -> 304,797.
  * Found 2: DMA Gets (RTL latency, by source; first touch of a 32 B line / repeat): RTL first mean 159 (p50 147,
    p90 262), repeat min 8 p50 20 mean 38; model first fixed ~125 (bandwidth-bound: 32 in flight x 4 cycles/line;
    mem.dram_latency has no effect), repeat 12. mem.dram_bytes_per_cycle 6.4 (5 cycles/line) -> first mean 158;
    mem.l2_hit_latency 8 -> repeat p50 8. kernel_cycles 306,505.
  * Phase breakdown (RTL / model with 1+2): START -> first Gemmini cmd 24,468 / 2,281 (−22.2k: Muon prologue);
    first cmd -> last loop 308,497 / 298,601 (−9.9k, all in the first ~50k: compute-only loops 0x3b8 −440..−1,312,
    Muon issues the first commands earlier); last loop -> END 16,316 / 5,623 (−10.7k: Muon's O stores).
  * cyclotron DRAM bytes_per_cycle 32 -> 8 and writeback rates 1.0: no effect on prologue / tail (2.5k / 6k).
    RTL prologue stack stores cost ~630 each: 16 stacks in one set at L0/L1/L2 -> dirty victim writebacks down to
    DRAM + refill, 4 MSHRs per core for 8 warps. Needs deterministic dirty tracking in cyclotron's caches.
- 2026-10-17 **libgemmini radiance preset committed** (cc9d917; cyclotron submodule bump 400953e): st.spad_read_interval 1,
  mem.dram_bytes_per_cycle 6.4, mem.l2_hit_latency 8 + trace events. fa_mxfp8_sr kernel_cycles 306,505, exact.
- 2026-10-17 **RTL memory path facts (simv 2026-10-07 = radiance-hbm-cy source; FSDB + NPI)** for the Muon prologue / tail:
  * L1 = TLNBDCache (Rocket NonBlockingDCache) 512 x 4 x 32 B, 8 MSHRs (FSDB l1cache.nbdCache.mshrs_0..7); L0d same cache
    type 64 x 1 x 64 B, 4 MSHRs. Write-back, write-allocate (no_alloc false). Dirty evictions = C Release, TLCToTLULNode
    -> A PutFullData to the level below; the MSHR does not wait for its ack.
  * L1 write-allocate confirmed: Puts that miss (address replay) are followed by an AcquireBlock of the line in 92 %
    (33/36, lag p50 4). But 3,204 of 3,240 Puts hit in the L1.
  * **MemParallelism (the prologue bottleneck)**: l0dInFlight 3 per core and l1InFlight 3 per cluster (DepthHellaCacheIF:
    a request held from A to D). FSDB: L1 input never above 3 (prologue: at 3 for 56 % of cycles, >= 2 for 76 %); core 0
    L0d input never above 3 (prologue at 3 for 69 %). L0d Acquire->Grant prologue p50 75 / mean 93, in flight mean 0.66.
  * Prologue L1 input: 796 Gets (mean 57) + 516 Puts (mean 21); an L0 miss = 2 L1 Gets (64 B / 32 B lines), a dirty L0
    eviction = 2 Puts. L1 -> L2 AcquireBlock p50 38 (L2 hits). Model: 85 / 127 prologue stores go to DRAM (its L2 thrashes:
    16 stacks in one 8-way set) -- RTL may hash L2 sets (radiance memory/AddressHash.scala), to check.
  * Prototype (scratch s6/cyproto2): MSHR same-set nack rule (l0/l1_mshr_set_conflict) -- no effect alone; in-flight caps
    l0_inflight_limit 3 / l1_inflight_limit 3 (cluster.rs admission + release at completion): tail 5,623 -> 10,497 (RTL
    16,316), prologue 2,281 -> 3,058 (RTL 24,468), kernel 306,505 -> 312,469. Prologue model stores n 127 p50 399 vs RTL
    n 290 p50 627.
- 2026-10-17 **Step A landed in cyclotron (uncommitted): in-flight caps** -- policy l0_inflight_limit / l1_inflight_limit
  (default 0 = off; radiance/gmem.toml 3 / 3): cluster.rs refuses a request while its core's L0 adapter (or, for an L0
  miss, its cluster's L1 adapter) holds the limit, takes the slots when accepted (also on an MSHR merge), releases them
  at completion (push_completion); request.rs inflight_held; cache.rs contains() (no LRU update).
  fa_mxfp8_sr: kernel 312,469 (RTL 349,281, −10.5 %), prologue 3,058 (24,468), Gemmini phase 298,914 (308,497), tail
  10,497 (16,316); exact, identical over 2 runs; per-level unloaded latencies unchanged (L0 13, L1 23, L2 36, DRAM 103).
  Default config identical to HEAD (409,985 -- HEAD changed from 429,662 with the libgemmini radiance preset of cc9d917,
  which the default [gemmini] uses). cargo test 192 pass.
  Worker warps' first barrier: RTL 21.7-23.4k, model 6.8-10.6k (was 1.4-5k). Warp 0 (Gemmini driver) first MMIO
  model ~3k vs RTL 24.5k. RTL prologue L1 input by region: stack 693 Gets (mean 61) + 512 Puts (22); code 103 Gets.
- 2026-10-18 Step A committed (cyclotron f40779a). **Steps B + C (uncommitted)**:
  * B `l1_inflight_per_line`: an L0 miss holds l0_line/l1_line (2) L1 slots. First version (wait for both) allowed one
    L0 miss at a time and starved core 1 (core 0 took every freed slot: workers 15.9k vs 30.1k). Refined: admitted with
    one free slot, the rest owed and granted in order as slots free; refused cores served oldest-waiter-first (a waiter
    counts only while still retrying: a stale stamp deadlocked the run at 40 Gemmini commands).
  * C `l0_writeback_model` (new gmem/dirty.rs): dirty L0 lines (store completions); an evicting fill or an L0 flush
    queues l0_line/l1_line Puts per dirty line; each takes an L1 slot for l0_writeback_cycles 22 (RTL L1 Put mean,
    99 % L1 hits); cache.rs fill() returns the evicted line.
  * fa_mxfp8_sr: A 312,469 -> B 322,164 -> C 324,406 (RTL 349,281, −7.1 %); prologue (warp 0 first cmd) 6,124 (24,468);
    workers' first barrier 16.4-17.4k (RTL 21.7-23.4k); Gemmini 303,765 (308,497); tail 14,517 (16,316). Exact, 2 runs
    identical, default config = HEAD (409,985), cargo test 192.
  * Still missing: RTL every warp's stack stores 597-658 apart (warp 0 too), per core ~1 L0 miss at a time (163 cycles
    per store per core; Acquire in flight mean 0.66). Model warp 0 89-103, workers 301-396. Candidate: the L0d nack --
    a miss to a set with an outstanding miss of another line is nacked, and with nackBlocksSetOnly = false (default,
    simv build) a nack blocks EVERY new request to that L0 until its replay succeeds.
- 2026-10-18 B + C committed (cyclotron 792c5aa). **Step E (uncommitted): l0_nack_blocks_all** -- mshr.rs conflicts()
  (no entry and table full, or another line of its set outstanding); cluster.rs: such an L0 miss is refused and
  recorded, and until a request for that line is accepted every other request of the core is refused.
  fa_mxfp8_sr 324,406 -> 324,061 (small); exact, deterministic, default config = HEAD, cargo test 192.
  Why small: model warp 0 runs its spill block before the workers' stores and finishes it uncontended. RTL warp 0 also
  starts first (latencies 157, 149, 6, 6), but once the workers store (~900) core 1's store latencies run 241 -> 909:
  the core clears ~one miss per 150-200 cycles; warp 0 then waits 1,172 -> 2,969 and stores ~600 apart. Model: one
  miss holds the L0 MSHR ~100 (the RTL's miss is serial: release, Acquire -> Grant ~93, refill, replay).
- 2026-10-18/19 Step E (L0 nack blocks all) and the l0_dirty_miss_cycles knob tried and **dropped** (user): E changed
  nothing measurable; the knob (FSDB: dirty-victim store miss +164 prologue / +64 Gemmini phase / +109 tail over a clean
  one; Release -> Acquire p50 65) fixed the Gemmini phase (−0.3 %) but pushed the tail +7..+22k and left the prologue.
  **Baseline = C (792c5aa).**
  * Prologue = known deviation, accepted: the slow RTL prologue is the kernels' callee-saved register spill to per-warp
    stacks that alias to one cache set; across ~25 traced kernels the stack-store burst runs 65-95 cycles per store
    per cluster (p50 latency 400-600; exceptions m0/m1 ~55-60, flash_attention_mx none). Cold L2: yes (670 of 833
    prologue stack fetches miss), but the L2 -> DRAM fetch is ~40 cycles, so not the cause. Model (C): 50 cycles/store.
  * Tail (last loop -> END = core 0 warp 0's path): RTL warps queue a burst of ~9 stores in ~40 cycles and then wait
    ~1.1-3.1k at the next load (lw ra / epilogue restores lw s0..s8) while the stores drain one at a time; the model
    stalled the warp at every store. Global-store memUpdate does not exist (LSU.scala:1446-1457: shared memory only).
- 2026-10-19 **store_in_order revised, committed (cyclotron 1b77626)**: stores queue without stalling the warp, go to
  memory one at a time per warp (pending.rs), loads wait for older stores; store-data entries held until the store
  issues; split-store counter bug fixed. fa_mxfp8_sr: kernel 330,801 (−5.3 %), prologue 8,268, Gemmini 303,637
  (−1.6 %), tail 18,896 (+2.6k). Remaining tail cause: four workers queue their bursts at once and block on the 8
  per-core store-data entries (RTL warps drift apart). Variants: global_stq 16 per warp: no change; 16 store-data
  entries: tail 17,547 -- not adopted (RTL value is 8).
