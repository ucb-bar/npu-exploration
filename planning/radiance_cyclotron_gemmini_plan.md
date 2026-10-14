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
