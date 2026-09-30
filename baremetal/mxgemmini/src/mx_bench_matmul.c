// Single large MX matmul: utilization of the native-loop schedule (include/mx_native.h) and how well it
// hides load latency. C[M][N] = A[M][K] @ B[K][N], fp8 E4M3 + E8M0, BF16 out, data from a linked random
// blob (DRAM-cold, never touched by the CPU). Default M = 128 is compute-bound (~4 B/c of DRAM for a fed
// mesh), so its utilization measures latency hiding; M = 32 (mx_bench_matmul_m32) is llama-shaped and
// bound by DRAM (~8.3 B/c needed). Correctness: compare the printed checksum between Spike and RTL.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils.h"
#include "mx_bench.h"

#define DIM 16
#ifndef LLAMA_BANK_ROWS
#define LLAMA_BANK_ROWS 4096
#endif
#undef BANK_ROWS
#define BANK_ROWS LLAMA_BANK_ROWS
#include "mx_native.h"

#ifndef BENCH_M
#define BENCH_M 128
#endif
#define BENCH_K MXB_K
#define BENCH_N MXB_N
typedef char bench_m_fits[(BENCH_M <= MXB_MMAX && BENCH_M % DIM == 0) ? 1 : -1];

static uint16_t C[BENCH_M][BENCH_N] __attribute__((aligned(64)));

// Mesh busy-state split of the run (internal counters).
static const int ev[8] = { EXE_ACTIVE_CYCLE, MAIN_EX_CYCLES, MAIN_LD_CYCLES, MAIN_ST_CYCLES,
                           MAIN_LD_EX_CYCLES, MAIN_ST_EX_CYCLES, MAIN_LD_ST_CYCLES, MAIN_LD_ST_EX_CYCLES };
static const char *evn[8] = { "exe_active", "ex", "ld", "st", "ld_ex", "st_ex", "ld_st", "ld_st_ex" };
// Pass 2: load path / DMA. Slots 5 and 7 are EXTERNAL counters: they clear only on counter_reset and the
// snapshot reads them as 0, so they are read live after the fence (the DMA is idle then).
static const int lev[8] = { LOAD_ACTIVE_CYCLE, LOAD_DMA_WAIT_CYCLE, RDMA_ACTIVE_CYCLE, RDMA_XACT_FULL_CYCLES,
                            RDMA_TL_WAIT_CYCLES, RDMA_TOTAL_LATENCY, LOAD_CMD_TRACKER_FULL_CYCLES, RDMA_BYTES_REC };
static const char *levn[8] = { "ld_active", "ld_dma_wait", "rdma_active", "xact_full", "tl_wait",
                               "inflight_sum", "ld_cmd_full", "rdma_bytes" };

// Run a load-path pass with the load/DMA counter set and print cycles, counters and Little's-law figures.
static void ld_counters_start(void) {
  counter_reset();   // the external counters (slots 5, 7) clear only on a global reset
  counter_snapshot_reset();
  for (int i = 0; i < 8; i++) counter_configure(i, lev[i]);
}
static void ld_counters_report(const char *tag, uint64_t cyc) {
  uint32_t lc[8];
  counter_snapshot_take();
  for (int i = 0; i < 8; i++) lc[i] = counter_read(i);
  counter_snapshot_reset();
  lc[5] = counter_read(5);
  lc[7] = counter_read(7);
  printf("PERF_LD_HW %s", tag);
  for (int i = 0; i < 8; i++) printf(" %s=%u", levn[i], lc[i]);
  printf("\n");
  // Little's law: inflight_sum = sum over cycles of occupied read-xact slots. avg in flight = sum / cycles;
  // requests = bytes / 64 (64 B Gets) -> average latency per request = sum / requests.
  const uint64_t reqs = lc[7] / 64;
  printf("PERF_LITTLE %s avg_inflight=%lu.%lu lat_per_64B=%lu cyc  xact_full=%lu%%  tl_wait=%lu%%  "
         "ld_cmd_full=%lu%%  dma_Bpc=%lu.%02lu\n", tag,
         (unsigned long) ((uint64_t) lc[5] / cyc), (unsigned long) ((uint64_t) lc[5] * 10 / cyc % 10),
         (unsigned long) (reqs ? lc[5] / reqs : 0),
         (unsigned long) ((uint64_t) lc[3] * 100 / cyc), (unsigned long) ((uint64_t) lc[4] * 100 / cyc),
         (unsigned long) ((uint64_t) lc[6] * 100 / cyc),
         (unsigned long) (lc[7] / cyc), (unsigned long) ((uint64_t) lc[7] * 100 / cyc % 100));
}

int main() {
#ifdef BENCH_BLOCKED
  const uint8_t *A = MXB_AT(MXB_OFF_A), *B = MXB_AT(MXB_OFF_B_BLK);   // B pre-blocked (contiguous per loop)
  const int blk_nc = MXB_BLK_NC, blk_kt = MXB_BLK_KT;
#else
  const uint8_t *A = MXB_AT(MXB_OFF_A), *B = MXB_AT(MXB_OFF_B);
  const int blk_nc = 0, blk_kt = 0;
#endif
  const uint8_t *A_sc = MXB_AT(MXB_OFF_A_SC), *B_sc = MXB_AT(MXB_OFF_B_SC);
  const int NC = mxn_pick_nc(BENCH_M, BENCH_N), Kt = mxn_pick_kt(BENCH_M, BENCH_K, NC);
  const int loops = (BENCH_N / NC) * (BENCH_K / Kt);
  printf("mx_bench_matmul: M=%d K=%d N=%d -> %d loops of %dx%dx%d (NC=%d, Kt=%d), B %s\n",
         BENCH_M, BENCH_K, BENCH_N, loops, BENCH_M, Kt, NC, NC, Kt, blk_nc ? "PRE-BLOCKED" : "row-major");

  gemmini_flush(0);

  // ---- pass 0: load-only DMA ceiling on this platform: stream the 4 MB row-major B sequentially with
  // 64 B-row mvins (1 KB each, 4 tiles) into a rotating 8192-row spad window, no compute ----
  {
    const uint64_t bytes = (uint64_t) MXB_K * MXB_N;
    gemmini_extended3_config_ld(64, MVIN_SCALE_IDENTITY, false, 0);
    ld_counters_start();
    uint64_t s0 = read_cycles();
    for (uint64_t i = 0; i < bytes / 1024; i++)
      gemmini_extended_mvin((void *) (MXB_AT(MXB_OFF_B) + i * 1024), (uint32_t) ((i % 128) * 64), 64, 16);
    gemmini_fence();
    uint64_t sc = read_cycles() - s0;
    printf("PERF stream load-only %lu B in %lu cycles -> %lu.%02lu B/c\n", (unsigned long) bytes,
           (unsigned long) sc, (unsigned long) (bytes / sc), (unsigned long) (bytes * 100 / sc % 100));
    ld_counters_report("stream", sc);
  }

  counter_reset();
  counter_snapshot_reset();
  for (int i = 0; i < 8; i++) counter_configure(i, ev[i]);

  uint64_t t0 = read_cycles();
  // A rows are MXB_K apart and its scales [K/32][MXB_MMAX]: an M < MMAX run reads the first M of each.
  if (mxn_matmul_ex(A, MXB_K, B, MXB_N, &C[0][0], BENCH_N, A_sc, MXB_MMAX, B_sc, MXB_N,
                    BENCH_M, BENCH_K, BENCH_N, blk_nc, blk_kt)) return 1;
  gemmini_fence();
  uint64_t cyc = read_cycles() - t0;

  uint32_t cnt[8];
  counter_snapshot_take();
  for (int i = 0; i < 8; i++) cnt[i] = counter_read(i);
  counter_snapshot_reset();

  const uint64_t ideal = MXN_IDEAL(BENCH_M, BENCH_K, BENCH_N);
  // DRAM traffic: each loop reads its A (M x Kt) and B (Kt x NC) tiles; C is written once as BF16.
  const uint64_t rd = (uint64_t) loops * ((uint64_t) BENCH_M * Kt + (uint64_t) Kt * NC)
                    + (uint64_t) loops * (Kt / 32) * (BENCH_M + NC);
  const uint64_t wr = (uint64_t) BENCH_M * BENCH_N * 2;
  printf("PERF mx_bench_matmul M=%d cycles=%lu ideal=%lu util=%lu.%lu%%\n", BENCH_M, (unsigned long) cyc,
         (unsigned long) ideal, (unsigned long) (ideal * 100 / cyc), (unsigned long) (ideal * 1000 / cyc % 10));
  printf("PERF traffic read=%lu B write=%lu B -> %lu.%02lu B/c (demand at ideal: %lu.%02lu B/c)\n",
         (unsigned long) rd, (unsigned long) wr,
         (unsigned long) ((rd + wr) / cyc), (unsigned long) ((rd + wr) * 100 / cyc % 100),
         (unsigned long) ((rd + wr) / ideal), (unsigned long) ((rd + wr) * 100 / ideal % 100));
  uint64_t sum = 0;
  printf("PERF_HW");
  for (int i = 0; i < 8; i++) { printf(" %s=%u", evn[i], cnt[i]); if (i) sum += cnt[i]; }
  printf(" idle=%ld\n", (long) cyc - (long) sum);

  // ---- pass 2: the same matmul again (data >> L2, still DRAM-cold) with the load/DMA counters ----
  ld_counters_start();
  t0 = read_cycles();
  if (mxn_matmul_ex(A, MXB_K, B, MXB_N, &C[0][0], BENCH_N, A_sc, MXB_MMAX, B_sc, MXB_N,
                    BENCH_M, BENCH_K, BENCH_N, blk_nc, blk_kt)) return 1;
  gemmini_fence();
  uint64_t cyc2 = read_cycles() - t0;
  printf("PERF pass2 cycles=%lu util=%lu.%lu%%\n", (unsigned long) cyc2,
         (unsigned long) (ideal * 100 / cyc2), (unsigned long) (ideal * 1000 / cyc2 % 10));
  ld_counters_report("matmul", cyc2);

  uint64_t csum = 0, cx = 0;
  for (int m = 0; m < BENCH_M; m++)
    for (int n = 0; n < BENCH_N; n++) { csum += C[m][n]; cx ^= (uint64_t) C[m][n] << ((m + n) & 47); }
  printf("CHECKSUM M=%d sum=%lu xor=%016lx\n", BENCH_M, (unsigned long) csum, (unsigned long) cx);
  return 0;
}
