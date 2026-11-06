// dual_loops.h -- loop helpers for the dual-Gemmini kernels (one instance per translation unit: XCUSTOM_ACC / DIM come
// from the params header the TU is built against).
#ifndef INCLUDE_DUAL_LOOPS_H
#define INCLUDE_DUAL_LOOPS_H

#include "include/gemmini_testutils.h"

// Hybrid native loop: A [I*DIM][K*DIM] and B from DRAM (the loop's own DMA loads), loop-managed E8M0 scales, BF16 C
// row-major into the SCRATCHPAD at c_sp (no DRAM round trip, so a VPU / SPAD_REQUANT behind it needs no fence).
//   a_sp: first spad row of A;  b_sp_end: one past B's last row;  store: last K-tile of a chunk (stores C, then the
//   next loop takes the other acc half);  skip_a: A already at a_sp (reuse; the loop does not reload it).
#define gemmini_loop_ws_mx_spad(I, J, K, A, B, A_stride, B_stride, a_sp, b_sp_end, c_sp, A_sc, B_sc, A_sc_pitch, \
                                B_sc_pitch, accumulate, store, skip_a) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(A_sc), (uint64_t)(B_sc), k_LOOP_WS_CONFIG_SCALES) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(A_sc_pitch), (uint64_t)(B_sc_pitch), k_LOOP_WS_CONFIG_SCALE_STRIDES) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), \
                             k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(A), (uint64_t)(B), k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(A_stride), (uint64_t)(B_stride), k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(a_sp), (uint64_t)(b_sp_end), k_LOOP_WS_CONFIG_SPAD_AB) \
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)((accumulate) ? 1 : 0), \
                             ((uint64_t)(c_sp) << 32) | 0x200U | 0x20U | ((store) ? 0x100U : 0x80U) | \
                             ((skip_a) ? 0x8U : 0U), k_LOOP_WS) \
  }

#endif // INCLUDE_DUAL_LOOPS_H
