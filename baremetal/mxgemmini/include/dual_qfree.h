// dual_qfree.h -- credit-based issue on the dual SoC: count this instance's RoCC commands, and query its free
// command-queue entries (QUEUE_FREE, funct 125: accepted even when the queue is full, answered at once, never enqueued),
// so the driver issues a step only when it fits and the CPU never blocks on one Gemmini's full queue.
// Include before any gemmini header (counts through ROCC_INSTRUCTION_0_R_R), and again with DQ_IMPL after them.
#ifndef DQ_IMPL
#ifndef DUAL_QFREE_PRE_H
#define DUAL_QFREE_PRE_H
#include <stdint.h>
#include "rocc-software/src/xcustom.h"
static uint32_t dq_n;   // commands issued by this instance
#undef ROCC_INSTRUCTION_0_R_R
#define ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, func7)                                                          \
  {                                                                                                          \
    dq_n++;                                                                                                  \
    asm volatile(".insn r " STR(CAT(CUSTOM_, x)) ", " STR(0x3) ", " STR(func7) ", x0, %0, %1" ::"r"(rs1), \
                 "r"(rs2));                                                                                  \
  }
#endif
#else
#ifndef DUAL_QFREE_IMPL_H
#define DUAL_QFREE_IMPL_H
#define k_QUEUE_FREE 125
static inline uint32_t dq_free(void) {
  uint64_t r;
  ROCC_INSTRUCTION_R_R_R(XCUSTOM_ACC, r, 0, 0, k_QUEUE_FREE);
  return (uint32_t) r;
}
#endif
#endif
