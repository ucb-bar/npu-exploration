// dual_attn_api.h -- the interface between src/dual_attn_llama.c (driver) and src/dual_attn_inst.c (one per Gemmini).
#ifndef INCLUDE_DUAL_ATTN_API_H
#define INCLUDE_DUAL_ATTN_API_H
#include <stdint.h>

typedef struct {
  const uint8_t *q;        // [NPASS][SQ][D] E4M3
  const uint8_t *q_fold;   // [NPASS][D/32][SQ] E8M0, 1/sqrt(d) folded in
  const uint8_t *kt, *kts; // K^T [D][SK], [D/32][SK]  (the KV group's, shared by every pass)
  const uint8_t *v, *vs;   // V [SK][D], [SK/32][D]
  uint16_t *o;             // [NPASS][SQ][D] BF16 out
} da_bufs_t;

typedef struct { int dim, opcode, bk, nb, stream, split; uint64_t pass_ideal; } da_info_t;

// step(): issue the next key block (or a pass's head / tail); > 0 = estimated cycles, -1 = all passes issued
void da_init_g16(const da_bufs_t *b, int first_pass, int n_passes);
int  da_step_g16(void);
void da_info_g16(da_info_t *i);
void da_init_g8(const da_bufs_t *b, int first_pass, int n_passes);
int  da_step_g8(void);
void da_info_g8(da_info_t *i);

#endif
