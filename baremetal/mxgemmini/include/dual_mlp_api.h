// dual_mlp_api.h -- the interface between src/dual_mlp_llama.c (driver) and src/dual_mlp_inst.c (one per Gemmini).
#ifndef INCLUDE_DUAL_MLP_API_H
#define INCLUDE_DUAL_MLP_API_H
#include <stdint.h>

typedef struct {
  const uint8_t *x, *x_sc, *wg, *wg_sc, *wu, *wu_sc, *wd, *wd_sc;
  uint8_t *h, *h_sc;     // H_full [M][F] E4M3 codes, [F/32][M] E8M0 -- written by both Gemminis in phase 1
  uint16_t *y;           // Y [M][D] BF16 -- written by both in phase 2
} dml_bufs_t;

typedef struct {
  int dim, opcode, f0, fg, n0, ng, nc, kt, nc2, sp_end, sp_rows;
  uint64_t ideal1, ideal2;
} dml_info_t;

// step(): issue the next op; > 0 = its estimated cycles, 0 = reached the phase barrier, -1 = done
void dml_init_g16(const dml_bufs_t *b);
int  dml_step_g16(void);
void dml_info_g16(dml_info_t *i);
void dml_init_g8(const dml_bufs_t *b);
int  dml_step_g8(void);
void dml_info_g8(dml_info_t *i);

#endif
