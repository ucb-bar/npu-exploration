// dual_mlp_inst.c -- one Gemmini's half of src/dual_mlp_llama.c, compiled once per instance (-iquote dual/<inst>
// selects XCUSTOM_ACC / DIM; -DINST=16|8 names the exports). The work is a list of ops issued one per step() call, so the
// driver can interleave the two Gemminis' command streams from one CPU without fences:
//   phase 1: G/U hybrid loops (X resident, B double-buffered, BF16 C -> spad), per F-chunk: VPU SwiGLU, SPAD_REQUANT,
//            mvout of the E4M3 codes into H_full (scales go straight to DRAM from the requantizer)
//   barrier  (the driver fences once: H_full must be complete before either Gemmini reads it)
//   phase 2: Y[:, N_g] = H_full @ Wd[:, N_g] as native DRAM loops (A loaded once, B double-buffered)
#include <stdint.h>
#include <stddef.h>
#include "dual_loops.h"
#include "dual_mlp_llama.h"
#include "dual_mlp_api.h"

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define FN(x) CAT(x, CAT(_g, INST))

#define M DML_M
#define D DML_D
#define F DML_F
#if INST == 16
#define F0 0
#define FG DML_F16
#define N0 0
#define NG DML_N16
#else
#define F0 DML_F16
#define FG (DML_F - DML_F16)
#define N0 DML_N16
#define NG (DML_D - DML_N16)
#endif
#ifndef DML_NC
#define DML_NC 128   // phase-1 F-chunk (C columns per loop)
#endif
#ifndef DML_KT
#define DML_KT 512   // phase-1 K-tile
#endif
#ifndef DML_NC2
#define DML_NC2 128  // phase-2 N-chunk
#endif
#define NC DML_NC
#define KT DML_KT
#define NC2 DML_NC2
#define NK (D / KT)
#define NCH (FG / NC)
#define NCH2 (NG / NC2)

// phase-1 scratchpad map (rows)
#define XT_ROWS (M * KT / DIM)              // one K-tile of X, loop A layout
#define B_ROWS  (KT * NC / 2 / DIM)         // one FP4 B tile
#define C_ROWS  (M * NC * 2 / DIM)          // one BF16 [M][NC] chunk
#define HC_ROWS (M * NC / DIM)              // its E4M3 codes
#define SP_X    0
#define SP_B    (SP_X + NK * XT_ROWS)
#define SP_G    (SP_B + 2 * B_ROWS)
#define SP_U    (SP_G + NCH * C_ROWS)
#define SP_T    (SP_U + NCH * C_ROWS)
#define SP_H    (SP_T + 2 * C_ROWS)
#define SP_END  (SP_H + 2 * HC_ROWS)
typedef char FN(p1_fits)[(SP_END <= BANK_NUM * BANK_ROWS && FG % NC == 0 && D % KT == 0 && NG % NC2 == 0) ? 1 : -1];
typedef char FN(p2_fits)[(M * F / DIM + F * NC2 / 2 / DIM <= BANK_NUM * BANK_ROWS / 2) ? 1 : -1];

enum { OP_CFG1, OP_LOOP, OP_VPU, OP_REQ, OP_MVOUT, OP_BARRIER, OP_CFG2, OP_DOWN, OP_END };
typedef struct { uint8_t kind, gu, t; uint16_t c; uint16_t n; } op_t;   // gu: 0 G, 1 U; n: loop ordinal
#define MAX_OPS (2 + NCH * (2 * NK + 3) + 2 + NCH2 + 1)
static op_t ops[MAX_OPS];
static int n_ops, next_op;
static const dml_bufs_t *bufs;

void FN(dml_init)(const dml_bufs_t *b) {
  bufs = b;
  n_ops = next_op = 0;
  int n = 0;
  ops[n_ops++] = (op_t){OP_CFG1};
  for (int c = 0; c < NCH; c++) {
    for (int gu = 0; gu < 2; gu++)
      for (int t = 0; t < NK; t++) ops[n_ops++] = (op_t){OP_LOOP, gu, t, c, n++};
    ops[n_ops++] = (op_t){OP_VPU, 0, 0, c};
    ops[n_ops++] = (op_t){OP_REQ, 0, 0, c};
    ops[n_ops++] = (op_t){OP_MVOUT, 0, 0, c};
  }
  ops[n_ops++] = (op_t){OP_BARRIER};
  ops[n_ops++] = (op_t){OP_CFG2};
  for (int c = 0; c < NCH2; c++) ops[n_ops++] = (op_t){OP_DOWN, 0, 0, c};
  ops[n_ops++] = (op_t){OP_END};
}

// ideal mesh cycles (A8W4: 2 MACs / PE / cycle) -- the driver's pacing weight
#define MESH_CYC(m, n, k) ((uint32_t) ((uint64_t) (m) * (n) * (k) / (2 * DIM * DIM)))

int FN(dml_step)(void) {
  const op_t o = ops[next_op++];
  switch (o.kind) {
  case OP_CFG1:
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 2, 3, 0);
    gemmini_mxquant_config_mvout(0, 1, 1, 1, 0, 0, 0);
    gemmini_extended3_config_ld(D, MVIN_SCALE_IDENTITY, false, 0);
    gemmini_extended3_config_ld(F / 2, MVIN_SCALE_IDENTITY, false, 1);
    return 1;
  case OP_LOOP: {
    const uint8_t *W = o.gu ? bufs->wu : bufs->wg, *Ws = o.gu ? bufs->wu_sc : bufs->wg_sc;
    const int col = F0 + o.c * NC;
    const uint32_t bend = SP_B + ((o.n & 1) + 1) * B_ROWS;
    gemmini_loop_ws_mx_spad(M / DIM, NC / (2 * DIM), KT / DIM,
                            bufs->x + o.t * KT, W + (size_t) o.t * KT * (F / 2) + col / 2, D, F / 2,
                            SP_X + o.t * XT_ROWS, bend, (o.gu ? SP_U : SP_G) + o.c * C_ROWS,
                            bufs->x_sc + (o.t * KT / 32) * M, Ws + (o.t * KT / 32) * F + col, M, F,
                            o.t > 0, o.t == NK - 1, !(o.c == 0 && o.gu == 0));
    return MESH_CYC(M, NC, KT);
  }
  case OP_VPU: {   // T = G * sigmoid(G) * U, in BF16: MULS -1, EXP, ADDS 1, RCP, MUL G, MUL U
    const uint32_t g = SP_G + o.c * C_ROWS, u = SP_U + o.c * C_ROWS, t = SP_T + (o.c & 1) * C_ROWS;
    gemmini_vpu_scalar(4, t, g, 0xBF80, C_ROWS);
    gemmini_vpu_unary(5, t, t, C_ROWS);
    gemmini_vpu_scalar(3, t, t, 0x3F80, C_ROWS);
    gemmini_vpu_unary(6, t, t, C_ROWS);
    gemmini_vpu_binary(2, t, t, g, C_ROWS);
    gemmini_vpu_binary(2, t, t, u, C_ROWS);
    return 6 * M * NC / DIM;   // DIM BF16 / cycle (2 units x DIM/2 lanes)
  }
  case OP_REQ:     // H chunk -> E4M3 codes (flat [M][NC]) + E8M0 [NC/32][M] straight into H_full's scales
    gemmini_spad_requant(SP_H + (o.c & 1) * HC_ROWS, SP_T + (o.c & 1) * C_ROWS, M, NC, 0,
                         (uint64_t) (bufs->h_sc + ((F0 + o.c * NC) / 32) * M), 0);
    return M * NC / DIM;
  case OP_MVOUT: { // codes row m (NC bytes = NC/DIM spad rows) -> H_full[m][F0 + c*NC ..]
    gemmini_config_st(DIM);
    const uint32_t h = SP_H + (o.c & 1) * HC_ROWS;
    for (int m = 0; m < M; m++)
      for (int s = 0; s < NC / DIM; s += DIM) {
        const int rows = (NC / DIM - s) < DIM ? (NC / DIM - s) : DIM;
        gemmini_extended_mvout(bufs->h + (size_t) m * F + F0 + o.c * NC + s * DIM, h + m * (NC / DIM) + s, DIM, rows);
      }
    return M * NC / DIM;
  }
  case OP_BARRIER:
    return 0;
  case OP_CFG2:
    gemmini_extended3_config_ld(F, MVIN_SCALE_IDENTITY, false, 0);
    gemmini_extended3_config_ld(D / 2, MVIN_SCALE_IDENTITY, false, 1);
    gemmini_config_st(D * sizeof(uint16_t));
    return 1;
  case OP_DOWN: {
    const int col = N0 + o.c * NC2;
    gemmini_loop_ws_mx(M / DIM, NC2 / (2 * DIM), F / DIM,
                       o.c == 0 ? bufs->h : NULL, bufs->wd + col / 2, bufs->y + col, F, D / 2, D,
                       bufs->h_sc, bufs->wd_sc + col, M, D, false, 1, 1 + (o.c & 1));
    return MESH_CYC(M, NC2, F);
  }
  default:
    next_op--;
    return -1;
  }
}

void FN(dml_info)(dml_info_t *i) {
  i->dim = DIM; i->opcode = XCUSTOM_ACC; i->f0 = F0; i->fg = FG; i->n0 = N0; i->ng = NG;
  i->nc = NC; i->kt = KT; i->nc2 = NC2; i->sp_end = SP_END; i->sp_rows = BANK_NUM * BANK_ROWS;
  i->ideal1 = 2ull * M * D * FG / (2 * DIM * DIM);
  i->ideal2 = (uint64_t) M * F * NG / (2 * DIM * DIM);
}
