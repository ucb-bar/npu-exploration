/* Actual phase preflight and v2 lease implementation; commands mocked, no
 * matmul numerical claim. Rejection must precede CPU workspace writes too. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "host_shim.h"
static int mxmm_project_fp8(const uint8_t *a, const uint8_t *sc, int m, int block,
                            const mxmm_weight *w, uint16_t *out, uint8_t *scratch,
                            mxmm_schedule s) {
  device_calls++;
  return 0;
}
#include "mx_micromix_layer.h"
#define M 64
#define K 256
#define N 128
#define WS (M * (K + K / 32))
#define A64 __attribute__((aligned(64)))
static uint8_t a[M * K] A64, sc[M * K / 32] A64, workspace[WS] A64, scratch[WS] A64,
    snapshot[WS] A64;
static uint8_t b4[7][K * N / 2] A64, b6[7][K * N] A64, b8[7][K * N] A64, bs[7][K * N / 32] A64;
static uint16_t output[M * N] A64;
static int failures, checks;
#define CHECK(x)                                                                                   \
  do {                                                                                             \
    checks++;                                                                                      \
    if (!(x))                                                                                      \
      failures++;                                                                                  \
  } while (0)
#define REJECT(i, dst)                                                                             \
  do {                                                                                             \
    unsigned calls = device_calls, pre = mxmp_prepares, em = mxmp_emits, re = mxmp_reuses;         \
    uint64_t pack = mxmm_pack_cycles, mesh = mxmm_mesh_cycles, ticks = host_ticks;                 \
    int used = p.phase.used, valid = p.phase.valid;                                                \
    mxmp_prepared *live = mxmp_live;                                                               \
    memcpy(snapshot, workspace, WS);                                                               \
    CHECK(mxmm_layer_project_fp8(&p, i, a, sc, M, 0, dst) == 1);                                   \
    CHECK(device_calls == calls && mxmp_prepares == pre && mxmp_emits == em &&                     \
          mxmp_reuses == re && mxmm_pack_cycles == pack && mxmm_mesh_cycles == mesh &&             \
          host_ticks == ticks && p.phase.used == used && p.phase.valid == valid &&                 \
          mxmp_live == live);                                                                      \
    CHECK(!memcmp(workspace, snapshot, WS));                                                       \
  } while (0)
int main(void) {
  memset(a, 56, sizeof a);
  memset(sc, 127, sizeof sc);
  memset(workspace, 0xa5, sizeof workspace);
  memset(output, 0x5a, sizeof output);
  mxmm_weight weights[7];
  for (int i = 0; i < 7; i++)
    weights[i] = (mxmm_weight){K, N, 4, 2, 2, b4[i], b6[i], b8[i], bs[i]};
  mxmm_layer_projections p = {.weights = weights,
                              .scratch = scratch,
                              .prepared_workspace = workspace,
                              .prepared_capacity = WS,
                              .schedule = {128, 4, 1},
                              .prepared_reuse = 1};
  CHECK(mxmm_layer_begin(&p, 1) == 0);
  CHECK(mxmm_layer_begin(&p, 1) == 1 && p.epoch == 1);
  memcpy(snapshot, workspace, WS);
  CHECK(mxmm_layer_project_fp8(&p, 0, a, sc, M, 96, output) == 1);
  CHECK(!memcmp(snapshot, workspace, WS) && !mxmp_live);
  REJECT(1, output);
  for (int future = 1; future <= 2; future++)
    for (int part = 0; part < 4; part++) {
      mxmm_weight save = weights[future];
      if (part == 0)
        weights[future].b4 = workspace;
      else if (part == 1)
        weights[future].b6 = workspace;
      else if (part == 2)
        weights[future].b8 = workspace;
      else
        weights[future].scales = workspace;
      REJECT(0, output);
      weights[future] = save;
    }
  REJECT(0, (uint16_t *)b6[1]);
  REJECT(0, (uint16_t *)workspace);
  REJECT(0, (uint16_t *)a);
  CHECK(mxmm_layer_project_fp8(&p, 0, a, sc, M, 0, output) == 0);
  CHECK(mxmp_live == &p.phase && p.phase.used == 1);
  const uint8_t *saved = weights[2].b8;
  weights[2].b8 = workspace;
  REJECT(1, output);
  weights[2].b8 = saved;
  CHECK(mxmm_layer_project_fp8(&p, 1, a, sc, M, 0, output) == 0);
  CHECK(mxmm_layer_project_fp8(&p, 2, a, sc, M, 0, output) == 0);
  CHECK(mxmp_live == NULL);
  saved = weights[5].scales;
  weights[5].scales = workspace;
  REJECT(4, output);
  weights[5].scales = saved;
  CHECK(mxmm_layer_project_fp8(&p, 4, a, sc, M, 0, output) == 0);
  CHECK(mxmm_layer_project_fp8(&p, 5, a, sc, M, 0, output) == 0);
  CHECK(!mxmp_live);
  REJECT(4, output);
  CHECK(mxmm_layer_begin(&p, 2) == 0);
  CHECK(mxmp_prepares == 2 && mxmp_emits == 5 && mxmp_reuses == 3);
  for (size_t i = 0; i < M * N; i++)
    CHECK(output[i] == 0x5a5a); /* no mocked device writes */
  printf(
      "MICROMIX_LAYER_TEST checks=%d failures=%d future_consumer_preflight=1 cpu_writes_on_rejection=0 device_calls_mocked=1\n",
      checks, failures);
  return failures != 0;
}
