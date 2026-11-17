/* Actual DIM16 device commands and independent signed-unit matrix oracle.
 * Additional guards reject before CPU packing writes or device submissions. */
#define main original_projection_main
#include "test_micromix_projection.c"
#undef main
#include "mx_micromix_layer.h"
static uint8_t saved_workspace[M * (K + K / 32)] A64;
static int workspace_equal(void) {
  for (size_t i = 0; i < sizeof owned; i++)
    if (saved_workspace[i] != owned[i])
      return 0;
  return 1;
}
int main(void) {
  int failed = original_projection_main();
  mxmm_weight weights[7];
  for (int i = 0; i < 7; i++)
    weights[i] = (mxmm_weight){K, N, 5, 2, 1, b4, b6, b8, bsc};
  mxmm_layer_projections p = {.weights = weights,
                              .scratch = workspace,
                              .prepared_workspace = owned,
                              .prepared_capacity = sizeof owned,
                              .schedule = {128, 4, 1},
                              .prepared_reuse = 1};
  failed += mxmm_layer_begin(&p, 3);
  unsigned loops = mxn_loops;
  uint64_t pc = mxmm_pack_cycles, mc = mxmm_mesh_cycles;
  memcpy(saved_workspace, owned, sizeof owned);
  weights[1].b8 = owned;
  failed += mxmm_layer_project_fp8(&p, 0, a, asc, M, 0, prepared) != 1;
  failed += !workspace_equal() || mxn_loops != loops || pc != mxmm_pack_cycles ||
            mc != mxmm_mesh_cycles || mxmp_live != NULL;
  weights[1].b8 = b8;
  unsigned guards = 1;
  uint64_t compared = 0;
  for (int projection = 0; projection < 7; projection++) {
    if (projection == 1) {
      loops = mxn_loops;
      pc = mxmm_pack_cycles;
      mc = mxmm_mesh_cycles;
      memcpy(saved_workspace, owned, sizeof owned);
      weights[2].scales = owned;
      failed += mxmm_layer_project_fp8(&p, projection, a, asc, M, 0, prepared) != 1;
      failed += !workspace_equal() || mxn_loops != loops || pc != mxmm_pack_cycles ||
                mc != mxmm_mesh_cycles || p.phase.used != 1;
      weights[2].scales = bsc;
      guards++;
    }
    failed += mxmm_layer_project_fp8(&p, projection, a, asc, M, 0, prepared);
    mxmm_numeric_result result = mxmm_numeric_compare(prepared, ref, NULL, M * N);
    failed += !result.pass || result.bit_mismatches;
    compared += M * N;
  }
  failed += mxmp_live != NULL;
  printf(
      "MICROMIX_LAYER_PROJECTION words=%llu future_consumer_guards=%u bit_mismatches=%d pass=%d\n",
      (unsigned long long)compared, guards, failed, !failed);
  return failed != 0;
}
