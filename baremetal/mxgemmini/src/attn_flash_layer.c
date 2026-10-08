// A whole layer's attention as GQA-packed passes: every pass is attn_flash's pipelined pass on ATTN_HEADS query heads that
// share one kv head (Sq = 64 x heads), reading that pass's Q and its kv head's K/V from a --layer header
// (gen_attn_vpu.py --layer). Passes run back to back with a fence between them, cold: no warm-up, so each kv group's
// first pass loads K/V from DRAM and its next passes find them in the L2. Each pass's O hash is checked against Spike's.
#ifndef ATTN_HEADER
#define ATTN_HEADER "attn_vpu_layer5_p2.h"
#define ATTN_LAYER_EXPECT "attn_flash_layer5_expect.h"
#endif
#ifndef BK
#define BK 128
#endif
#define ATTN_LAYER 1
static int cur_pass, cur_kv;   // the pass being run and its kv head
#define ATTN_Q    LQ_IN[cur_pass]
#define ATTN_QS   LQ_SCALES[cur_pass]
#define ATTN_KT   LKT_IN[cur_kv]
#define ATTN_KTS  LKT_SCALES[cur_kv]
#define ATTN_V    LV_IN[cur_kv]
#define ATTN_VS   LV_SCALES[cur_kv]
#define ATTN_OREF LO_REF_F_F32[cur_pass]
#include "attn_flash.c"
#include ATTN_LAYER_EXPECT   // EXP_LAYER_HASH_O[ATTN_NPASS]

int main() {
  attn_print_config();
  printf("layer: %d query heads over %d kv heads, %d passes of %d heads (GQA packed), passes cold and fenced\n",
         ATTN_NPASS * ATTN_HEADS, ATTN_NKV, ATTN_NPASS, ATTN_HEADS);
  gemmini_flush(0);
  const uint64_t mesh = 2ULL * SQ * SK * D / (DIM * DIM);   // per pass
  uint64_t kern = 0, total = 0, kv_first = 0, kv_rest = 0;
  int bad = 0;
  for (int p = 0; p < ATTN_NPASS; p++) {
    cur_pass = p; cur_kv = LPASS_KV[p];
    memset(O_hw, 0xa5, sizeof(O_hw));
    fmask = 0;
    const uint64_t t0 = read_cycles();
    attn_load_q_scales();
    const uint64_t c = run_pipelined();
    total += read_cycles() - t0;
    kern += c;
    if (p == 0 || LPASS_KV[p] != LPASS_KV[p - 1]) kv_first += c; else kv_rest += c;
    printf("pass %2d: query heads %d-%d (kv head %d)\n", p, LPASS_HEAD0[p], LPASS_HEAD0[p] + ATTN_HEADS - 1, cur_kv);
    bad += attn_report(fnv(O_hw, sizeof(O_hw)), EXP_LAYER_HASH_O[p]);
    printf("PERF pass %2d: %6llu cycles; mesh ideal %llu -> util %llu%%\n", p, (unsigned long long)c,
           (unsigned long long)mesh, (unsigned long long)(mesh * 100 / c));
  }
  const int heads = ATTN_NPASS * ATTN_HEADS, n_first = ATTN_NKV, n_rest = ATTN_NPASS - ATTN_NKV;
  printf("PERF layer: %llu cycles over %d passes (%llu incl. per-pass Q-scale loads); mesh ideal %llu -> util %llu%%\n",
         (unsigned long long)kern, ATTN_NPASS, (unsigned long long)total, (unsigned long long)(mesh * ATTN_NPASS),
         (unsigned long long)(mesh * ATTN_NPASS * 100 / kern));
  printf("PERF layer: %llu cycles per head; first pass of a kv group (K/V cold) %llu avg, other passes %llu avg\n",
         (unsigned long long)(kern / heads), (unsigned long long)(kv_first / n_first),
         (unsigned long long)(n_rest ? kv_rest / n_rest : 0));
  printf("layer O hashes: %d/%d passes match Spike\n", ATTN_NPASS - bad, ATTN_NPASS);
  printf("attn_flash_layer %s\n", bad ? "FAILED" : "PASSED");
  return bad != 0;
}
