#include <stdio.h>
#include "host_shim.h"
#include "mx_micromix_profiles.h"
int other_profile_default(void);
static int failures, checks;
#define CHECK(x)                                                                                   \
  do {                                                                                             \
    checks++;                                                                                      \
    failures += !(x);                                                                              \
  } while (0)
int main(void) {
  CHECK(mxmm_integer_pack == 0);
  CHECK(other_profile_default());
  for (int model = 0; model < 3; model++)
    for (int regime = 0; regime < 2; regime++)
      for (int mix = 1; mix <= 2; mix++) {
        const mxmm_profile *p = mxmm_profile_get(model, regime, mix);
        mxmm_schedule s = {7, 7, 7};
        CHECK(p != NULL && p->decode_issue_mode == 0);
        CHECK(mxmm_profile_apply(model, regime, mix, &s) == 0);
        CHECK(s.nc == p->schedule.nc && s.run == p->schedule.run &&
              s.fence_groups == p->schedule.fence_groups);
        CHECK(mxmm_integer_pack == 1 && mxmm_fast_pack == 1);
        CHECK(other_profile_default());
      }
  mxmm_schedule s = {7, 8, 9};
  mxmm_integer_pack = 0;
  mxmm_fast_pack = 0;
  CHECK(mxmm_profile_apply(0, 0, 0, &s) == 1);
  CHECK(mxmm_profile_apply(3, 0, 1, &s) == 1);
  CHECK(mxmm_profile_apply(0, 2, 1, &s) == 1);
  CHECK(mxmm_profile_apply(0, 0, 1, NULL) == 1);
  mxmp_prepared live = {0};
  mxmp_live = &live;
  CHECK(mxmm_profile_apply(0, 0, 1, &s) == 1);
  CHECK(s.nc == 7 && s.run == 8 && s.fence_groups == 9 && !mxmm_integer_pack && !mxmm_fast_pack);
  mxmp_live = NULL;
  CHECK(mxmm_profile_get(MX_LLM_TINYLLAMA, MXMM_PROFILE_PREFILL, MXMM_PROFILE_MIX625)
            ->prepared_reuse == 0);
  CHECK(mxmm_profile_get(MX_LLM_LLAMA2_7B, MXMM_PROFILE_PREFILL, MXMM_PROFILE_MIX50)
            ->schedule.fence_groups == 1);
  CHECK(mxmm_profile_get(MX_LLM_QWEN25_7B, MXMM_PROFILE_DECODE, MXMM_PROFILE_MIX625)
            ->attention_mode == 2);
  CHECK(
      mxmm_profile_get(MX_LLM_LLAMA2_7B, MXMM_PROFILE_DECODE, MXMM_PROFILE_MIX50)->attention_mode ==
      6);
  CHECK(mxmm_profile_get(MX_LLM_LLAMA2_7B, MXMM_PROFILE_DECODE, MXMM_PROFILE_MIX50)
            ->cache_time_tile == 16);
  CHECK(mxmm_profile_get(MX_LLM_LLAMA2_7B, MXMM_PROFILE_DECODE, MXMM_PROFILE_MIX625)
            ->attention_mode == 3);
  CHECK(mxmm_profile_get(MX_LLM_LLAMA2_7B, MXMM_PROFILE_DECODE, MXMM_PROFILE_MIX625)
            ->cache_time_tile == 0);
  for (int model = 0; model < 3; model++)
    CHECK(mx_llm_geometry_valid(mx_llm_geometry_for(model)));
  CHECK(mx_llm_geometry_for(3) == NULL);
  printf("MICROMIX_PROFILE_TEST checks=%d failures=%d\n", checks, failures);
  return failures != 0;
}
