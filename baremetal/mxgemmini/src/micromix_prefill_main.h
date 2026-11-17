/* Shared invocation/reset only: model-specific operator bodies remain separate. */
#ifndef MICROMIX_MODEL
#error "Explicit model selector required"
#endif
#ifndef MICROMIX_MIX
#error "Explicit MIX50 or MIX625 profile selector required"
#endif
#ifndef MICROMIX_REPEATS
#define MICROMIX_REPEATS 1
#endif
#if MICROMIX_REPEATS < 1
#error "At least one functional invocation required"
#endif
static uint8_t saved_kt[E2E_NKV * E2E_HD * (E2E_SK + E2E_M)] A64;
static uint8_t saved_kts[E2E_NKV * (E2E_HD / 32) * (E2E_SK + E2E_M)] A64;
static uint8_t saved_v[E2E_NKV * (E2E_SK + E2E_M) * E2E_HD] A64;
static uint8_t saved_vs[E2E_NKV * ((E2E_SK + E2E_M) / 32) * E2E_HD] A64;
int main(void) {
  const mxmm_profile *profile =
      mxmm_profile_get(MICROMIX_MODEL, MXMM_PROFILE_PREFILL, MICROMIX_MIX);
  const int expected_k[7] = {E2E_D, E2E_D, E2E_D, E2E_D, E2E_D, E2E_D, E2E_F};
  const int expected_n[7] = {E2E_D, E2E_NKV * E2E_HD, E2E_NKV * E2E_HD, E2E_D, E2E_F, E2E_F, E2E_D};
  for (int i = 0; i < 7; i++)
    if (mxmm_weights[i].k != expected_k[i] || mxmm_weights[i].n != expected_n[i])
      return 1;
  if (!profile ||
      mxmm_profile_apply(MICROMIX_MODEL, MXMM_PROFILE_PREFILL, MICROMIX_MIX, &layer_schedule))
    return 1;
  uint8_t *capture = (uint8_t *)E2E_AT(0, uint8_t);
  asm volatile("" : "+r"(capture));
#define SAVE(name, offset) memcpy(saved_##name, capture + E2E_OFF_##offset, sizeof(saved_##name))
  SAVE(kt, KT_CACHE);
  SAVE(kts, KT_SCALES);
  SAVE(v, V_CACHE);
  SAVE(vs, V_SCALES);
#undef SAVE
  projections = (mxmm_layer_projections){.weights = mxmm_weights,
                                         .scratch = mxmm_scratch,
                                         .prepared_workspace = prepared_workspace,
                                         .prepared_capacity = sizeof(prepared_workspace),
                                         .schedule = layer_schedule};
  for (int variant = 0; variant < 2; variant++)
    for (int trial = 0; trial < MICROMIX_REPEATS; trial++) {
      mxmm_device_fence();
#define RESTORE(name, offset) memcpy(capture + E2E_OFF_##offset, saved_##name, sizeof(saved_##name))
      RESTORE(kt, KT_CACHE);
      RESTORE(kts, KT_SCALES);
      RESTORE(v, V_CACHE);
      RESTORE(vs, V_SCALES);
#undef RESTORE
      projections.prepared_reuse = variant ? profile->prepared_reuse : 0;
      if (mxmm_layer_begin(&projections, ++capture_epoch))
        return 1;
      mxmm_variant = variant ? "candidate" : "baseline";
      mxmm_trial = trial;
      mxmm_warmup = trial == 0;
      char id[128], *q = id;
      const char *parts[] = {"micromix_prefill_", mxmm_variant, "_"};
      for (int z = 0; z < 3; z++)
        for (const char *c = parts[z]; *c; c++)
          *q++ = *c;
      char digits[12];
      int nd = 0, t = trial;
      do {
        digits[nd++] = (char)('0' + t % 10);
        t /= 10;
      } while (t);
      while (nd)
        *q++ = digits[--nd];
      *q = 0;
      if (mxmm_run_once(id) || mxmp_live)
        return 1;
    }
  printf("MICROMIX_PREFILL_DONE numerical_runs=%d prepared_profile=%d timing_qualified=false\n",
         2 * MICROMIX_REPEATS, profile->prepared_reuse);
  return 0;
}
