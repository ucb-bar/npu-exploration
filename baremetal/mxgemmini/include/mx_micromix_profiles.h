/* Explicit reviewed mixed-format profile; defaults remain unchanged. */
#ifndef MX_MICROMIX_PROFILES_H
#define MX_MICROMIX_PROFILES_H
#include "mx_llm_geometry.h"
#include "mx_micromix_prepared.h"
typedef enum {MXMM_PROFILE_MIX50=1,MXMM_PROFILE_MIX625=2} mxmm_profile_mix;
typedef enum {MXMM_PROFILE_PREFILL=0,MXMM_PROFILE_DECODE=1} mxmm_profile_regime;
typedef struct {mxmm_schedule schedule;unsigned char integer_pack,prepared_reuse,attention_mode,cache_lut,physical_padding,cache_time_tile,decode_issue_mode;} mxmm_profile;
/* attention_mode:0 scalar,1 GQA8 FMA,2 GQA7 FMA,3 MHA4 FMA,4 GQA8 tiled FMA,5 GQA7 tiled FMA,6 MHA QKT8/PVH1J16 FMA.
 * Values are bytes so prefill never implicitly includes strict-FP helpers. */
static const mxmm_profile mxmm_profiles[3][2][2]={
 { /* tinyllama */
  {{{128,4,0},1,1,0,0,0,0,0},{{128,4,0},1,0,0,0,0,0,0}}, /* prefill */
  {{{256,8,1},1,0,4,0,0,0,0},{{256,8,1},1,0,1,0,0,0,0}}, /* decode */
 },
 { /* qwen25_7b */
  {{{128,4,0},1,1,0,0,0,0,0},{{128,4,0},1,1,0,0,0,0,0}}, /* prefill */
  {{{256,8,1},1,0,5,0,0,0,0},{{256,8,1},1,0,2,0,0,0,0}}, /* decode */
 },
 { /* llama2_7b */
  {{{128,4,1},1,0,0,0,0,0,0},{{128,4,1},1,0,0,0,0,0,0}}, /* prefill */
  {{{256,8,1},1,0,6,1,1,16,0},{{256,8,1},1,0,3,1,1,0,0}}, /* decode */
 },
};
static inline const mxmm_profile *mxmm_profile_get(mx_llm_model model,mxmm_profile_regime regime,mxmm_profile_mix mix) {
 if((unsigned)model>=3 || (unsigned)regime>=2 || (mix!=MXMM_PROFILE_MIX50 && mix!=MXMM_PROFILE_MIX625))return NULL;
 return &mxmm_profiles[model][regime][mix-1];
}
/* Apply in the SAME translation unit as projection calls. An outstanding lease
 * prevents all state changes. Caller separately applies returned attention/cache
 * settings and explicitly chooses prepared vs canonical entrypoints. */
static inline int mxmm_profile_apply(mx_llm_model model,mxmm_profile_regime regime,mxmm_profile_mix mix,mxmm_schedule *schedule) {
 const mxmm_profile *p=mxmm_profile_get(model,regime,mix);
 if(!p || !schedule || mxmp_live)return 1;
 *schedule=p->schedule;mxmm_fast_pack=1;mxmm_integer_pack=p->integer_pack;return 0;
}
#endif
