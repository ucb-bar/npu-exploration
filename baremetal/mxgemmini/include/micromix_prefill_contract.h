#ifndef MICROMIX_PREFILL_CONTRACT_H
#define MICROMIX_PREFILL_CONTRACT_H
#include MICROMIX_CAPTURE_HEADER
#define LMX_M E2E_M
#define LMX_SK E2E_SK
#define LMX_D E2E_D
#define LMX_F E2E_F
#define LMX_NH E2E_NH
#define LMX_NKV E2E_NKV
#define LMX_HD E2E_HD
#define LMX_EPS E2E_EPS
#define LMX_BIAS E2E_BIAS
#define LMX_FMT 0
#define LMX_MODEL_NAME "caller supplied pretrained layer"
#define LMX_FMT_NAME "direct mixed FP4/6/8; FP8 VPU intermediates"
#define LMX_RMS_NC ((E2E_D % 2048) == 0 ? 2048 : 1792)
#define LMX_BK 128
#define LMX_BK_FIRST 128
#define LMX_BK_LAST E2E_M
#define LMX_AT(off, type) E2E_AT(off, type)
#define LMX_OFF_H_PRE E2E_OFF_H_PRE
#define LMX_OFF_W_IN_LN E2E_OFF_W_IN_LN
#define LMX_OFF_W_POST_LN E2E_OFF_W_POST_LN
#define LMX_OFF_BQ E2E_OFF_BQ
#define LMX_OFF_BK E2E_OFF_BK
#define LMX_OFF_BV E2E_OFF_BV
#define LMX_OFF_ROPE_CQ E2E_OFF_ROPE_CQ
#define LMX_OFF_ROPE_SQ E2E_OFF_ROPE_SQ
#define LMX_OFF_ROPE_CK E2E_OFF_ROPE_CK
#define LMX_OFF_ROPE_SK E2E_OFF_ROPE_SK
#define LMX_OFF_KT_CACHE E2E_OFF_KT_CACHE
#define LMX_OFF_KT_SCALES E2E_OFF_KT_SCALES
#define LMX_OFF_V_CACHE E2E_OFF_V_CACHE
#define LMX_OFF_V_SCALES E2E_OFF_V_SCALES
#define LMX_OFF_MASK E2E_OFF_MASK
#define LMX_OFF_H_MODEL E2E_OFF_H_MODEL
#define LMX_OFF_REF_O E2E_OFF_REF_O
#define LMX_OFF_REF_YATTN E2E_OFF_REF_YATTN
#define LMX_OFF_REF_HMID E2E_OFF_REF_HMID
#define LMX_OFF_REF_YMLP E2E_OFF_REF_YMLP
#define LMX_OFF_REF_HOUT E2E_OFF_REF_HOUT
#if MICROMIX_MODEL == 1
typedef char micromix_prefill_capture_shape[(E2E_M == 64 && E2E_SK == 2048 && E2E_HD == 128 &&
                                             E2E_D == 3584 && E2E_F == 18944 && E2E_NH == 28 &&
                                             E2E_NKV == 4 && E2E_BIAS == 1)
                                                ? 1
                                                : -1];
#elif MICROMIX_MODEL == 2
typedef char micromix_prefill_capture_shape[(E2E_M == 64 && E2E_SK == 2048 && E2E_HD == 128 &&
                                             E2E_D == 4096 && E2E_F == 11008 && E2E_NH == 32 &&
                                             E2E_NKV == 32 && E2E_BIAS == 0)
                                                ? 1
                                                : -1];
#else
#error "Generic prefill requires Qwen2.5-7B or Llama2-7B selector"
#endif
#endif
