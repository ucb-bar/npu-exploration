#ifndef MX_LLM_GEOMETRY_H
#define MX_LLM_GEOMETRY_H
#include <stddef.h>
typedef enum { MX_LLM_TINYLLAMA = 0, MX_LLM_QWEN25_7B = 1, MX_LLM_LLAMA2_7B = 2 } mx_llm_model;
typedef struct {
  size_t hidden, intermediate, heads, kv_heads, head_dim;
  float rms_epsilon, rope_theta;
  int qkv_bias;
} mx_llm_geometry;
static inline const mx_llm_geometry *mx_llm_geometry_for(mx_llm_model model) {
  static const mx_llm_geometry geometries[] = {{2048, 5632, 32, 4, 64, 1e-5f, 10000.0f, 0},
                                               {3584, 18944, 28, 4, 128, 1e-6f, 1000000.0f, 1},
                                               {4096, 11008, 32, 32, 128, 1e-5f, 10000.0f, 0}};
  return (unsigned)model < 3 ? geometries + model : NULL;
}
/* Cache capacity includes valid prefix plus teacher inputs. Physical padding
 * is storage only: it must not change valid keys, scores or token positions. */
static inline int mx_llm_geometry_valid(const mx_llm_geometry *g) {
  return g && g->hidden == g->heads * g->head_dim && g->heads && g->kv_heads &&
         !(g->heads % g->kv_heads) && !(g->hidden % 32) && !(g->intermediate % 32);
}
#endif
