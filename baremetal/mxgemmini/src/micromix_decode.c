#ifndef DECODE_HOST
#include "mx_micromix_dim16.h" /* exact DIM16 shared-accumulator target image */
#endif
/* Build with one explicit geometry. Inputs are generated headers and aligned blobs. */
#if defined(MICROMIX_DECODE_TINY)
#include "micromix_decode_tiny.h"
#elif defined(MICROMIX_DECODE_QWEN)
#include "micromix_decode_qwen.h"
#elif defined(MICROMIX_DECODE_LLAMA)
#include "micromix_decode_llama.h"
#else
#error "Select MICROMIX_DECODE_TINY, MICROMIX_DECODE_QWEN or MICROMIX_DECODE_LLAMA"
#endif
