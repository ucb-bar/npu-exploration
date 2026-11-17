#ifndef MX_MICROMIX_DIM16_H
#define MX_MICROMIX_DIM16_H
/* Explicit software view of the qualified DIM16 mixed-precision image.
 * Include before mx_micromix.h. This never changes the repository's default
 * gemmini_params.h or the default FP8 build. Hardware must match this view. */
#include "include/gemmini_params.h"
#undef DIM
#undef BANK_ROWS
#undef ACC_ROWS
#undef LUT_W_CODEBOOKS
#define DIM 16
#define BANK_ROWS 4096
#define ACC_ROWS 512
#define LUT_W_CODEBOOKS 128
#endif
