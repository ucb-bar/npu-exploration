// The device's chained-requant block, lifted VERBATIM from gemmini.cc:1445-1509, as a standalone
// oracle. Not a model of it -- the same statements, so a disagreement is a bug in the Python.
//
//   stdin  (binary): int32 M, int32 N, int32 fmt (0=e5m2 1=e4m3_quad 2=e2m3 3=e3m2 4=e4m3-direct),
//                    int32 G,
//                    then M*N float32 values (row-major), then (M>>G)+1 rows of 16 uint8 book codes
//   stdout (binary): M*(N/32) uint8 scale codes, then M*N uint8 element codes, then M*N uint8 indices
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <vector>
#include "mx_fp_math.h"
using namespace mx;

int main() {
  int M, N, fmt, G;
  if (fread(&M, 4, 1, stdin) != 1) return 1;
  if (fread(&N, 4, 1, stdin) != 1) return 1;
  if (fread(&fmt, 4, 1, stdin) != 1) return 1;
  if (fread(&G, 4, 1, stdin) != 1) return 1;
  std::vector<float> C((size_t)M * N);
  if (fread(C.data(), 4, C.size(), stdin) != C.size()) return 1;
  int nbooks = ((M - 1) >> G) + 1;
  std::vector<uint8_t> books((size_t)nbooks * 16);
  if (fread(books.data(), 1, books.size(), stdin) != books.size()) return 1;

  const bool out_e5m2 = fmt == 0, out_e4m3_quad = fmt == 1, out_e2m3 = fmt == 2;
  const bool direct   = fmt == 4;   // E4M3-single, the 8-bit path (gemmini.cc:1303-1378)
  const int GROUP_OUT = 32, N_blocks = N / GROUP_OUT, log2_pmax = 0;
  std::vector<uint8_t> scales((size_t)M * N_blocks), elems((size_t)M * N), idxs((size_t)M * N);

  for (int m = 0; m < M; m++) {
    const uint8_t *lut_codes = &books[(size_t)(m >> G) * 16];
    for (int bi = 0; bi < N_blocks; bi++) {
      float vals[32], max_abs = 0.0f;
      for (int jj = 0; jj < GROUP_OUT; jj++) {
        // The device reads BF16 out of smem; the caller hands us fp32, so round as the store did.
        float v = bf16_to_f32(f32_to_bf16_rne(C[(size_t)m * N + bi * GROUP_OUT + jj]));
        vals[jj] = v;
        float a = fabsf(v);
        if (a > max_abs) max_abs = a;
      }
      uint8_t scale_code;
      float scale;
      if (direct) {
        // gemmini.cc:1335-1346: epsilon clamp, so an all-zero block is code 104 rather than 0.
        const float amax = (max_abs < FLT_EPSILON) ? FLT_EPSILON : max_abs;
        int s = (int)floorf(log2f(amax)) + 127;
        if (s < 0) s = 0;
        if (s > 254) s = 254;
        scale_code = (uint8_t)s;
        scale = ldexpf(1.0f, (int)scale_code - 127);
      } else if (max_abs == 0.0f) {
        scale_code = 0;
        scale = fpe8m0_decode(scale_code);
      } else {
        int max_exp = (int)floorf(log2f(max_abs));
        int s = (max_exp - log2_pmax) + 127;
        if (s < 0) s = 0;
        if (s > 254) s = 254;
        scale_code = (uint8_t)s;
        scale = fpe8m0_decode(scale_code);
      }
      scales[(size_t)m * N_blocks + bi] = scale_code;
      for (int jj = 0; jj < GROUP_OUT; jj++) {
        const int j = bi * GROUP_OUT + jj;
        float scaled = vals[jj] / scale;
        if (direct) {
          // The 8-bit path encodes the RAW float -- no bf16 round -- and has no finder.
          uint8_t c = fp8_e4m3_to_code(scaled);
          elems[(size_t)m * N + j] = c;
          idxs[(size_t)m * N + j] = c;
          continue;
        }
        uint16_t scaled_bf16 = f32_to_bf16_rne(scaled);
        uint8_t elem_code = out_e5m2      ? bf16_bits_to_e5m2_code(scaled_bf16)
                          : out_e4m3_quad ? fp8_e4m3_to_code(bf16_to_f32(scaled_bf16))
                          : out_e2m3      ? bf16_bits_to_fp6_e2m3_code(scaled_bf16)
                                          : bf16_bits_to_fp6_e3m2_code(scaled_bf16);
        uint8_t code = (uint8_t)(out_e5m2      ? fp8_e5m2_nearest_finder(elem_code, lut_codes)
                               : out_e4m3_quad ? fp8_e4m3_nearest_finder(elem_code, lut_codes)
                               : out_e2m3      ? fp6e2m3_nearest_finder(elem_code, lut_codes)
                                               : fp6e3m2_nearest_finder(elem_code, lut_codes));
        elems[(size_t)m * N + j] = elem_code;
        idxs[(size_t)m * N + j] = code;
      }
    }
  }
  fwrite(scales.data(), 1, scales.size(), stdout);
  fwrite(elems.data(), 1, elems.size(), stdout);
  fwrite(idxs.data(), 1, idxs.size(), stdout);
  return 0;
}
