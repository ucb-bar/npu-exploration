#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#include "mx_micromix_formats.h"
#include "mxmm_pack2.h"
#include "mx_micromix_numeric.h"
static unsigned long checks, failures;
#define CHECK(x)                                                                                   \
  do {                                                                                             \
    checks++;                                                                                      \
    if (!(x))                                                                                      \
      failures++;                                                                                  \
  } while (0)
int main(void) {
#ifdef __SSE__
  /* RISC-V has no x86 DAZ/FTZ; compare converters under target semantics. */
  _mm_setcsr(_mm_getcsr() & ~0x8040u);
#endif
  /* Normalized finite BF16 inputs across legal block powers, both signs,
  * all rounding midpoints, subnormals, zero and saturation. */
  for (unsigned u = 0; u < 65536; u++)
    if ((u & 0x7f80) != 0x7f80) {
      uint32_t raw = u << 16;
      float x;
      memcpy(&x, &raw, 4);
      int maximum = (int)((u & 0x7fff) >> 7) - 127;
      for (int f = 4; f <= 8; f += 2)
        for (int exponent = maximum < -127 ? -127 : maximum; exponent <= 127; exponent++) {
          CHECK(mxmm_pack2_encode(u, exponent, f) == mxmm_encode(ldexpf(x, -exponent), f));
        }
      if ((u & 0x7fff) != 0)
        CHECK(mxmm_numeric_value(u) == ldexp((double)((u & 127) | ((u & 0x7f80) ? 128 : 0)),
                                             ((u & 0x7f80) ? (int)((u >> 7) & 255) - 134 : -133)) *
                                           ((u & 0x8000) ? -1 : 1));
    }
#ifdef __SSE__
  /* The numeric gate must retain BF16 subnormals even when host DAZ/FTZ is on. */
  _mm_setcsr(_mm_getcsr() | 0x8040u);
#endif
  uint16_t zero = 0, tiny = 1, nan = 0x7fc1, inf = 0x7f80, base = 0x42c8, a = 0x42ca, b = 0x42cc;
  CHECK(mxmm_numeric_compare(&zero, &zero, NULL, 1).pass);
  CHECK(!mxmm_numeric_compare(&tiny, &zero, NULL, 1).pass);
  CHECK(!mxmm_numeric_compare(&nan, &zero, NULL, 1).finite);
  CHECK(!mxmm_numeric_compare(&zero, &inf, NULL, 1).finite);
  CHECK(mxmm_numeric_compare(&a, &base, NULL, 1).pass); /* 101 vs 100: 1% */
  CHECK(!mxmm_numeric_compare(&b, &a, &base, 1).pass);  /* update 2 vs 1 */
  printf("MICROMIX_FORMAT_TEST checks=%lu failures=%lu\n", checks, failures);
  return failures != 0;
}
