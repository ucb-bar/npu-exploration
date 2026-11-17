#include "mx_micromix_dim16.h"
#ifdef MXMM_LEGACY_FIRST
#include "mx_native.h"
#include "mx_micromix.h"
#else
#include "mx_micromix.h"
#include "mx_native.h"
#endif
/* Compile-only: old read_cycles remains available, both headers share one
 * loop counter regardless of include order, and defaults retain FP8 helpers. */
uint64_t micromix_include_order_check(void) {
  mxn_loops++;
  return read_cycles() + mxn_loops + MXN_HALF_ROWS;
}
