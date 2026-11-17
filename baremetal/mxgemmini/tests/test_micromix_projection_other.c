#include "mx_micromix_dim16.h"
#include "mx_micromix_profiles.h"
int micromix_other_tu_defaults(void) {
  return mxmm_integer_pack == 0 && mxmm_fast_pack == 1 && !mxmp_live;
}
