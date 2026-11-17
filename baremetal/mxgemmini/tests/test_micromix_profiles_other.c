#include "host_shim.h"
#include "mx_micromix_profiles.h"
int other_profile_default(void) {
  return mxmm_integer_pack == 0 && mxmm_fast_pack == 1 && !mxmp_live;
}
