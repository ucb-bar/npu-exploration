#ifndef MX_NATIVE_STATE_H
#define MX_NATIVE_STATE_H
/* Shared include guard, TU-local execution state. Standalone kernel headers
 * need the device interface, not gemmini_testutils' external test functions. */
#define MXN_HALF_ROWS ((BANK_NUM * BANK_ROWS) / 2)
static unsigned mxn_loops; /* parity selects the next scratchpad half */
#endif
