/* Host-only packing/key/lifetime proof. Native device calls are counted no-ops:
 * this test makes NO matmul/VPU numerical or target timing claim. Full real
 * layer numeric correctness is separately checked by corrected Spike. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "host_shim.h"
#include "mx_micromix_prepared.h"
static unsigned failures, checks, rejections, byte_checks;
static void check(int value) {
  checks++;
  if (!value)
    failures++;
}
static void *mem64(size_t bytes) {
  void *p = NULL;
  if (posix_memalign(&p, 64, bytes)) {
    exit(90);
  }
  memset(p, 0, bytes);
  return p;
}
static void ref_pack(const uint8_t *x, const uint8_t *sc, int m, int k, const int *ns, uint8_t *out,
                     uint8_t *as) {
  uint8_t lut[2][256];
  for (int g = 0; g < 2; g++)
    for (int c = 0; c < 256; c++)
      lut[g][c] = mxmm_encode(mxmm_decode(c, 8), g ? 6 : 4);
  size_t offset = 0;
  int kb = 0;
  for (int g = 0; g < 3; g++) {
    int kg = ns[g] * 32;
    uint8_t *dst = out + offset;
    for (int b = 0; b < ns[g]; b++)
      for (int r = 0; r < m; r++) {
        as[(kb + b) * m + r] = sc[(kb + b) * m + r];
        for (int j = 0; j < 32; j++) {
          uint8_t z = x[(size_t)r * k + (kb + b) * 32 + j];
          if (g < 2)
            z = lut[g][z];
          size_t at = (size_t)(g == 0 ? r / 2 : r) * kg + b * 32 + j;
          if (g == 0) {
            if (r & 1)
              dst[at] |= z << 4;
            else
              dst[at] = z;
          } else
            dst[at] = z;
        }
      }
    offset += (size_t)m * kg / (g == 0 ? 2 : 1);
    kb += ns[g];
  }
}
static int unchanged(mxmp_prepared *p, int used, unsigned calls, uint64_t ticks, uint64_t pack,
                     uint64_t mesh, unsigned emits, unsigned reuses) {
  return p->used == used && device_calls == calls && host_ticks == ticks &&
         mxmm_pack_cycles == pack && mxmm_mesh_cycles == mesh && mxmp_emits == emits &&
         mxmp_reuses == reuses && mxmp_live == p && p->valid;
}
#define REJECT(expr)                                                                               \
  do {                                                                                             \
    int use = p.used;                                                                              \
    unsigned calls = device_calls, em = mxmp_emits, re = mxmp_reuses;                              \
    uint64_t ticks = host_ticks, pc = mxmm_pack_cycles, mc = mxmm_mesh_cycles;                     \
    check((expr) == 1);                                                                            \
    check(unchanged(&p, use, calls, ticks, pc, mc, em, re));                                       \
    rejections++;                                                                                  \
  } while (0)
int main(void) {
  const int m = 64;
  uint64_t epoch = 0;
  for (int model = 0; model < 3; model++)
    for (int mix = 0; mix < 2; mix++) {
      int k = model == 0   ? 2048
              : model == 1 ? 3584
                           : 4096,
          f = model == 0   ? 5632
              : model == 1 ? 18944
                           : 11008,
          nkv = model == 0   ? 256
                : model == 1 ? 512
                             : 4096,
          nb = k / 32;
      int ns[3] = {mix ? nb * 5 / 8 : nb / 2, nb / 4, mix ? nb / 8 : nb / 4};
      check(ns[0] + ns[1] + ns[2] == nb);
      uint8_t *input = mem64((size_t)m * k), *sc = mem64((size_t)m * nb),
              *owned = mem64(mxmm_workspace_bytes(m, k)),
              *reference = mem64(mxmm_workspace_bytes(m, k));
      for (int i = 0; i < m * k; i++) {
        unsigned x = (unsigned)i * 773 + model * 13 + mix * 1381;
        input[i] = (x & 128) | ((x >> 8) % 127);
      }
      for (int i = 0; i < m * nb; i++)
        sc[i] = (uint8_t)(120 + (i * 173) % 15);
      ref_pack(input, sc, m, k, ns, reference, reference + (size_t)m * k);
      epoch++;
      for (unsigned phase = 1; phase <= 2; phase++) {
        int consumers = phase == 1 ? 3 : 2, n[3] = {phase == 1 ? k : f, phase == 1 ? nkv : f, nkv};
        mxmm_weight w[3];
        uint8_t *weight_alloc[3][4];
        uint16_t *output[3], *out_alloc[3];
        for (int t = 0; t < consumers; t++) {
          size_t spans[4] = {(size_t)ns[0] * 32 * n[t] / 2, (size_t)ns[1] * 32 * n[t],
                             (size_t)ns[2] * 32 * n[t], (size_t)nb * n[t]};
          for (int z = 0; z < 4; z++)
            weight_alloc[t][z] = mem64(spans[z]);
          w[t] = (mxmm_weight){k,
                               n[t],
                               ns[0],
                               ns[1],
                               ns[2],
                               weight_alloc[t][0],
                               weight_alloc[t][1],
                               weight_alloc[t][2],
                               weight_alloc[t][3]};
          out_alloc[t] = mem64((size_t)m * n[t] * 2 + 128);
          memset(out_alloc[t], 0x5a, (size_t)m * n[t] * 2 + 128);
          output[t] = out_alloc[t] + 32;
        }
        mxmp_prepared p = {0}, other = {0};
        mxmm_schedule schedule = {128, 4, model == 2 ? 1 : 0};
        check(mxmp_prepare_fp8(&p, input, sc, m, 0, &w[0], owned, mxmm_workspace_bytes(m, k),
                               schedule, epoch, phase, consumers) == 0);
        int kb = 0;
        size_t offset = 0;
        for (int g = 0; g < 3; g++) {
          size_t bytes = (size_t)m * ns[g] * 32 / (g == 0 ? 2 : 1);
          check(memcmp(p.ag[g], reference + offset, bytes) == 0);
          byte_checks += bytes;
          offset += bytes;
          kb += ns[g];
        }
        check(memcmp(p.as, reference + (size_t)m * k, (size_t)m * nb) == 0);
        byte_checks += m * nb;
        REJECT(mxmp_emit_fp8(&p, input, sc, m, 0, &w[0], output[0], schedule, epoch + 1, phase));
        REJECT(mxmp_emit_fp8(&p, input + 64, sc, m, 0, &w[0], output[0], schedule, epoch, phase));
        REJECT(mxmp_emit_fp8(&p, input, sc + 64, m, 0, &w[0], output[0], schedule, epoch, phase));
        REJECT(mxmp_emit_fp8(&p, input, sc, m, 32, &w[0], output[0], schedule, epoch, phase));
        REJECT(mxmp_emit_fp8(&p, input, sc, m, 0, &w[0], output[0], schedule, epoch, phase + 1));
        REJECT(mxmp_prepare_fp8(&other, input, sc, m, 0, &w[0], reference,
                                mxmm_workspace_bytes(m, k), schedule, epoch, phase + 1, 2));
        REJECT(mxmp_release(&p));
        for (int t = 0; t < consumers; t++) {
          if (t < 2) {
            for (int z = 0; z < 4; z++) {
              mxmm_weight bad = w[t];
              if (z == 0)
                bad.b4 = owned;
              else if (z == 1)
                bad.b6 = owned;
              else if (z == 2)
                bad.b8 = owned;
              else
                bad.scales = owned;
              REJECT(mxmp_emit_fp8(&p, input, sc, m, 0, &bad, output[t], schedule, epoch, phase));
              REJECT(mxmp_emit_fp8(&p, input, sc, m, 0, &w[t], (uint16_t *)weight_alloc[t][z],
                                   schedule, epoch, phase));
            }
            REJECT(mxmp_emit_fp8(&p, input, sc, m, 0, &w[t], (uint16_t *)owned, schedule, epoch,
                                 phase));
            REJECT(mxmp_emit_fp8(&p, input, sc, m, 0, &w[t], (uint16_t *)input, schedule, epoch,
                                 phase));
            REJECT(
                mxmp_emit_fp8(&p, input, sc, m, 0, &w[t], (uint16_t *)sc, schedule, epoch, phase));
          }
          check(mxmp_emit_fp8(&p, input, sc, m, 0, &w[t], output[t], schedule, epoch, phase) == 0);
          for (int j = 0; j < 32; j++) {
            check(out_alloc[t][j] == 0x5a5a);
            check(output[t][(size_t)m * n[t] + j] == 0x5a5a);
          }
        }
        check(mxmp_release(&p) == 0);
        check(!mxmp_live && !p.valid);
        check(mxmp_emit_fp8(&p, input, sc, m, 0, &w[0], output[0], schedule, epoch, phase) == 1);
        check(mxmp_prepare_fp8(&p, input, sc, m, 0, &w[0], owned, mxmm_workspace_bytes(m, k),
                               schedule, epoch, phase, consumers) == 1);
        for (int t = 0; t < consumers; t++) {
          for (int z = 0; z < 4; z++)
            free(weight_alloc[t][z]);
          free(out_alloc[t]);
        }
      }
      printf(
          "MX_MICROMIX_HOST_MODEL model=%s mix=%s full_rows=64 k=%d phases=2 byte_reference=independent_scalar_encode\n",
          model == 0   ? "tinyllama"
          : model == 1 ? "qwen7b"
                       : "llama7b",
          mix ? "mix625" : "mix50", k);
      free(input);
      free(sc);
      free(owned);
      free(reference);
    }
  printf(
      "MX_MICROMIX_HOST_GUARD pass=%d failures=%u checks=%u rejections=%u packed_bytes_checked=%u native_calls_mocked=1 full_layer_numerics_claim=0 target_timing_claim=0\n",
      failures == 0, failures, checks, rejections, byte_checks);
  return failures ? 1 : 0;
}
