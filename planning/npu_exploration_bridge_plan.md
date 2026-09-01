# npu-exploration: PyTorch → MxGemmini(Rocket) bridge plan

**Created** 2026-09-01 · **Status** Step 1 (this file) done; nothing implemented yet.

Goal: an end-to-end exploration platform in `generators/gemmini/npu-exploration/` where a model is
defined in PyTorch, compiled down to Rocket-hosted MxGemmini RoCC instructions, and run — first on
spike (`libgemmini`), later on a cycle-accurate substrate. Layers split by directory; all layers
coexist in this one repo.

**Guiding constraints (user, 2026-09-01):**
1. The Radiance MxGemmini flow is *the* reference. Almost every operation is identical or directly
   analogous. Follow it; do not redesign it.
2. **But do not depend on radiance-kernels.** Reimplement, taking ideas rather than code. Nothing we
   ship may include, link, or path into that tree. See §1.3 and §2.3.

These are compatible: radiance is the reference for *structure*, while every *fact* the target needs
is grounded in `generators/gemmini/` itself (§2.3).

---

## 1. What already exists (verified 2026-09-01)

### 1.1 `merlin/` — the compiler framework (submodule, `git@github.com:Rakanic/merlin.git`)

Package root is `merlin/merlin/`. It already implements the whole spine we want, for **int8 Gemmini**:

| Capability | Location |
|---|---|
| PyTorch capture → MLIR | `docs/guides/model2mlir.md` |
| whole model on an accelerator | `docs/guides/whole_model_on_accelerator.md:47` — `compile_model("small_llama", "int8", target="gemmini", run="mesh")` |
| command buffer → bare-metal C driver | `merlin/targets/gemmini/backend/gemmini_codegen.py` (178 lines; emits C calling `libgemmini` intrinsics) |
| run on spike | `merlin/targets/gemmini/backend/gemmini.py:201` — `spike --extension=gemmini <elf>`; `:79` `libgemmini_dir()` |
| oracle ladder | `merlin/targets/gemmini/contracts/target_contract.yaml` — L1 spike → L2 Verilator (`GemminiRocketConfig`) → L4 VCS → L5 FireSim |
| out-of-tree target packages | `merlin/python/merlin/targetgen/target_registry.py`; `docs/guides/target_resolution.md` |

**OOT target resolution precedence** (`target_registry.py:38-46`) — this is why our target can live in
this repo without forking merlin:

1. `MERLIN_TARGET_PATH` entries (`external`) — **always wins**
2. in-tree `merlin/targets/<name>/` (`reference`)
3. `out/build/generated/<name>/` (`external`)
4. `out/artifacts/targets/<name>/` (`generated`)

A package is identified by its contract's `name:` field, **not** its directory name. A path entry may
be a package root or a directory *of* package roots.

### 1.2 MX support in merlin today — and why it does not serve us

There is **no `merlin/merlin/targets/mx_gemmini/`**. What exists is two things, neither Rocket-hosted:

- `merlin/targets/muon/backend/muon_mx_codegen.py:1` — drives *cyclotron's MX-Gemmini co-model*
  through `mxgemm_lib.hpp`. SIMT host. Its own docstring: bakes operand codes from the capsule
  golden, "a PUBLIC-capsule reference path … not a general compiler capability."
- `merlin/experiments/capsule_bench/targets/mx_gemmini/contracts/hwbringup_mx_v0/isa_include/isa_definition.py:1`
  — this "mx_gemmini" ISA is the **MMIO/command-buffer** endpoint (`mxgemmini_mmio.h`); the only
  decodable word is the composed RoCC trigger. That is the Radiance-embedded MX PE.

**Reusable as-is** (not target-coupled):

- `merlin/contract/capsules/profiles/mx_gemmini.yaml` — the MX corpus profile (mxfp8/6/4, E8M0 block
  scale, bf16 accumulate, `atol=rtol=0.03125`, oracle tiers L0–L3).
- `merlin/contract/capsules/mx_gemmini/` — built capsules: `isa/M0..M5`, `layers/MB0_mxfp8_linear`,
  `model_slices/MF0..MF7`, `model/{M0_small_llama_mx,M1_lstmnetvit_mx}`. Each has `capsule.yaml` +
  `capsule.interface.mlir`.
- `merlin/out/artifacts/targets/radiance/hand_v0/` — a **complete OOT package to use as the template**:
  `manifest.yaml` (37), `contracts/target_contract.yaml`, `dialect.py` (204), `lowering.yaml` (18),
  `backend.py` (636), `derive_facts.py` (413).
- `merlin/out/artifacts/targets/radiance/contracts/residual.yaml:138` — cites
  `radiance-kernels/lib/include/mxgemmini_mmio.h` as its ABI source.

### 1.3 `radiance-kernels/` — read-only idea reference, NOT a dependency

**Decision (user, 2026-09-01): this work must not depend on radiance-kernels.** Reimplement; take
ideas, not code, headers, or build paths. Nothing we ship may `#include` from it, link against it, or
resolve a path into it. It stays an untracked local checkout and is not added to `.gitmodules`.

That is affordable because radiance-kernels is not the sole source of any *fact* we need — see §2.3.
Consulted while authoring, cited for provenance of ideas, never built against:

- `lib/include/mxgemmini_mmio.h` (119 lines) — the MMIO transport shim.
- `lib/mxgemm/mxgemm_lib.hpp` (853 lines) — the generalized MX GEMM structure: tiling,
  `calculate_spad_addr<is_b>()`, scale double-buffering, the `GemmConfig` knob factoring.
- `kernels/gemm_mxgemmini_ws/kernel.cpp` (38 lines) — the caller shape: `GemmConfig CFG{TILE_M=256,
  TILE_N=64, TILE_K=64, DATATYPE=FP8, QUANT_OUTPUT=false}` then `mxgemm<CFG>(M,N,K, C, ...)`.
- Other MX kernels worth reading later: `gemm_mxgemmini{,_ws_restream,_ws_downproj_fp4}`,
  `flash_attention_mx{,_fp6,_gemma,_gqa,_stable}`, `gemma_4norm`, `rmsnorm_gemma`,
  and `lib/mxgemm/mxgemm_lib_fused.hpp` beside the main lib.

### 1.4 The Rocket MX side (target)

- `software/libgemmini/` — spike functional model. `README.md` documents MX functs **23–29** and the
  `CONFIG_EX` MX bit-fields; **13/13 MX tests PASS**, bit-exact for BF16 out, code-exact for requant.
- `software/gemmini-rocc-tests/include/gemmini.h:60-83` — `k_MVOUT_SPAD 23`,
  `k_LOOP_WS_CONFIG_SPAD_AB 24`, `k_MX_LOAD_SCALES 27`, `k_MX_READ_SMEM 28`, `k_MX_LOAD_LUT 29`.
- `software/gemmini-rocc-tests/include/gemmini_mx_rocket.h` (30 lines) — **the Rocket transport shim**,
  mirror of `mxgemmini_mmio.h`.
- `software/gemmini-rocc-tests/bareMetalC/matmul_tiled_fp8_64x64.c` (239 lines) — the milestone golden.
  21 MX baremetal tests total; `build_spike.sh` builds them under `-DSPIKE_SIM`.
- RTL: `chipyard/GemminiConfigs.scala:49` `class MxGemminiRocketConfig`, wrapping
  `ConfigsFP.scala:330` `GemminiMxFPConfigs.standaloneMxFPConfig`.

---

## 2. The central finding: radiance and Rocket differ only in transport

`mxgemmini_mmio.h:62-67` does exactly one structural thing:

```c
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    store64_shared(GEMMINI_CTRL, GEMMINI_RS1_OFFSET, gemmini_arg_to_u64(rs1)); \
    store64_shared(GEMMINI_CTRL, GEMMINI_RS2_OFFSET, gemmini_arg_to_u64(rs2)); \
    store_shared  (GEMMINI_CTRL, GEMMINI_INST_OFFSET, \
      (0x7B)|(0<<7)|(3<<12)|(1<<15)|(2<<20)|((funct)<<25)); }
```

It replaces the RoCC instruction with three MMIO stores carrying **the same rs1/rs2/funct**.
Everything above that line is stock `gemmini.h`. The Rocket path is the *un-overridden* version.

The RTL says the same thing (`ConfigsFP.scala:328-329`):

> "Faithful standalone twin of the Radiance MX flow (`WithRadianceMxGemmini`): same functional
> params, but internal scratchpad + MMIO requant path instead of shared SRAM."

And its constants match the Rocket header exactly — `scale_mem baseAddr = 0x20000000` ≡
`MX_SCALE_BASE`; `mx_mmio_base = 0x20010000` ≡ `MX_LUT_BASE`. `gemmini_mx_rocket.h:1-5` states the
intent outright: scales go to a flat RAM window *"exactly like radiance's shared-mem path, just at a
different base."*

### 2.1 Operation correspondence

`mxgemm_lib.hpp:193-249` vs `matmul_tiled_fp8_64x64.c:82-135`:

| Operation | radiance (MMIO) | Rocket (RoCC) | Delta |
|---|---|---|---|
| `gemmini_flush(0)` | ✓ | ✓ | none |
| `gemmini_extended3_config_ex(WS,0,0,ACC_SCALE_IDENTITY,1,1,0,0,false,`**`A,B,C,uselut`**`)` | ✓ | ✓ | **none** — identical arg positions |
| `gemmini_mxquant_config_mvout(scale_dram, I, J, K, 0, 0, G)` (funct 26) | ✓ | ✓ | none |
| `gemmini_loop_ws_config_bounds(I,J,K,0,0,0)` | ✓ (×2 — two FSMs) | ✓ | none |
| `gemmini_loop_ws_spad(...)` (funct 24 + 8) | ✓ | ✓ | none |
| `gemmini_config_ld` / `config_st` / `extended_mvin` | ✓ | ✓ | none |
| `gemmini_fence()` | poll `GEMMINI_BUSY_ADDR` | RoCC fence | transport |
| **A/B block scales** | stores → `GEMMINI_SF_MEM_A/B` (`0x8A000`/`0x88000`) | flat window `MX_SCALE_A/W` @ `0x20000000`, **or** funct 27 `gemmini_mx_load_scales(ptr,len,sel)` | base + delivery |
| **FP6 LUTs** | stores → `GEMMINI_LUT0/1/2` (`0x84080/380/680`) | `mx_load_lut()` → `MX_LUT0/1/2` @ `0x20010100/500/900` + GO, **or** funct 29 | base + GO commit |
| **Output drain** | read from SMEM | no SMEM → mvout from accumulator (half-width, `mx_chunk_id` @ bit 54), **or** funct 28, **or** V1 internal-spad | real difference |
| **Addressing** | `rad_device_to_host_address()` | plain pointers | transport |
| **Host** | SIMT, `mu_schedule(…, 2 warps)` | one scalar Rocket core | tiling loop flattens |

Identical and portable verbatim: the tiling structure, `calculate_spad_addr<is_b>()` spad math, and
the constraints `TILE_K >= 32 && TILE_K % 32 == 0`, `TILE_M % PE_M == 0`, `TILE_N % PE_N == 0`.

### 2.2 Consequences for the design

1. **The emitter reimplements a known sequence, it does not invent one.** The target sequence is
   written down twice already — once Rocket-native (`matmul_tiled_fp8_64x64.c`, which PASSes) and once
   generalized (`mxgemm_lib.hpp`). We write from the former and let the latter inform the structure.
2. **Transport is the only target-specific axis, so it gets its own seam.** Not to serve Radiance —
   we are not depending on it — but because our own roadmap needs it: the V1/V2/V3 output modes in
   `../planning/mxgemmini_rocket_standalone_plan.md` vary exactly here (internal spad vs externalized
   TLRAM vs chunked acc-mvout), and V2 is explicitly the radiance-like externalized path.
3. **`GemmConfig{TILE_M,TILE_N,TILE_K,DATATYPE,QUANT_OUTPUT}` is a proven knob factoring.** Adopt the
   shape (an idea, not code); it is also exactly the axis set merlin's DSE would sweep.

### 2.3 What we take, and where each fact actually comes from

Every *fact* the target needs is grounded in `generators/gemmini/`, which we already depend on.
radiance-kernels contributes *structure* only.

| Need | Rocket-native source (the dependency) | radiance contributes |
|---|---|---|
| funct codes, macro signatures, rs1/rs2 packing | `software/gemmini-rocc-tests/include/gemmini.h` | nothing |
| MX functs 23–29, `CONFIG_EX` MX bit-fields | `software/libgemmini/README.md` + `gemmini.cc` | nothing |
| transport, scale window, LUT regmap + GO | `software/gemmini-rocc-tests/include/gemmini_mx_rocket.h` | the shim *pattern* |
| the full working sequence for one shape | `bareMetalC/matmul_tiled_fp8_64x64.c` (+20 more MX tests) | nothing |
| mesh, dtypes, E8M0 scale, bf16 accum, base addrs | `ConfigsFP.scala`, `MxRequantizer.scala`, `GemminiConfigs.scala` | nothing |
| generalizing one shape → arbitrary M/N/K | — | tiling + spad-addressing + double-buffering **approach** |
| knob surface for DSE | — | the `GemmConfig` **shape** |

Consequence: if radiance-kernels disappeared, Steps 2–7 would still be executable. Only the Step-4
generalization would need to be re-derived from first principles.

---

## 3. Repo layout

```
npu-exploration/
├── app/            L1  PyTorch models, quantization, experiment definitions
├── compiler/       L2  the OOT merlin target
│   └── targets/mx_gemmini_rocket/
│       ├── contracts/target_contract.yaml    # ISA/mesh/dtypes/sim tiers
│       ├── contracts/dialect_plan.yaml       # generated from the contract
│       └── backend/
│           ├── mxgemm_emit.py                # port of mxgemm_lib.hpp — transport-agnostic
│           ├── transport_rocket.py           # gemmini_mx_rocket.h: direct RoCC, flat window, mvout
│           └── runner.py                     # spike --extension=gemmini; parse OUT/METRIC/DONE
├── runtime/        L3  C harness, output protocol, include glue
├── sim/            L4  spike (libgemmini) runner; later verilator/firesim
├── merlin/             submodule — unforked
├── radiance-kernels/   reference implementation (see Q1)
├── planning/           this file
└── out/                runs + artifacts
```

Selected with `MERLIN_TARGET_PATH=$PWD/compiler/targets`, which resolves the whole directory of
packages (precedence 1, beats everything in-tree).

`radiance-kernels/` sits in the tree as a reading reference only — no build path resolves into it,
and it is deliberately absent from `.gitmodules` (§1.3).

**Decisions taken** (user, 2026-09-01):
- Target lives OOT in `npu-exploration/`, not in the merlin submodule. Keeps merlin pullable upstream.
- First milestone is the 64×64 mxfp8 matmul, mirroring `matmul_tiled_fp8_64x64.c` — a known-PASSing
  spike test, so any divergence localizes to our codegen rather than the model or spike.
- No dependency on radiance-kernels: reimplement, take ideas not code.

---

## 4. Contract field mapping

`compiler/targets/mx_gemmini_rocket/contracts/target_contract.yaml`. Merlin's cardinal rule is
*derive, never hardcode*; where a value cannot be derived, record `UNKNOWN` and fail closed rather
than substituting a default.

Every row below is sourced from `generators/gemmini/` — headers, the spike model, or the RTL Scala.
merlin's `hwbringup_mx_v0/isa_include/mmio_abi.py` is cited where it already transcribed the same RTL
fact with provenance (a convenience, not a dependency: each is independently re-derivable from the
Scala). No row depends on radiance-kernels.

| Field | Value | Source |
|---|---|---|
| `name` | `mx_gemmini_rocket` | ours (distinct from the MMIO `mx_gemmini`) |
| `family` | `tensor_resident` | same as gemmini contract |
| `endpoint_kind` | `inline_asm_insn` | real RoCC, unlike the MMIO endpoint |
| `encoding.rocc_custom_slot` | **3** (`XCUSTOM_ACC`, custom3, opcode `0x7B`) | `gemmini-rocc-tests/include/gemmini_params.h:7` `#define XCUSTOM_ACC 3`. **`libgemmini/README.md` says "custom-2" — that is a doc error**; the header, the RTL slot, and merlin's `mmio_abi.py` all agree on custom-3 |
| `encoding.addr_len` | 32 | `gemmini_params.h:9` |
| `semantic_class` | stock 0–15 **plus** 23–29 (MX) | `gemmini.h:31-66`. Funct 26 is spelled **`CONFIG_SCALE_MEM`** in the header (the README calls it `MXQUANT_CONFIG_MVOUT`); the macro is `gemmini_mxquant_config_mvout`. Header wins |
| mesh | 16×16, tile 1×1, WS | `ConfigsFP.scala` `defaultMxFPConfig`; mirrored in `mmio_abi.py` MESH |
| operand dtypes | mxfp8 e4m3 / mxfp6 e3m2 / mxfp4 e2m1 | `MxRequantizer.scala` `MxFloatFormat` |
| block scale | E8M0, 8-bit, bias 127, group 32 | `MxRequantizer.scala:450,458-468` |
| accumulate | bf16 (`MxFloat(8,8,4)`); no int32 path | `MxRequantizer.scala:442,456,461` |
| `runtime.rtl_sim_config` | `MxGemminiRocketConfig` | `chipyard/GemminiConfigs.scala:49` |
| `runner.tier_sim` | `{L1: spike, L2: verilator}` | mirrors gemmini contract; VCS/FireSim later |
| `runner.sim_via` | `chipyard` | same as gemmini |
| scale window | base `0x20000000`, W `+0x0000`, A `+0x2000` | `ConfigsFP.scala:338` ≡ `gemmini_mx_rocket.h:10-12` |
| LUT regmap | base `0x20010000`, ports 0/1/2, GO `+0x300` | `ConfigsFP.scala:339` ≡ `gemmini_mx_rocket.h:17-21` |

Known-`UNKNOWN` carried over from `mmio_abi.py` (record, do not fabricate): per-format exponent bias
(only the generic `MxFloat.bias` formula exists in `Arithmetic.scala`); the funct7 command enumeration
for the MMIO endpoint. Neither blocks the Rocket path — functs 23–29 *are* enumerated in `gemmini.h`.

---

## 5. Steps

| # | Step | Output | State |
|---|---|---|---|
| 1 | This plan file | `planning/npu_exploration_bridge_plan.md` | **DONE** 2026-09-01 |
| 2 | Verify toolchain: `$RISCV`, `spike`, `libgemmini.so`; build + run `matmul_tiled_fp8_64x64` under spike to establish the golden | a recorded PASS | **DONE** 2026-09-01 — see §6 |
| 3 | Skeleton dirs + `target_contract.yaml`; confirm `target_registry.resolve("mx_gemmini_rocket")` finds it via `MERLIN_TARGET_PATH` | resolvable OOT package | **DONE** 2026-09-01 — see §7 |
| 4 | `mxgemm_emit.py` — emit C for one fixed 64×64×64 mxfp8 tile, **reimplemented from `matmul_tiled_fp8_64x64.c`** (Rocket-native, PASSing). Structure informed by `mxgemm_lib.hpp` but written fresh; no radiance include or path | C that compiles | **DONE** 2026-09-01 — see §8 |
| 5 | `transport_rocket.py` + `runtime/` harness; emitted C reproduces the Step-2 golden bit-exactly | **the bridge** | |
| 6 | `runner.py` — spike invocation, OUT/METRIC/DONE parsing, wire to merlin's capsule grading | graded capsule | |
| 7 | Drive from `merlin/contract/capsules/mx_gemmini/isa/M0_mxfp8_single_tile` etc. — reuse the existing MX capsules | capsules PASS | |
| 8 | `app/` — PyTorch `nn.Linear` mxfp8 → capsule → spike, vs a PyTorch golden | PyTorch↔HW bridge closed | |
| 9 | Scale out: `layers/MB0_mxfp8_linear`, then `model/M0_small_llama_mx` via `compile_model(..., run="mesh")` | whole model | |
| 10 | Cycle-accurate tier: Verilator `MxGemminiRocketConfig` (L2) | perf numbers | |

Steps 4–5 are the load-bearing ones. Everything after reuses merlin machinery that already works for
int8 Gemmini.

---

## 6. Step 2 results — toolchain verified, golden established (2026-09-01)

**Golden output** (this is what Step 5 must reproduce bit-exactly):

```
fp8 WS matmul test PASSED (no mismatches).
Gemmini extension configured with:
    dim = 16
```

### 6.1 Reproducible recipe

```bash
cd /bwrcq/scratch/nicorakela/radiance-cy-dev && source ./env.sh   # sets RISCV=<repo>/.conda-env/riscv-tools

# 1. libgemmini (rebuild if sources are newer — the Makefile does NOT track header deps)
cd generators/gemmini/software/libgemmini && rm -f libgemmini.so && make     # NOT `make install`

# 2. the test ELF. build_spike.sh only exposes the DIRECTORY target `bareMetalC`; an individual
#    test needs the recursive sub-make with the top Makefile's vars passed through:
T=<...>/software/gemmini-rocc-tests
cd $T/build_spike/bareMetalC && make -f $T/bareMetalC/Makefile \
  abs_top_srcdir=$T XLEN=64 PREFIX=examples-bareMetalC src_dir=$T/bareMetalC \
  RUNNER="spike --extension=gemmini " matmul_tiled_fp8_64x64-baremetal

# 3. run, against the freshly built in-tree .so (NOT $RISCV/lib — see 6.3)
spike --extlib=$G/libgemmini/libgemmini.so --extension=gemmini \
      $T/build_spike/bareMetalC/matmul_tiled_fp8_64x64-baremetal
```

Environment as found: `spike` 1.1.1-dev; `riscv64-unknown-{elf,linux-gnu}-gcc` 13.2.0 present (so
`build_spike.sh` takes the non-`BAREMETAL_ONLY` path). `$RISCV` is **not** set by default — source
`env.sh` first. Q2 resolved.

### 6.2 The two `gemmini_params.h` differ — and it is benign for MX

`software/libgemmini/gemmini_params.h` and `software/gemmini-rocc-tests/include/gemmini_params.h` are
**different configurations**, not skewed copies:

| | libgemmini | gemmini-rocc-tests |
|---|---|---|
| `elem_t` | `int8_t` | `uint8_t` + `ELEM_T_IS_LOWPREC_FLOAT` |
| `acc_t` | `int32_t` | `uint64_t` |
| exp/sig bits | — | `ELEM_T_EXP_BITS 3`, `SIG_BITS 3`; `ACC_T_* 8/8` |
| `ACC_ROWS` | 1024 | 512 |
| MX scaling | — | `HAS_MX_SCALING` |

`libgemmini/gemmini.h:8` includes the **int8** header, and `elem_t`/`acc_t` appear 134× in
`gemmini.cc`. This is exactly the skew merlin documents as producing all-zero output
(`docs/guides/reproducing_whole_model_on_rtl.md:133`). **Empirically it does not bite the MX path** —
the test PASSes. Consistent explanation: MX operands are 8-bit codes, byte-compatible with `int8_t`
scratchpad storage, and the MX datapath accumulates in its own `mx_smem` (bf16) via `mx_fp_math.h`
rather than the int32 accumulator. Recorded as *verified for fp8 e4m3*, not proven in general —
re-check if an fp6-LUT or fp4 path ever returns zeros.

### 6.3 libgemmini staleness — fixed locally, `$RISCV` deliberately untouched

`libgemmini.so` (2026-05-28) predated `mx_fp_math.h` (2026-06-05); the Makefile lists only
`gemmini.cc` as a prerequisite, so `make` alone would not have rebuilt it. Forced a rebuild:
112304 → **123832 bytes**, so the binary genuinely differed.

**Differential run: both the stale and the rebuilt `.so` PASS this test.** So the staleness was not
implicated here, and the rebuild changed no result — it is hygiene, not a fix. It may still matter for
the fp6-LUT / fp4 paths, which this test does not exercise.

`$RISCV/lib/libgemmini.so` was **not** updated (`make install` not run) — it remains the 2026-05-28
build, so other work in this tree is unaffected by anything done here. Anyone running via `$RISCV`
rather than the in-tree path is on the older binary. Old artifact backed up at
`<scratchpad>/libgemmini.so.bak-20260528`; `*.so` is gitignored in that directory.

### 6.4 Follow-ups surfaced

- The `bareMetalC/Makefile` **does** track `$(GEMMINI_HEADERS)` as prerequisites (line 166), so the
  ELF rebuilds correctly on header changes — but only when invoked at the sub-make level. The
  top-level `build_spike.sh <test>` silently reports "Nothing to be done" for an individual test.
  Worth wrapping in `sim/` (Step 6) so this is not re-discovered.
- `libgemmini/Makefile` should list the headers as prerequisites; today a stale `.so` is silent.

## 7. Step 3 results — package resolves (2026-09-01)

Created:

```
README.md                                    repo overview + the two standing constraints
app/README.md  runtime/README.md  sim/README.md  compiler/README.md
compiler/targets/mx_gemmini_rocket/
├── README.md
└── contracts/target_contract.yaml           ~200 lines, every field provenance-tagged
```

**Verified** with `MERLIN_TARGET_PATH=$PWD/compiler/targets`,
`PYTHONPATH=$PWD/merlin/merlin/python`:

- `external_targets()` → `{'mx_gemmini_rocket': <pkg root>}`
- `all_targets()` → `['gemmini', 'muon', 'mx_gemmini_rocket', 'saturn', 'toy_npu']`
- `resolve('mx_gemmini_rocket').kind` → `external` (precedence 1, beats anything in-tree)
- all 8 schema-required top-level fields present
- `compute_units()` parses: unit `mx_systolic_mesh`, kind `systolic`, scaling `block_e8m0`, dtypes
  `(mxfp8, mxfp6, mxfp4)`, three `AccumRule(*, *, acc='bf16')`, three semantic capabilities
- all five referenced format names resolve in the registry; `mxfp8/6/4` come back
  `kind=mx_block`, `Scale(kind='block_e8m0', block=32)` — matching what the contract declares

merlin is importable straight from `merlin/merlin/python` (no venv needed for the registry path).

### 7.1 Corrections this step forced

- **`XCUSTOM_ACC` is 3, not 2.** `gemmini_params.h:7` defines it as 3 → RISC-V custom3 → opcode
  `0x7B`. `libgemmini/README.md` says "custom-2"; that is a documentation error. §4 corrected.
- **Funct 26 is `CONFIG_SCALE_MEM`** in `gemmini.h:62`, not `MXQUANT_CONFIG_MVOUT` as the README
  names it. The macro that emits it is `gemmini_mxquant_config_mvout`. Header wins.

### 7.2 Gotchas worth not re-discovering

- **`scaling:` takes a SCALE KIND, not a format name.** The schema prose says it "REFERENCES
  quant_formats.registry.yaml by name", which is misleading. `e8m0` is a registered *format* and is
  **rejected**; the accepted vocabulary (`compute_units.py:187`) is `block_affine | block_e8m0 |
  kquant_superblock | none | nvfp4_block | per_channel | per_group | per_tensor`. Use `block_e8m0`.
- **No `plugin:` block until `backend/` exists.** Every plugin reference must resolve to a file
  inside the package; an unresolvable pointer is an *error*, not a silently-ignored key. Added in
  Step 4 alongside the module it names.
- **`datatype_tokens(unit)` takes one unit**, not a list and not a contract.

### 7.3 Deliberate scope choice: ranks [2], not [2, 4]

The int8 `gemmini` contract declares `ranks: [2, 4]` because its RTL funct table carries
`LOOP_CONV_WS` (15) + configs (16–21), so a rank-4 im2col conv region is genuinely acceleratable —
and its own comment records that declaring `[2]` alone once made conv capsules score ineligible and
silently drop out of the ARR denominator. **The MX command set does not extend the conv path**
(functs 23–29 are scales/LUTs/spad/smem only), so this contract declares `ranks: [2]`. That is the
opposite call from the sibling contract, made deliberately: here `[2, 4]` would put capsules in the
denominator that this datapath cannot run. Revisit if MX ever gains a conv funct.

## 8. Step 4 results — emitter works, and the compiler/app line (2026-09-01)

The emitted kernel **reproduces the Step-2 golden exactly** on spike:

```
fp8 WS matmul test PASSED (no mismatches).
```

Verified for both operand paths — the `#include`-the-header path and the fully **baked** path
(307 lines of C with `A_in` / `B_in` / `A_scales_row` / `B_scales_col` / `C_out_bf16` inlined). The
baked path is the one Step 8 needs, so it was proven now rather than assumed.

### 8.1 The compiler/app split (user, 2026-09-01)

> "under `compiler/`, only include the generic elements specific for merlin. Everything that merlin
> would include in their own repo if they were supporting mxgemmini. In the `app/` section, include
> operand specific stuff, or anything else that is needed."

This settled an open design question and forced a mid-step refactor. The line:

| | `compiler/targets/mx_gemmini_rocket/` | `app/mxgemm_bringup/` |
|---|---|---|
| knows about | a **command buffer** | a specific matmul, its operands, its golden |
| input | `cb: dict` | a C data header (today), PyTorch tensors (Step 8) |
| output | C source string | a written `.c` + the experiment |
| files | `backend/mxgemm_emit.py`, `contracts/` | `build_fp8_64x64.py`, `header_operands.py` |

`header_operands.py` was originally written into `backend/` and **moved** — it parses one specific
gemmini-rocc-tests header, which is exactly the operand-specific material that does not belong in a
package merlin could upstream.

### 8.2 Convergence on merlin's backend convention

Prompted by "do the other targets in merlin follow this same approach?" — they do, consistently, and
the first draft did not. Surveyed:

| Backend | Entry point | Input |
|---|---|---|
| `gemmini` | `generate_driver(cb, *, mode="explicit") -> str` | cb dict |
| `muon` | `emit_kernel_cpp(cb, *, num_warps=4) -> str` | cb dict |
| `radiance` hand_v0 | `_plan(cb)` → `EmittedKernel` | cb dict |

Universal: **command buffer in, source string out**; tunables as keyword args, never a config
object; a private `_c_array`/`_carray` bake helper; a module-level error class; dataclasses used for
*outputs* (`EmittedKernel`, `ResultBuffer`), not for input config; the package `__init__.py` calls
`register(BackendInfo(name, TargetClass.NPU, BackendKind.KERNEL, __name__))`.

Refactored to match: the first draft took a caller-supplied `MxGemmConfig`; now `_plan(cb) ->
MxGemmPlan` **derives** it, mirroring `muon_codegen._plan` and the radiance backend's `_plan` (typed
here, where merlin returns a bare tuple). Public entry is `generate_driver(cb, *, transport=None)`.

**MX operands ride the cb as `mx_operands`.** The generic tensor table carries *decoded* values while
the datapath consumes raw codes plus a separate E8M0 block-scale stream that cannot be reconstructed
from them. This is not a workaround — it is the established MX approach (`muon_mx_codegen.py:13`
reads the same side-channel).

**`Transport` is ours, not merlin's.** No merlin backend has a transport seam. Kept because this
target's own V1/V2/V3 output modes vary at exactly that point; labelled as an invention in the
module so it is not mistaken for convention.

### 8.3 Deliberate non-correction, needs a non-square shape to settle

The reference walks B with the **same row stride it uses for A** (`j*DIM*M + k*DIM`) — reading B as
if laid out `[N][K]` though it is declared `[K][N]`. At M == N == K = 64 that is dimensionally
identical and the shipped data matches it. Reproduced **verbatim** rather than "corrected": changing
it would break the bit-exact gate for no verified gain. **Do not generalize past square until this is
resolved against a non-square case** — it is the kind of stride bug that does not fault, it returns
plausible wrong numbers (the same failure mode merlin documents at
`whole_model_on_accelerator.md`: a padding mistake that returned cos 0.9847). Flagged in the emitter.

### 8.4 Still open for Step 5

`transport_rocket.py` (the accumulator-mvout transport for the standalone `MX_ROCKET` config,
half-width `mx_chunk_id` drain) and the `runtime/` harness packaging. The numeric gate Step 5 was
meant to prove is already met on the spike transport; what remains is the second transport and the
packaging, plus a `sim/` wrapper for the build recipe in §6.1.

## 9. Open questions

- **Q1 — `radiance-kernels/` tracking. RESOLVED 2026-09-01:** no dependency. Not a submodule, not
  tracked, no build path into it. Read for ideas while authoring; cite in comments where an idea came
  from it. Step 4 reimplements from the Rocket-native `matmul_tiled_fp8_64x64.c` instead. A follow-up
  worth adding once code exists: a check that nothing under `compiler/`, `runtime/`, or `sim/`
  references the tree.
- **Q2 — toolchain.** `$RISCV` is unset in the session shell; `libgemmini.so` build status and
  `spike` availability unverified. Step 2 resolves this. Likely just needs the conda env activated.
- **Q3 — output path for the milestone.** Three drains exist on the Rocket side (accumulator mvout
  with `mx_chunk_id`; funct 28 `MX_READ_SMEM`; V1 internal-spad via `ex_write_to_spad=true`).
  `matmul_tiled_fp8_64x64.c` uses accumulator mvout. Per `../CLAUDE.md`, V1 is sim-verified as of
  2026-08-31 with `matmul_tiled_fp8_64x64_requant` PASSing. Start with accumulator mvout to match the
  chosen golden; keep V1 as the requant-output path in Step 7.
- **Q4 — dialect.** Whether `dialect_plan.yaml` can be synthesized from the contract for an MX
  datapath, or whether the MX ops need the hand-authored route the radiance `hand_v0/dialect.py`
  took. Deferred to Step 3; does not block Steps 4–5, which emit C directly from a command buffer.

## 10. Long-term direction: one config artifact, two consumers

**User, 2026-09-01:** eventually software emits a **JSON of configurations** that drives *both*
compilation *and* hardware generation. The §4 table is the near-term stand-in for that.

Implications to keep in view while building Steps 3–5, so we do not have to unpick them later:

- The §4 target contract is already half of this — it is the *compile-side* consumer. The missing half
  is the elaboration-side consumer (today: `GemminiMxFPConfigs.standaloneMxFPConfig` +
  `MxGemminiRocketConfig`, hand-edited Scala). `../mxgen/` is the existing hardware-generation path in
  this generator and is the natural place for that half to land.
- **Facts that appear on both sides must have one owner.** Today mesh geometry, dtypes, the E8M0
  group size, the scale-window base (`0x20000000`) and the LUT regmap base (`0x20010000`) are each
  written down in at least two places — `ConfigsFP.scala` and `gemmini_mx_rocket.h` — and kept in sync
  by hand. Those are exactly the fields the JSON should own and emit into both.
- This matches merlin's cardinal rule (derive, never hardcode) rather than fighting it: the JSON
  becomes the derivation source, and `target_contract.yaml` becomes a projection of it rather than a
  hand-authored twin.
- Practical near-term discipline: when Step 3 writes the contract, mark each field with whether it is
  (a) a projection of an RTL fact, (b) an ABI fact the RTL cannot ground, or (c) a compile-side-only
  choice. Group (a) is what the JSON will later own; keeping it identifiable now makes that migration
  mechanical.

Not in scope for Steps 1–10. Recorded so the contract is authored in a shape that can be generated.

## 11. Related plans

`../planning/mxgemmini_rocket_standalone_plan.md` (the RTL-side V1/V2/V3 output-mode work this
consumes), `../planning/fp8_bubble_and_perf_plan.md`, `../planning/acc_raw_debug.md`.
