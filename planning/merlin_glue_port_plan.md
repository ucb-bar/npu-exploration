# merlin as the glue: one repo, one ELF, any datatype

**Created** 2026-09-08 · **Status** plan agreed, Step 0 next.

**Goal (user, 2026-09-08):** merlin is the glue between the ML workload, the PyTorch scripts, the
compiler, ELF generation and the hardware. Everything the workload needs lives in **this** repo, so
nothing is left floating in `../software/gemmini-rocc-tests/`. A PyTorch kernel compiles to **one
ELF** for **any datatype**, and that ELF runs on spike and on the RTL.

## Decisions taken (user, 2026-09-08)

| # | Decision |
|---|---|
| D1 | **Depend on `gemmini.h` only.** `include/gemmini.h` + `rocc-software/` are the accelerator's ABI and stay external, exactly like `libgemmini`. Everything workload-side — generators, goldens, `mx_host.h`, the llama capture — moves into this repo. `gemmini-rocc-tests` keeps its hand-written tests as the RTL regression suite; we stop adding to it. |
| D2 | **Foundations first**: one quantizer, then the golden model. |
| D3 | **MXQuant is the only quantization strategy.** Every other convention and every seam implementation is deleted, not deprecated. |
| D4 | **Every kernel is a single ELF. No host-mediated seam.** A matmul→matmul junction runs on device (resident codes + resident scales); a non-matmul stage is emitted as C into the *same* ELF. numpy never sits between two stages again. |
| D5 | **Host glue is in scope**: `mx_host.h` is ported and emitted (RMSNorm, SiLU, softmax + causal mask, RoPE, MX quantize, bf16 conversions). |
| D6 | RTL simulators are out of scope for now (user, 2026-09-08). The C must stay MX_ROCKET-clean and build for it; running it is deferred. |
| D7 | **The grade is against MXQuant, not fp32.** User, 2026-09-08: *"i don't really care that much about difference with fp32. I care more about difference with the same mlps running on MxQuant."* fp32 measures the cost of the format, which is a known quantity and not the question. The question is whether the hardware reproduces the model the quantization work is actually done in. See §4.3. |

---

## 1. Hardware facts, re-verified 2026-09-08

The user's instruction was to distrust the older notes: spike and the RTL were rebuilt around
requantization and datatypes. Everything below was read from the current sources today, not carried
over. Where it contradicts an older plan, **this section wins**.

### 1.1 The format selector is three-way, not a 2-bit code

`gemmini.cc:460-464` reads from `config_ex` rs1:

| bits | field |
|---|---|
| `[11:10]` | `act_fmt` — A side |
| `[13:12]` | `wgt_fmt` — B side |
| `[15:14]` | `out_fmt` — output |
| `[6]` | `mx_fp8_altfmt` |
| `[5]` | `uselut` |

and a *runtime* flag `mx_lut_en`, set by `MX_LOAD_LUT` and cleared by the new **funct 30
`k_MX_LUT_DISABLE`** (`gemmini.h:73`, `gemmini.cc:2206`). The reachable formats
(`gemmini.cc:1183-1187, 1222-1224, 1356-1368, 1424-1433`):

| fmt code | altfmt | lut_en | format | operand storage |
|---|---|---|---|---|
| 0 | 0 | 0 | **E4M3** direct | 8-bit codes |
| 0 | 0 | 1 | **E4M3 quad** | 4-bit LUT indices, 8-bit codebook |
| 0 | 1 | (always) | **E5M2** | 4-bit LUT indices, 8-bit codebook |
| 1 | 0 | — | **E3M2** (fp6) | 4-bit LUT indices, 6-bit codebook |
| 1 | 1 | (always) | **E2M3** (fp6) | 4-bit LUT indices, 6-bit codebook |
| 2 | — | — | **E2M1** (fp4) direct | 4-bit codes |
| 3 | — | — | **BF16** — output only | 2 bytes/elem |

Output encoding is *symmetric*: only plain E4M3-single is written 8-bit; every LUT sub-format
writes 4-bit LUT indices (`gemmini.cc:1183-1188`).

### 1.2 `MX_LOAD_LUT` carries the codebook width

`gemmini_mx_load_lut_dt(dram, num_luts, sel, entry_bits)` — rs2 `[39:34] entry_bits`, `[33:32] sel`
(0=B, 1=A, 2=C), `[31:0] num_luts` (`gemmini.h:70-76`). 6 for FP6, 8 for the FP8 codebooks. The old
`gemmini_mx_load_lut` is the `entry_bits=6` back-compat wrapper.

### 1.3 The RTL is a family of elaborations

`ConfigsFP.scala:331-420` + `chipyard/GemminiConfigs.scala:49-95`:

| chipyard config | mesh formats |
|---|---|
| `MxGemminiRocketConfig` | E4M3 (`standaloneMxFPConfig`) |
| `MxE5M2GemminiRocketConfig` | E5M2 |
| `MxAllGemminiRocketConfig` | **{FP4, E3M2, E2M3, E4M3, E5M2}** in one mesh |
| `MxE4M3LutGemminiRocketConfig` | alias of the above |
| `MxFp4Only / MxE3M2Only / MxE2M3Only / MxE4M3Only / MxE5M2Only` | one format each |

Single-format builds pass an explicit `MxConfig` (`MxFloat.withConfig`) so the PE elaborates only
that format's decode. **"Any datatype" is therefore gated by the elaboration**, and the compiler has
to know which one it is targeting — this is exactly the "one config artifact, two consumers" idea in
`npu_exploration_bridge_plan.md` §16, now forced rather than aspirational.

### 1.4 The FP8 requantizer targets MXQuant's convention — the seam is dead

`chain_seam_hw_notes.md` §8, landed in both spike and RTL: the `- log2_pmax_floor` term is gone for
FP8, so a block max lands in `[1, 2)` and `16·|A|max·|B|max = 64` sits **4× under** the accumulator
bound. Consequences for this repo:

* `--seam weight` / `--seam rescale`, `CHAIN_EXP_SHIFT`, `code_shift_lut`,
  `rescale_for_next_gemm`, `WEIGHT_SEAM_TARGET_EXP`, `RESCALE_SEAM_TARGET_EXP`, `_emit_seam` and
  `mx_gemmini.chain_code_shift` are all **compensation for a hardware behaviour that no longer
  exists**. D3/D4 delete them.
* FP4 and FP6 keep the OCP shift (`out_pmax` 2 and 4) and their own requant rounding
  (`out_requant="model"`), §9.4 — so the format table is per-format, not one constant.

### 1.5 The drain is a flat spad→DRAM mvout

`matmul_tiled_fp8_64x64.c:146-160`: BF16 output lands in the internal scratchpad (F2c full-width
store, flag `0x38`) and is drained with `gemmini_extended_mvout` in `DIM`-row steps — "identical
instruction stream on Spike and RTL". Our backend still drains with `MX_READ_SMEM` (funct 28), which
works on spike but is not the unified path.

### 1.6 Zero-seam chaining exists in hardware

`LOOP_WS_REQUANT_TILED` (rs2 bit 10) deposits the requantized output in the operand-A *tiled* layout,
and `gemmini_mxquant_config_mvout_resident` (rs1 bit 63, `gemmini.h:247-257`) writes the E8M0 bytes
straight into the act-scale window already transposed. The next matmul issues **no A mvin and no
A-scale load**. Proven on device in `llama_attention.c` (`llama_layer_hw_plan.md` §8.1). This is what
D4 requires and what replaces `_emit_seam`.

### 1.7 Baseline still green

`run_kernel.py --kernel linear` on the rebuilt spike: **PASS**, 282 cycles, rel_fro 5.9122% —
unchanged from the recorded history. The FP8/bf16 path did not regress; it is simply far behind what
the hardware now does.

---

## 2. Where the work is: the two stacks

| | this repo (merlin path) | `gemmini-rocc-tests` (hand-written) |
|---|---|---|
| formats | FP8 E4M3 only; out bf16 or fp8 | all six + LUT codebooks; requant to each |
| chaining | host scale transpose + code-shift LUT | resident tiled codes + resident scales, no seam |
| host glue | numpy, **between ELFs** | `mx_host.h`, in C, **inside** one ELF |
| operands | `torch.randn` | real TinyLlama, 27/27 tests bit-exact |
| golden | **none** — fp32 tier only | `fp8/fp4/lut_*_matmul_model.py`, 0 mismatches vs spike |
| quantizer | `app/mxquant.py`, old convention | MXQuant, via this repo's `app/mxq_golden.py` |
| drain | `MX_READ_SMEM` | flat spad mvout, spike + RTL identical |

The dependency already runs backwards: `gen_matmul_llama.py` and `gen_llama_layer.py` live in
`gemmini-rocc-tests` but import `app/mxq_golden.py` from here. D1 straightens that out.

## 3. Target layout

```
app/            quantization (MXQuant only), interface MLIR, real-data capture
  mxformats.py    NEW  the per-format table: element format, pmax shift, requant mode,
                       fmt code / altfmt / lut, codebook width, packing
  mxq_golden.py        the ONLY quantizer -- wraps MXQuant, generalized past FP8
  mxquant.py           wire-format primitives ONLY (encode/decode tables, e8m0, bf16)
  mxiface.py           merlin_iface emission
  capture_llama_*.py   real operands
grade/
  golden/         NEW  the bit-exact datapath model (ported, unified from four copies)
  pipeline.py          one command buffer -> one ELF -> one run, always
compiler/targets/mx_gemmini_rocket/
  contracts/           + the elaboration table (which config supports which formats)
  backend/
    mxgemm_emit.py     all formats, LUTs, resident chaining, flat mvout, no seam
    mx_host/           NEW  the emitted C host runtime (ported mx_host.h)
    runner.py          spike today; MX_ROCKET-clean C
kernels/          registry: synthetic + real-llama kernels
```

External, unchanged: `../software/gemmini-rocc-tests/include/gemmini.h`, `rocc-software/`,
`../software/libgemmini/`.

## 4. Steps

| # | step | gate | state |
|---|---|---|---|
| 0 | **Freeze the baseline.** Record every registered kernel's current spike result so every later change is a differential, not an absolute claim | `results/baseline_20260908.json` | **DONE** 2026-09-08 — §4.1 |
| 1 | **One quantizer.** `mxq_golden` generalized past FP8 into `app/mxformats.py`; `app/mxquant.py` cut to wire-format primitives. Delete `TARGET_CODE_EXP`, `quantize_rows`, `quantize_matmul_operands`, `rescale_for_next_gemm`, `code_shift_lut`, `CHAIN_EXP_SHIFT` | `linear` bit-identical to Step 0 (FP8 operands were already `_po2`) | **DONE** 2026-09-08 — §4.2 |
| 2 | **Grade against MXQuant** (D7). `rtl_exact` moves from an offline artifact onto the live path: every kernel reports vs MXQuant-`rtl_exact` (the correctness gate) and vs MXQuant-as-shipped (the research delta); fp32 is demoted to context | `linear` **bit-identical** to MXQuant under `rtl_exact`, then the chains | **DONE** 2026-09-08 — §4.4 |
| 3 | **Seam removal + resident chaining.** Delete `_emit_seam` and the whole `--seam` surface; emit `LOOP_WS_REQUANT_TILED` + `mxquant_config_mvout_resident` instead | every kernel still **bit-identical to MXQuant** | **DONE** 2026-09-08 — §4.5 |
| 4 | **The drain, and dual-target C.** `MX_READ_SMEM` → flat spad mvout; C stays MX_ROCKET-clean | spike output bit-identical across the change; builds for MX_ROCKET | **DONE** 2026-09-08 — §4.6 |
| 5a | **FP4 E2M1** — the direct 4-bit format, no codebook. Real selector, packed operands, `--dtype` | reproduces `matmul_tiled_fp4_*`'s golden bit for bit, square AND non-square | **DONE** 2026-09-08 — §4.7 |
| 5b | **The LUT family** — E4M3-quad, E5M2, E3M2, E2M3. Codebook load/disable, `altfmt`, data-derived codebooks on the side channel | as 5a, against each format's shipped test | **DONE** 2026-09-08 — §4.8 |
| 5c | **Chained requant to a nibble format**. Reference: `matmul_tiled_fp4_64x64_chain.c` | a 2-stage FP4 chain's resident intermediate vs that test | **DONE** 2026-09-08 — §4.9 (LUT chain still refused, with cause) |
| 6a | **Port `mx_host.h`** as the backend's C runtime, on the compiler's include path | existing kernels bit-identical | **DONE** 2026-09-09 — §4.12 |
| 6b | **Declarative `HostStage`** — `op` + `params` naming both a Python twin and an `mx_host.h` entry point | attention bit-identical | **DONE** 2026-09-09 — §4.12 |
| 6c | **Emit host stages into the fused driver**: drained bf16 → host op → `mx_quantize_rows` → mvin | attention in **ONE** ELF, bit-identical to the 6-ELF path | **DONE** 2026-09-09 — §4.13 |
| 6d | **C ↔ Python twin gate** | 0 code / 0 scale mismatches | **DONE** 2026-09-09 (pulled early) |
| 7 | **Real-data kernels.** `llama_mlp` / `llama_attention` registered, built from the captured layer | reproduces the recorded rel_fro 11.59% / 13.42%, every stage bit-exact | **DONE** 2026-09-09 — §4.14 |
| 8 | **Retire the floating pieces.** Delete the rocc-tests generators whose capability now lives here; document what stays as reference | nothing under `app/`, `grade/`, `compiler/` reads a path into `gemmini-rocc-tests` except `include/gemmini.h` | **PARTIAL** 2026-09-09 — §4.15: the runtime dependency is gone; the generators still exist there |

D4's invariant — one ELF, no host seam — is *delivered* by Steps 3 and 6 and **enforced from then
on**: `grade/pipeline.py` loses its per-stage lowering entirely rather than keeping it as a fallback.

### 4.1 Step 0 results — the baseline, and what it already proves (2026-09-08)

`results/baseline_20260908.json`. The **gate is the output bits** (sha256 of `hardware_output.npy`);
cycles are recorded as context only — performance is explicitly not a concern for this port
(user, 2026-09-08).

| kernel | sha256[:16] | rel_fro | finite | ELFs | cycles | pass |
|---|---|---|---|---|---|---|
| `linear` | `8397a0229cacccee` | 5.9122% | 4096/4096 | 1 | 282 | yes |
| `mlp2` | `b2c612b57d8fdede` | 9.3423% | 4096/4096 | 1 | 1052 | yes |
| `mlp3` | `781fce1d43b10038` | 11.9607% | 4096/4096 | 1 | 1809 | yes |
| `mlp4` | `740a9abfc9547087` | 13.8512% | 4096/4096 | 1 | 2566 | yes |
| `mlp6` | `05c11f34b4fc3d8f` | 18.6801% | 4096/4096 | 1 | 4143 | no |
| `mlp8` | `22002bcafeff4720` | 22.6752% | 4096/4096 | 1 | 5662 | no |
| `attention` | `cb4765b92fc0cc80` | 14.3546% | 4096/4096 | **6** | 1692 | yes |

`mlp6`/`mlp8` FAIL at `--tol 0.15`, unchanged in kind from the recorded history: MX error
accumulating over N re-quantizations in a chain that deliberately has no normalization. Every value
is finite. `attention` is the one kernel still on N ELFs — D4 kills that in Step 6.

**The numbers moved since `npu_exploration_bridge_plan.md` §5c, and the pattern is diagnostic:**

| kernel | §5c | now | requantizes? |
|---|---|---|---|
| `linear` | 5.9122% | **5.9122%** | no — 1 stage, bf16 out |
| `attention` | 14.3546% | **14.3546%** | no — bf16 at every GEMM |
| `mlp2` | 9.5134% | 9.3423% | yes |
| `mlp3` | 12.4445% | 11.9607% | yes |
| `mlp4` | 15.0278% | 13.8512% | yes |
| `mlp6` | 20.5344% | 18.6801% | yes |
| `mlp8` | 25.7168% | 22.6752% | yes |

Exactly the requantizing kernels moved, all of them improved, and the two that never requantize are
bit-identical to their historical runs. That is independent confirmation that §1.4's MXQuant
migration landed in spike — and that the seam compensation Step 3 deletes is now **over-correcting**
rather than protecting anything. The per-stage telemetry agrees: `peak code 2`, `E8M0 121..128`,
i.e. a block max in `[1,2)`, which is the new convention and not the old `[256,512)` one.

Recorded as the number to beat: after Step 3, every `mlp*` must be finite and **no worse** than the
row above, and bit-exact against the Step-2 golden.

### 4.2 Step 1 results — one quantizer, and it is the baremetal tests' own (2026-09-08)

**User requirement (2026-09-08):** *"This quantizer should be exactly the same as MXQuant and the
same as the baremetal examples currently in gemmini-rocc-tests."* Both halves are now checked, not
asserted — `tests/selftest_quantizer.py`:

| header | shape | A codes | A scales | B codes | B scales |
|---|---|---|---|---|---|
| `matmul_fp8_32x32x32.h` | 32×32×32 | == | == | == | == |
| `matmul_fp8_64x64.h` | 64×64×64 | == | == | == | == |
| `matmul_fp8_96x32x32.h` | 96×32×32 | == | == | == | == |
| `matmul_fp8_64x96x64.h` | 64×64×96 | == | == | == | == |
| `matmul_fp8_96x96x64.h` | 96×64×96 | == | == | == | == |
| `matmul_fp8_128x128.h` | 128×128×128 | == | == | == | == |
| `matmul_fp8_128x128x256.h` | 128×256×128 | == | == | == | == |

**Byte-for-byte on all seven**, square and non-square. "It is MXQuant" holds by construction
(`quantize_mx_block32` is imported, never transcribed); "it is what the baremetal tests run" is now
a regression test rather than a belief. That is what makes the ELF our compiler emits and the ELF in
`gemmini-rocc-tests` the same program numerically, which is the premise of D1/D3.

#### What changed

* **new `app/mxformats.py`** — every format fact in one table: the MXQuant alias, `out_pmax`,
  `out_requant`, and the *three-part* hardware selector (`fmt_code`, `altfmt`, `lut`/`entry_bits`).
  Ported from `gen_matmul_llama.py:72-100` (fp8/fp6/fp4) and extended with the three LUT
  sub-formats that had no table anywhere. Also carries `ELABORATIONS` — which chipyard config can
  run which formats — and `check_elaboration()`, because **spike decodes every format regardless of
  config**, so passing on spike is not evidence the target build can run it.
  Only `fp8_e4m3` is marked `proven`; the rest **raise** rather than emit an unchecked datapath.
* **`app/mxq_golden.quantize_operand(V, side=...)`** — the one entry point the live path calls.
  `side` alone fixes both the blocking axis and the scale layout, so the two questions the seam used
  to answer implicitly are now answered at the call site.
* **`app/mxquant.py` → `app/mxwire.py`** — the module stopped quantizing, so it lost the name. What
  remains is decode-only (`fp8_e4m3_decode`, `e8m0_decode`, `bf16_bits_to_float`), which has no
  convention to disagree about. `app/mxquant.py` is now a 25-line shim so the generators still
  living in `gemmini-rocc-tests` keep importing; Step 8 deletes both.
* **deleted, not deprecated**: `TARGET_CODE_EXP`, `_shared_exponent`, `quantize_rows`,
  `quantize_matmul_operands`, `CHAIN_EXP_SHIFT`, `code_shift_lut`, `rescale_for_next_gemm`,
  `WEIGHT_SEAM_TARGET_EXP`, `RESCALE_SEAM_TARGET_EXP`, and the `--seam` flag.

#### The gate

| kernel | baseline | now | delta | bits |
|---|---|---|---|---|
| `linear` | 5.9122% | 5.9122% | +0.0000 | **IDENTICAL** |
| `mlp2` | 9.3423% | 8.9745% | −0.3679 | changed |
| `mlp3` | 11.9607% | 11.5640% | −0.3968 | changed |
| `mlp4` | 13.8512% | 13.6586% | −0.1926 | changed |
| `mlp6` | 18.6801% | 18.3452% | −0.3349 | changed |
| `mlp8` | 22.6752% | 23.3041% | **+0.6289** | changed |
| `attention` | 14.3546% | 13.6248% | −0.7298 | changed |

`linear` is **bit-identical**, which is the strongest available statement that the single-stage FP8
operand path did not move. Everything that requantizes improved by a consistent ~0.3 points, as
predicted: the next weight is no longer quantized 224× down to survive an overflow the hardware
stopped producing.

**`attention` changed even though it never requantizes**, and the reason is worth keeping: its
operands are *computed* values read back as bf16, and quantizing a bf16 value (8 mantissa bits) to
E4M3 (3) hits exact ties constantly — where the retired encoder rounded half-to-**even** and MXQuant
rounds half-**away**. On fp32 operand data ties are measure-zero, which is why the two encoders
agreed 0/8192 on random tensors and why `linear` is untouched.

**`mlp8` moved the wrong way, and it is seed noise — measured, not assumed.** Across seeds 0-3:
`mlp8` spans 22.69–23.59 (0.90 points) and `mlp6` spans 17.66–19.93 (2.27 points). A +0.63 change
sits inside that band. Stated honestly: **the fp32 tier cannot resolve a 0.6-point question at
depth 8.** It measures the cost of the format, not the correctness of the datapath, so it will never
settle this. Step 2's golden answers it with no tolerance at all, which is why it comes next.

Also re-run clean: `tests/selftest_grade.py` (24 checks), `app.mxq_golden` self-check (5 046 272
real TinyLlama elements, all losslessly encoded), and the `gemmini-rocc-tests` generators still
import through the shim.

### 4.3 Step 2, reframed: the golden IS MXQuant (D7, 2026-09-08)

The original Step 2 was going to port the four bit-exact mesh models
(`fp8_matmul_model.tiled_matmul_hwlike` and friends) into `grade/golden/`. **That is the wrong
golden.** It would answer "does the hardware match a Python model of the hardware", which is a
tautology one layer removed. The question the user actually has is whether the hardware matches
**the model the quantization work is done in** — MXQuant.

That comparison is already established and already lives in this repo. `rtl_exact/` (from
`llama_layer_hw_plan.md` §8.2b) is the MXQuant config under which `MXLinearSim` is **bit-identical**
to the datapath — re-verified 2026-09-08, 81 s for both configs:

```
as shipped         :  10.63% vs fp32    17.90% vs hardware
rtl_exact installed:  11.59% vs fp32     0.00% vs hardware
identical elements : 65536/65536   max abs diff 0
```

It is simply **not on the graded path**: `grade/pipeline.py` compares against `torch.matmul` in
fp32 and `metrics["tier"]` reads `fp32`. Step 2 wires it in.

#### Three tiers, and what each is for

| tier | compares against | answers | expected |
|---|---|---|---|
| **`mxquant_rtl_exact`** | `MXLinearSim` + `rtl_datapath.install()` | *is the hardware correct?* | **bit-identical**; any difference is a bug in the RTL, spike, or our codegen |
| **`mxquant_shipped`** | `MXLinearSim` as a researcher would configure it | *how far is the model I quantize in from the silicon?* | ~18% element-wise — the number worth tracking |
| `fp32` | `torch.matmul` | what MX costs at all | context only, demoted from the verdict |

The verdict becomes the first row: a **tolerance-free** gate, which is what Step 1's `mlp8` question
needed and the fp32 tier could never answer (§4.2 — a 0.63-point move inside a 0.90-point seed band).

#### Why it extends, and where it has to be re-proven

`rtl_exact` was verified on ONE workload: a llama MLP whose every matmul takes host-quantized
operands. Two of our kernels go beyond that, so each is its own gate rather than an assumption:

* **chained MLPs** — stage *i+1*'s A operand is the *device's requantizer output*, not a
  host-quantized tensor. The golden has to requantize between stages too. That is expressible
  (FP8 requant is exactly MXQuant, byte-identical, `chain_seam_hw_notes.md` §9.4) but it is an
  extension of what was checked. Gate `linear` first, then `mlp2`, then the depth sweep.
* **attention** — a host softmax in fp32 sits mid-graph; the golden must mirror it.

#### Cost

~40 s per MXQuant run at 12.6 M MACs, so `linear` (262 K) is well under a second and `mlp8`
(2.1 M) a few seconds. Both tiers on every kernel is affordable. The `rtl_exact` README's warning
stands — `fp_quantize_rne` and `fp_add_exact` are scalar Python loops — and it becomes the binding
constraint at llama scale (Step 7), not here. Vectorizing them, checked elementwise against the
scalar versions, is the fix when it bites.

### 4.4 Step 2 results — every kernel is bit-identical to MXQuant (2026-09-08)

`grade/mxquant_ref.py` walks a `KernelSpec` and runs each mesh stage through MXQuant's own
`MXLinearSim`, twice — once with `rtl_datapath.install()` and once as shipped. `grade/pipeline.py`
feeds both into `metrics.compare`, and the verdict is now **bit-identity with no tolerance in it**.

| kernel | hardware vs MXQuant/`rtl_exact` | MXQuant as shipped, vs hardware | fp32 (context) |
|---|---|---|---|
| `linear` | **4096/4096, max\|d\| 0** | 7.05% | 5.91% |
| `mlp2` | **4096/4096, max\|d\| 0** | 12.17% | 8.97% |
| `mlp3` | **4096/4096, max\|d\| 0** | 16.33% | 11.56% |
| `mlp4` | **4096/4096, max\|d\| 0** | 19.75% | 13.66% |
| `mlp6` | **4096/4096, max\|d\| 0** | 26.90% | 18.35% |
| `mlp8` | **4096/4096, max\|d\| 0** | 33.46% | 23.30% |
| `attention` | **4096/4096, max\|d\| 0** | 18.97% | 13.62% |

Also swept `_mlp(depth)` for **every depth 2–8**: all bit-identical.

Three things this establishes that nothing here could state before:

1. **The compiler is correct**, on every kernel shape we can express. The emitted C, built by gcc
   and run on spike, computes exactly what MXQuant predicts — through fused chains up to depth 8,
   through a requant seam at every junction, and through attention's transposed and computed
   operands plus its host softmax. That is a real gate: `linear`'s 4096 elements agreeing to the
   last bit is not something a threshold could have told us.
2. **Step 1's `mlp8` question is closed.** The +0.63-point fp32 move was format noise, exactly as
   the seed spread suggested. The hardware is exact at every depth.
3. **`mlp6` and `mlp8` now PASS.** They FAILed the old `--tol 0.15` fp32 gate, which was the gate
   being wrong, not the hardware: the error it measured was the format's, and the datapath is
   bit-perfect underneath it.

**The model-vs-silicon gap grows with depth** — 7.05% at one matmul to 33.46% at eight. This is the
number D7 asked for, and it is the one worth watching: an MXQuant accuracy result on a deep stack
describes something increasingly unlike what the chip computes, because the three unmatched
behaviours (`rtl_exact/README.md`) compound across re-quantizations. It costs `rtl_datapath.install()`
to remove.

#### A false alarm worth recording

An intermediate probe reported `mlp6` and `mlp8` diverging by ~150%, which read like a depth-6 chain
bug. It was the **probe**, not the hardware: it compared a freshly built seed-0 spec against the
newest `results/` directory for that kernel, which was a leftover from Step 1's seed sweep — seed 3.
Two different models. `mlp4` looked clean only because it had not been seed-swept.

The fix is structural, not a note to be careful: `pipeline.py` builds both references **from the
spec it just ran, in-process**, and never reads a saved artifact. The general lesson is the one the
repo already learned twice (§10.2, §14.1) — a comparison against a stored artifact is only as good
as the provenance of that artifact, and here the artifact carried no seed in its name.

#### Cost

`linear` 1.1 s, `mlp8` ~8 s, `attention` 6.5 s for the `rtl_exact` run; as-shipped is milliseconds.
Both references on every graded run is affordable at these shapes. The scalar-Python primitives will
bind at llama scale (Step 7), as `rtl_exact/README.md` warns.

### 4.5 Step 3 results — the seam is now literally empty (2026-09-08)

Two hardware knobs, both set on the PRODUCING stage, replace every byte of host work between two
matmuls:

* `gemmini_mxquant_config_mvout_resident` — rs1 bit **63**, `MX_SCALE_RESIDENT`. The requantizer
  also writes the output block-scales into the on-chip act-scale window, already transposed to
  `[GN][M]` (`gemmini.cc:1325`, `a_off_out = bi*M + m`). It still honours the DRAM address, so host
  telemetry costs nothing.
* `LOOP_WS_REQUANT_TILED` — rs2 bit **10**. Commits the codes in the block-tiled operand-A layout,
  whose address `gemmini.cc:1199` documents as *identical* to the operand-A read.

So a consuming stage points `a_base` at the previous stage's `spad_dest` and issues **no A mvin and
no A-scale load**. The emitted C for a junction is now a comment:

```c
  /* ---- 0 -> 1: NOTHING. Stage 0 committed its codes tiled at
     spad row 128 and its scales into the act-scale window, so stage
     1 reads both in place. This is the seam, and it is empty. */
```

**Gate — every kernel still bit-identical to MXQuant/`rtl_exact`**, which is the right bar: this
changes how data moves, not what is computed, so anything other than identity would be a bug.

| kernel | vs MXQuant | cycles before | cycles after |
|---|---|---|---|
| `linear` | 4096/4096 | 282 | 282 (no seam to remove) |
| `mlp2` | 4096/4096 | 1052 | **464** |
| `mlp3` | 4096/4096 | 1809 | **631** |
| `mlp4` | 4096/4096 | 2566 | **800** |
| `mlp6` | 4096/4096 | 4143 | **1136** |
| `mlp8` | 4096/4096 | 5662 | **1467** |
| `attention` | 4096/4096 | 1692 | 1692 (per-stage path, Step 6) |

Also re-checked non-square and degenerate shapes, all bit-identical: `linear 32x128x96`,
`mlp2 32,128,64,96`, `mlp3 --h 32` (a one-block intermediate), `mlp4 m=128`, `mlp3 32,64,128,64`.

Performance is explicitly not a goal here, but the cycle column is evidence the change did what it
claims: a chained stage went from 296 cycles + a 454-cycle seam to **171** cycles, because it no
longer moves its A operand in or loads its A scales at all.

#### What was deleted

`_emit_seam`, `_a_scales_name`, `_c_array_1d`, `CHAIN_CODE_SHIFT_ATTR`, `_SEAM_UNROLL_MAX`, the
per-seam cycle metrics, the `A{i}_scales` transpose buffers, and `MatmulStage.chain_code_shift`.

That last one matters beyond tidiness: `mx_gemmini.chain_code_shift` was this target's **only
extension to frozen grammar v0.1** — a target-namespaced commit attribute the grammar does not
define a mechanism for. Every module this front end emits is now plain `merlin_iface`.

#### Two things worth keeping

* **The B-tile overlap is now checked, not assumed.** Output regions grow up from `spad_dest`
  while B tiles grow down from the top of the scratchpad; `_assign_smem` raises if they would meet.
  An overlap would not fault — it would return a plausible wrong answer.
* **Intermediates are read back exactly once, after the measured region**, purely so the per-stage
  telemetry (peak code, E8M0 range) survives. The readback is a contiguous mvout plus a software
  de-tile, *not* a strided mvout: a strided de-tile makes the writer DMA emit whole 64-byte cache
  lines and zero-fill the gaps on RTL, corrupting the result at N >= 128
  (`matmul_tiled_fp8_64x64_chain.c:82` records the same finding).

### 4.6 Step 4 results — the drain now works on hardware, not just on spike (2026-09-08)

**The issue.** The emitter drained every result with `gemmini_mx_read_smem` (funct 28). The
baremetal tests do not: every use of it in `bareMetalC` sits inside an `#ifdef SPIKE_SIM`, and the
sequence that serves both substrates is a flat MVOUT loop. `matmul_tiled_fp8_64x64_requant.c:161`
says so directly — the mvout is "replacing SPIKE's `gemmini_mx_read_smem(mx_smem)`. One instruction
stream for both." We were emitting a spike-only variant of a sequence the reference had already
settled.

**The reference tests are the contract**, and matching them is the whole justification. An earlier
draft of this section led instead with a survey of which functs the generator's Scala decodes. That
was the wrong layer to argue from and it does not belong here: D1 is precisely the decision to
depend on `gemmini.h` and on these tests as the ABI, not on the generator's internals. A compiler
test that asserts a fact about the RTL starts lying the day the RTL changes.

**What we emit now**, transcribed from `matmul_tiled_fp8_64x64.c:146-160` (BF16) and
`..._requant.c:170-176` (FP8) — the same loop, differing only in the row count, because a spad row
holds `DIM` bytes and an element is `out_elem_bytes` of them:

```c
  /* Drain: bf16 in the internal scratchpad -> DRAM, 2 byte(s)/elem => 64*64*2/16 = 512 spad rows.
     Flat contiguous MVOUT (funct 3) -- identical instruction stream on Spike and RTL. */
  gemmini_fence();
  gemmini_config_st(16 * sizeof(uint8_t));
  { uint8_t *c_base = (uint8_t *)&C_hw[0][0];
    for (int r = 0; r < 512; r += 16)
      gemmini_extended_mvout(c_base + r * 16, 384 + r, 16, 16); }
```

Contiguous, never strided — a strided mvout makes the writer DMA emit whole 64-byte cache lines and
zero-fill the gaps on RTL, corrupting the readback at N >= 128.

`SpikeSmemTransport` is renamed `RoccSpadTransport`, since it is no longer spike-specific and no
longer touches smem.

**Gate 1 — the drain change is numerically invisible.** All seven kernels still 4096/4096 identical
to MXQuant/`rtl_exact`, max\|d\| 0. It changes how bytes are copied, not what was computed.

**Gate 2 — `tests/selftest_mx_rocket_build.py`, 15 checks, 0 failures.** For a single matmul, a
3-chain and a non-square chain it asserts that the generated C carries **no** `SPIKE_SIM`/`MX_ROCKET`
conditional, drains with a flat MVOUT and never funct 28, compiles under **both** `-DSPIKE_SIM` and
`-DMX_ROCKET`, and that both targets receive byte-identical source. `compile_command_buffer` gained
`target=` (`TARGET_DEFINE`), mirroring how the reference tree builds `build_spike/` vs
`build_mx_rocket/`.

#### 4.6a The missing scale fence — an RTL-only bug spike could never have caught

Asked whether the ELFs are now RTL-runnable, a diff against the reference tests found one real
defect: **we emitted no `gemmini_fence()` after `MX_LOAD_SCALES`.** Every reference test does, and
`matmul_tiled_fp8_64x64.c:84` states the reason outright — "The fence orders the async scale DMA on
the RTL and is a no-op on Spike, so both emit the identical instruction stream."

So on spike the scale load is instantaneous and the mesh always sees the right scales; on the RTL the
DMA is asynchronous and `LOOP_WS` could begin before they land. A wrong answer on hardware, a perfect
score in every test we can run. Fixed; all seven kernels remain 4096/4096 identical to MXQuant, which
is the expected result — on spike the fence *is* a no-op, so bit-identity is the confirmation that
nothing else moved.

This is the argument for diffing against the reference tests instruction by instruction rather than
only matching their shape, and it is worth repeating for each format in Step 5.

**What this does NOT establish.** Nothing was RUN on the RTL — no simulator is built here (D6). The
ELFs compile for `-DMX_ROCKET` and their instruction stream now matches the reference tests', but
"matches a test that is itself only compiled here" is the honest ceiling: `llama_layer_hw_plan.md`
§6 records the same limit for the hand-written kernels. Two hazards are avoided *by construction*
rather than demonstrated — the async scale DMA above, and the strided-mvout cache-line fill that
corrupts readback at N >= 128.

### 4.7 Step 5a results — FP4 E2M1 runs, gated on the reference's own golden (2026-09-08)

**The gate (option (b), user's choice):** take the shipped test's operands verbatim, emit OUR C for
them, run on spike, require our output to equal its `C_out_bf16`. `tests/selftest_formats.py`:

| format | shape | result |
|---|---|---|
| `fp8_e4m3` | 64×64×64 | 4096/4096 identical to `matmul_tiled_fp8_64x64`'s golden |
| `fp4_e2m1` | 64×64×64 | 4096/4096 identical to `matmul_tiled_fp4_64x64`'s golden |
| `fp4_e2m1` | **128×512×128** | **16384/16384** identical to `matmul_tiled_fp4_128x128x512`'s golden |
| `fp4_e2m1` | 128×128×128 | 16384/16384 identical to `matmul_tiled_fp4_128x128`'s golden |

The non-square row is the one that matters. At K=512 against M=128, an A row stride of M instead of
K is catastrophically wrong — so that case is what actually validates the packing, and the shipped
64×64×64 test could never have.

#### The packing was derived from the model, not copied from the test

`matmul_tiled_fp4_64x64.c:106` uses `config_ld(MATMUL_M)` for A, which is only correct because
M == K == 64 there — the same ambiguity §10.2 already cost this repo three bugs. The layout comes
from `gemmini.cc:1533-1541` instead:

```
A: spad[A_t + (m >> 1)][kk]   nibble (m & 1) ? high : low   -> two m-ROWS per byte
B: spad[B_t + kk][n >> 1]     nibble (n & 1) ? high : low   -> two n-COLUMNS per byte
```

So A packs **down its rows** to `[M/2][K]` and B packs **across its columns** to `[K][N/2]`. They are
different operations, indistinguishable by size at a square shape. The shipped header agrees:
`A_in_hw[32][64]`, `B_in[64][32]`. A row of packed A is still K bytes — packing halves the row
*count*, not the row *length* — so the stride is K.

#### What else FP4 changed

* **Mesh tiles are format-dependent.** The nibble formats march 32×32, not 16×16
  (`gemmini.cc:1519` `TM = TN = 32`), so `tiles_i`/`tiles_j` and the legality check are now driven
  by the format table. `--m 16` is refused for FP4 and accepted for FP8, correctly.
* **`config_ex` emits the real three-way selector**, including `altfmt` (rs1[6]) longhand when set,
  as `matmul_tiled_fp8_e5m2_64x64.c:47-64` does. 5b needs that.
* **Output width is counted in BITS**, not bytes: a requant output element is 4 bits for a nibble
  format, so `out_bytes` replaces `out_elem_bytes` wherever a total is needed.
* **The exact encoder is now derived from the decoder** (`app/mxwire.DECODERS`), so an element
  format's semantics are written down exactly once. `golden()` takes the encode/decode pair together
  — passing one without the other turned its losslessness assertion into a false alarm, which it
  duly raised on the first FP4 run.

#### Two things deliberately refused rather than emitted

* **Chained requant to a nibble format.** `mlp2 --dtype fp4_e2m1` raises, naming
  `matmul_tiled_fp4_64x64_chain.c` as the reference to port (Step 5c). The alternative was emitting
  a plausible-looking chain whose 4-bit packed intermediate nothing has checked.
* **The MXQuant tier.** `rtl_exact` was established on FP8 only, so a non-FP8 run logs
  `mxquant SKIPPED ... rtl_exact covers fp8_e4m3 only` and grades at the fp32 tier. FP4 measures
  ~29-31% vs fp32, in line with `chain_seam_hw_notes.md` §9.7's 35-41% on real llama data — and it
  FAILs the default `--tol 0.15`, which is an FP8-calibrated threshold, not a statement about the
  hardware. The correctness gate for FP4 is the table above.

All seven FP8 kernels remain bit-identical to MXQuant, and `selftest_quantizer` (38 checks),
`selftest_grade`, `selftest_mx_rocket_build` (15) all pass.

### 4.8 Step 5b results — all six formats (2026-09-08)

`tests/selftest_formats.py`, 9 cases, 0 failures. Each is: the shipped test's operands (and, for a
codebook format, its codebooks) -> our C -> spike -> compare to its `C_out_bf16`.

| format | shape | reference test |
|---|---|---|
| `fp8_e4m3` | 64×64×64 | `matmul_tiled_fp8_64x64` |
| `fp4_e2m1` | 64×64×64, **128×512×128**, 128×128×128 | `matmul_tiled_fp4_*` |
| `fp8_e5m2` | 64×64×64 | `matmul_tiled_fp8_e5m2_64x64` |
| `fp8_e4m3_quad` | 64×64×64 | `matmul_tiled_fp8_e4m3_lut_64x64` |
| `fp6_e2m3` | 64×64×64 | `matmul_tiled_fp6_e2m3_lut_64x64` |
| `fp6_e3m2` | 128×128×128, **128×512×128** | `matmul_tiled_fp6_128x128*` |

All bit-identical. **`--dtype` now works for every format from PyTorch**, which needed the codebooks
to be *built* rather than read from a header:

| dtype | wire | rel_fro vs fp32 (`linear` 64³) |
|---|---|---|
| `fp8_e4m3` | 8-bit direct | **5.91%** |
| `fp8_e4m3_quad` | 4-bit → 8-bit codebook | 13.95% |
| `fp6_e2m3` | 4-bit → 6-bit codebook | 14.44% |
| `fp8_e5m2` | 4-bit → 8-bit codebook | 15.72% |
| `fp6_e3m2` | 4-bit → 6-bit codebook | 15.79% |
| `fp4_e2m1` | 4-bit direct | 28.76% |

Two things in that table are worth keeping. The four codebook formats **cluster at 14-16% regardless
of their element width** — the binding constraint is the 16-entry codebook, not whether its entries
are 6-bit or 8-bit. And FP4, at the *same 4 bits on the wire*, is **~2× worse**, because its 16
values are fixed by the format while a codebook's 16 are fitted to the data.

#### New: `app/mxlut.py`

Ported from `gemmini-rocc-tests/llama_operands.py` (D1). MX-quantize first so every value is already
a valid element code, then reduce each group's code set to 16 signposts by deterministic weighted
1-D k-means (quantile init — most of a real block sits near zero, and a range-uniform init would
spend its slots on the sparse tail), snap the centroids back onto the format's own code set, dedupe
and pad. The codebook space is derived from `mxwire.DECODERS`, so it cannot disagree with what the
mesh does with an index.

**A codebook is compile OUTPUT that depends on the DATA** — the only such artifact in this compiler.
`quantize_operand` now returns `(codes, scales, codebooks)` and the backend refuses to emit a LUT
format without them rather than inventing one.

#### The bug, and what it says about the emitter's shape

The first LUT run gave zeros for E5M2/E2M3 and NaN for E4M3-quad. The *pattern* localized it before
any debugging: zeros mean an all-zero codebook, and NaN means `lut_en` was never set so E4M3 took the
**direct 8-bit** path over nibble-packed data. Both say MX_LOAD_LUT never executed.

Cause: `_emit_single` built its `_Names` from a hand-written literal instead of calling
`_stage_names`, so the codebook names were empty and `_emit_load_luts` returned nothing. The chain
driver, which does call `_stage_names`, would have been fine. Two constructors for the same record,
one of which silently lacked a field. `_emit_single` now goes through `_stage_names` like everything
else.

#### Also corrected: the LUT tile geometry

The table had E5M2 and E4M3-quad at 16×16. `gemmini.cc:1371` sets `TM = TN = 32` for the **whole**
LUT path — the mesh width follows the 4-bit *storage*, not the codebook's element width. Only plain
8-bit E4M3 is 16×16. Caught by reading the model rather than by a failing test, since the reference
tests' `tiles_I = MATMUL_M / 32` would have looked like an unrelated convention.

### 4.9 Step 5c results — the FP4 chain, gated on the resident intermediate (2026-09-08)

A chained commit to a 4-bit format now emits. The gate is the **resident intermediate** — stage 0's
requantized codes and block scales, read back out of the scratchpad — because that is exactly what
the resident chain produces, and it is what the reference test checks first:

| chain | result |
|---|---|
| `fp8_e4m3` | 4096/4096 codes + 128/128 scales identical to `matmul_tiled_fp8_64x64_chain`'s golden |
| `fp4_e2m1` | **2048/2048 codes + 128/128 scales** identical to `matmul_tiled_fp4_64x64_chain`'s golden |

2048, not 4096, is the point: a nibble commit packs **two m-rows per byte**, so the resident
intermediate is `[M/2][N]` — `gemmini.cc:1201-1210` puts it as "hw = m for FP8, m/2 for the nibble
formats, **exactly as operand A is packed**", which is what makes it readable in place as the next
stage's A operand. The shipped header agrees: `C1_out[32][64]`.

That drove the one real change: the emitter had been sizing every intermediate buffer, readback and
OUT line as `[M][N]` bytes, which is right only for an 8-bit format. `out_bytes`,
`out_byte_rows`/`out_byte_cols` replace it, and the reference reads its own back with the same dims
(`matmul_tiled_fp4_64x64_chain.c:151`: `mvout_detile(C1_hw, SPAD_DEST1, M/2, N)`).

FP4 chains at depth: 34.8% (mlp2), 37.8% (mlp3), 40.1% (mlp4) vs fp32 — the expected shape, growing
sublinearly as each re-quantization costs less than the first.

#### The LUT chain is refused, and the reason is structural

```
stage 0 requantizes to fp6_e3m2, a codebook format. A chained LUT commit requires stage i+1's A
codebook to equal stage i's C codebook, which means predicting the intermediate with a mesh model
for that format (reference: gen_fp6_chain.py). Not ported. Chain with fp8_e4m3 or fp4_e2m1, or
take bf16 out.
```

The requantizer writes indices into the **output** codebook (sel 2) and the next stage reads them as
**activation** indices (sel 1), so the two must be the same table. Choosing it means knowing stage
0's output before running it — i.e. a bit-exact mesh model for that format. MXQuant's `rtl_exact`
covers FP8 only, and the per-format models still live in `gemmini-rocc-tests` (Step 8 moves them).
Emitting a chain whose codebooks were a guess would produce plausible wrong numbers, which is the
failure mode this repo has hit three times; refusing costs nothing today, since no kernel here needs
a chained FP6.

### 4.10 Step 5c, part 2 — the codebook chain, and a datapath limit it exposed (2026-09-09)

**The emitter drives a codebook chain correctly**, and that is now proven the same way as the other
two — `selftest_formats.py`, 12 cases, 0 failures:

| chain | resident intermediate vs the reference's golden |
|---|---|
| `fp8_e4m3` | 4096/4096 codes + 128/128 scales |
| `fp4_e2m1` | 2048/2048 codes + 128/128 scales |
| `fp6_e3m2` | **2048/2048 codes + 128/128 scales** |

The FP6 case supplies the reference's own `C1_lut` — stage 0's OUTPUT book, which is also stage 1's
ACTIVATION book. `_mx_operands` now checks that coupling and refuses a mismatch, because index 7
meaning one value going in and another coming out produces a plausible wrong answer.

#### Choosing the codebook: the reference does not predict, and it does not have to

`lut_mapping_demo.py:489` builds every C book with `make_lut` — `torch.randn`, quantized, first 16
distinct values. It works because the requantizer divides by the block scale *before* projecting,
so a C book spans a NORMALIZED range known a priori. `gen_fp6_chain.py:209` does better, running the
mesh model and fitting the book to the actual requantized values. We do the middle thing: an fp32
matmul estimate, normalized with the format's own `out_pmax`, then k-means. Cheap, no mesh model,
and shaped to the kernel's data.

Getting it wrong costs accuracy, not correctness — whatever table is loaded, the hardware rounds to
its nearest entry and the next stage decodes with the same one.

#### THE FINDING: the nearest-entry finder aliases, and it bounds what can chain

The first attempt gave 77% error on a 2-stage FP6 chain. The cause is not in our code:

**The finder does not compare floats.** It converts the incoming code and every codebook entry to a
fixed-point magnitude and takes the smallest difference (`FP6E3M2NearestFinder.scala`,
`mx_fp_math.h::*_to_fixed_point`) — and those conversions apply a WIDTH MASK:

| E3M2 value | fixed-point | aliases |
|---|---|---|
| 4.0 | 64 | — |
| **20.0** | **64** | 4.0 |
| **16.0** | **0** | **zero** |
| 28.0 | 192 | 12.0 |

A book holding an aliased entry is not merely coarse, it is wrong: values near zero get assigned to
the entry whose fixed-point is 0 and decode as ±16. `app/mxlut.codebook_values` now filters the
value space to what the finder can distinguish — which puts E3M2's maximum at exactly **14**, the
same number the reference's own `C1_lut` tops out at.

That filter fixes the *book*, but it exposes the real limit. The requantizer normalizes output to
`[2**out_pmax, 2**(out_pmax+1))`, and for two formats that range starts **above** anything the
finder can represent:

| format | out_pmax | requant range | finder max | 2-stage mlp2 |
|---|---|---|---|---|
| `fp6_e2m3` | 2 | [4, 8) | 7.5 | **21.6%** — fits |
| `fp6_e3m2` | 4 | [16, 32) | **14** | 62% — aliases |
| `fp8_e5m2` | 16 | [65536, 131072) | **57344** | NaN — aliases |
| `fp8_e4m3_quad` | 0 | [1, 2) | 480 | 99% — **fits, yet fails; unexplained** |

`mxformats.chain_refusal` encodes this as a derived rule, so it cannot drift from the format table.
`grade/pipeline.py` refuses to *estimate* a book for those formats; the **backend does not**, because
given a valid book it is provably correct — which is why the FP6 chain gate above still runs.

**This is a datapath property, not a gap in this compiler — and it is MEASURED on the reference's own
artifact, not inferred.** Asked how these formats can fail when `gemmini-rocc-tests` ships a working
chained FP6 example on real llama data, the answer is that the shipped example has the same
behaviour and nobody had measured it:

```
reference FP6 chain (matmul_fp6_64x64_chain.h), its OWN C1
    vs exact arithmetic on its OWN operands:   rel_fro = 58.46%   corr = 0.818
```

Ours is 62%. Its test passes because it compares spike against a Python model that uses the same
finder and the same codebook, so both are wrong identically; `gen_fp6_chain.py` computes no accuracy
metric at all, and the baremetal test only checks codes against its own golden.

Note its `C1_lut` maxes at **14 with zero entries >= 16** — it already avoids the aliasing entries and
is still 58% off, because the requantizer normalizes into `[16, 32)` while the book cannot hold
anything above 14. Everything above 14 clamps. So the aliasing filter is necessary but not
sufficient; the range mismatch is the deeper fault.

Worth raising with the hardware owner: widening the finder's fixed-point, or lowering `log2_pmax`
for the chained-output case, would make E3M2 and E5M2 chainable.

**Refusing outright would be over-reach**, since studying this effect is a legitimate reason to run
one. It is an explicit opt-in instead: `--allow-lossy-chain` runs it and logs a warning naming the
cause. Off by default, because a silently-wrong chain is the failure mode this repo has hit
repeatedly.

#### 4.10a The LUTs are NOT mis-programmed — it is the requantizer's `out_pmax` (2026-09-09)

Asked whether the codebooks simply need programming better, the answer is measured and negative.
Best-possible 16 entries drawn from the finder-safe set, on the reference's own chain data:

| codebook | error vs the normalized target |
|---|---|
| the reference's own `C1_lut` | 27.77% |
| our k-means | **27.70%** |
| **best achievable 16 entries** | **27.70%** |

Our book is already optimal. No programming strategy helps, because the constraint is not the book.

**MXQuant's own LUT scheme never meets this problem**, and its pipeline says why
(`HW_complete_integration_e2e/lut_quantization.py`, a verbatim copy of
`microxcaling/mx/level2_scratch.py`):

```
_safe_lshift(out)      <- scale to mantissa space
_round_mantissa(out)   <- values are now valid FP6 floats
_quantize_level2(out)  <- LUT: reduce 64 FP6 values -> 16 signposts   [HERE]
_safe_rshift(out)      <- undo scaling
```

The LUT runs on values **already normalized by the MX block scale**, which under MXQuant's e2e
convention puts a block max in `[1, 2)`. Its codebook does list 16, 20, 24 and 28, but a k-means over
normalized values never selects them. And, stated outright in that file: **"LUT is weights-only. The
branch explicitly skips activations."** A chained intermediate IS an activation, so chaining a
codebook format is outside the methodology MXQuant validates.

**The fix is one constant, and it is already on the books.** The hardware requantizer applies
`log2_pmax_floor = 4` for FP6, pushing its output into `[16, 32)`. Sweeping it on the reference's own
data:

| `out_pmax` | normalized range | fraction above the finder's max (14) | codebook error |
|---|---|---|---|
| **4** — today | [-31.1, 31.8] | **16.33%** | **27.70%** |
| 2 | [-7.8, 7.9] | 0.00% | **11.12%** |
| 0 — MXQuant's convention | [-1.9, 2.0] | 0.00% | 11.17% |

Halving the codebook error, from a single constant. And this is exactly the migration
`chain_seam_hw_notes.md` §8 performed **for FP8 only** — §9.4 records that "FP4/FP6 were never
migrated to MXQuant's convention". So *finishing that migration for the nibble formats* is what makes
chained FP6 work, and it is the same change that fixed FP8's chain seam.

Recommendation for the hardware owner: set the requantizer's `log2_pmax_floor` to 0 for the FP6/FP4
output paths, as FP8 already has. Widening the finder's fixed-point would also work but is a much
larger change, and MXQuant's convention is the one the quantization work is done in anyway.

#### 4.10b RESOLVED: one constant explains every chain failure (2026-09-09)

The "unexplained" `fp8_e4m3_quad` was a **wrong value in our own format table**. The RTL has a SECOND
override table that the first does not mention (`MxRequantizer.scala:184-188`):

```scala
val log2_pmax_floor = MuxCase(log2_pmax_floor_raw, Seq(
  (format_reg === 0.U && altfmt_reg) -> 16.U,                 // E5M2
  (format_reg === 0.U && !altfmt_reg && lut_en_reg) ->  8.U,   // E4M3-quad
  (format_reg === 1.U && altfmt_reg) ->  2.U                   // E2M3
))
```

So E4M3-**quad** uses 8, while plain E4M3 uses 0. We had 0, built the codebook for `[1, 2)`, and the
hardware emitted `[256, 512)`. Corrected, and its chain now fails as NaN rather than 99% -- which
puts it in the same bucket as the others and makes the whole picture one story:

**`chain_seam_hw_notes.md` §1's accumulator bound is the single cause.** The mesh accumulates a
16-deep column at exponent width 4, saturating near 2^8. A chained stage feeds the requant output
(range `[2^p, 2^(p+1))`) against a fresh weight in `[1, 2)`, so it needs `16 · |A|max · |B|max <= 256`:

| format | `out_pmax` | \|A\|max | 16·\|A\|·\|B\| | vs 256 | finder ok | observed |
|---|---|---|---|---|---|---|
| `fp8_e4m3` | **0** | 2 | 64 | **under** | yes | bit-exact |
| `fp4_e2m1` | 2 | 8 | 256 | at the bound | yes | 34.8% |
| `fp6_e2m3` | 2 | 8 | 256 | at the bound | yes | 21.6% |
| `fp6_e3m2` | 4 | 32 | 1 024 | 4x over | **NO** | 62% |
| `fp8_e4m3_quad` | 8 | 512 | 16 384 | 64x over | yes | NaN |
| `fp8_e5m2` | 16 | 131 072 | 4.19e6 | 16 384x over | **NO** | NaN |

Every format with a non-zero `log2_pmax_floor` fails to chain, by accumulator overflow, by finder
aliasing, or both. The only one that chains cleanly is the one §8 already migrated to 0. FP4 and E2M3
sit exactly ON the bound and survive on sign cancellation -- the same "measured-safe, not
bounded-safe" situation §1a called out and fixed once before.

**So `out_pmax != 0` is not a per-format tuning choice; for a CHAINED output it is a bug.**

#### Chain support, as it stands

| dtype | single matmul | chained |
|---|---|---|
| `fp8_e4m3` | yes | **yes** (bit-exact vs MXQuant) |
| `fp4_e2m1` | yes | **yes** (34.8% mlp2) |
| `fp6_e2m3` | yes | **yes** (21.6% mlp2) |
| `fp6_e3m2` | yes | refused — finder aliasing |
| `fp8_e5m2` | yes | refused — finder aliasing |
| `fp8_e4m3_quad` | yes | refused — unexplained |

### 4.11 VERIFIED on spike: `log2_pmax_floor = 0` fixes chaining for all six formats (2026-09-09)

The RTL and spike were both changed — `MxRequantizer.scala:37` `log2_pmax_floor = 0.U` with the
per-(format, altfmt) override table removed at `:180`, and `gemmini.cc:1450,1578`
`const int log2_pmax = 0`. The `matmul_fp4_*` / `fp6_*` / LUT headers were regenerated. Re-tested:

| dtype | single matmul | chain BEFORE | chain AFTER |
|---|---|---|---|
| `fp8_e4m3` | 5.91% | 8.97% | 8.97% (already p=0) |
| `fp8_e4m3_quad` | 13.95% | **99%** | **20.41%** |
| `fp8_e5m2` | 15.72% | **NaN** | **24.56%** |
| `fp6_e3m2` | 15.79% | **62%** | **25.15%** |
| `fp6_e2m3` | 14.44% | 21.60% | 21.73% |
| `fp4_e2m1` | 28.76% | 34.80% | **46.02%** ← regressed |

Every chain now produces a sensible number, sitting above its own single-matmul error by roughly one
requantization step. All six are marked `chain_proven`; `chain_refusal` is vacuous and the
`--allow-lossy-chain` escape hatch is no longer needed by anything.

**Regression checks green**: all FP8 kernels still bit-identical to MXQuant/`rtl_exact`,
`selftest_formats` 12/12 against the REGENERATED goldens (so those passes are meaningful, not stale).

#### But p=0 over-corrects for the narrow formats — p=1 is the better constant

Requantizing the same stage-0 output at each `p`, measuring element error and how much of the code
ladder is reachable:

| | p=0 (now) | p=1 | p=2 | accumulator margin |
|---|---|---|---|---|
| `fp4_e2m1` err | **23.44%** (5/8 levels) | **13.26%** (7/8) | 11.61% (8/8) | 4x / 2x / **1x** |
| `fp6_e2m3` err | 5.86% (17/32) | 3.44% (25/32) | 2.80% (32/32) | 4x / 2x / **1x** |
| `fp8_e4m3` err | 2.66% | 2.66% | 2.66% | 4x / 2x / **1x** |

Three things follow:

1. **For FP8 the constant is irrelevant** — 2.66% at every `p`, because 3 mantissa bits plus a wide
   exponent make the subnormal runway long. So `p = 0` was free for FP8, which is why §8's migration
   cost nothing and why nobody noticed the trade.
2. **For FP4 it is expensive.** E2M1 has only 8 magnitudes; normalizing a block max into `[1, 2)`
   leaves just 5 of them reachable, so the intermediate is effectively ~2-bit. That is the whole of
   the 34.80% -> 46.02% regression.
3. **`p = 2` is numerically best and is exactly ON the accumulator bound** (`16·2^3·2 = 256`), which
   §1a already established is not safe — it survives on sign cancellation and breaks
   data-dependently at depth. That is what FP4 and E2M3 were doing before, and why they "worked".

**`p = 1` would satisfy both** on an accuracy-vs-fp32 basis: `16·2^2·2 = 128`, a 2x margin, FP4 back
to 13.26% element error. That was the initial recommendation.

#### RETRACTED: p = 0 is correct, because MXQuant uses p = 0 for FP4 too

The recommendation above optimizes the wrong objective. D7 is that the hardware should reproduce
**MXQuant**, not minimize error against fp32 — and MXQuant's block scale is
`_compute_block_scales -> _po2(amax)`, which takes **no format argument**:

```python
def _po2(x):                                   # mx_block_quant.py
    exp = torch.floor(torch.log2(torch.clamp(x.abs(), min=eps)))
    return torch.pow(2.0, exp)                 # docstring says "nearest"; it is FLOOR
```

Measured across every format, the quantized block max lands at exactly 2.0 — `p = 0` universally,
FP4 included. And MXQuant's FP4 shows the **same 5 of 8 reachable magnitudes** we measured on the
hardware at p=0. So the coarseness is not a hardware artifact; it is what the format does under this
convention.

Confirmed end to end. MXQuant, format-only (exact arithmetic, no mesh model), same kernel:

| | MXQuant single | MXQuant chain | hardware single | hardware chain |
|---|---|---|---|---|
| FP4 E2M1 | 29.95% | **43.73%** | 28.76% | **46.02%** |
| FP6 E3M2 | 7.37% | 10.58% | 15.79% | 25.15% |
| FP8 E4M3 | 3.57% | 5.09% | 5.91% | 8.97% |

**FP4 now matches MXQuant** — 43.73% vs 46.02%, the residual being mesh arithmetic (§9.7 measured
FP4's mesh error at ~0.05%, so almost all of the format's loss is the format). The 34.80% -> 46.02%
move is therefore **convergence, not regression**: the old `p = 2` was more accurate than MXQuant,
which for our purposes is the wrong kind of different.

**DECIDED (user, 2026-09-09): keep `log2_pmax_floor = 0` for every format, as the RTL now ships.**

The FP4 case for `p = 1` is recorded below as *exploration*, not as a pending change. Measured, model
validated against hardware (it reproduces the old `p = 2` hardware number of 34.80% exactly, and
`p = 0` to within the 2.7 points of mesh arithmetic):

| FP4 chain | p=0 (shipped) | p=1 | p=2 (old, ON the bound) |
|---|---|---|---|
| `mlp2` | 43.35% | 36.36% | 34.80% |
| `mlp4` | 59.32% | 44.86% | 40.32% |
| `mlp8` | **96.59%** | **63.29%** | 54.13% |

The benefit compounds with depth — at depth 8, `p = 0` loses the signal entirely while `p = 1`
recovers 33 points, with a 2x accumulator margin still in hand. If FP4 chains at depth ever matter,
this is the constant to revisit, and the same reasoning would apply to MXQuant's own `_po2` (which is
format-independent: harmless for the 8-bit formats, where `p` is irrelevant at 2.66% flat, but poor
for FP4). Not acted on now.

#### A separate divergence this exposed: FP6's hardware error is ~2x MXQuant's

FP6 E3M2 is 15.79%/25.15% on hardware against 7.37%/10.58% in MXQuant — and unlike FP8's gap, this is
not mesh arithmetic. The hardware's FP6 operands are **4-bit indices into a 16-entry codebook**, so
there is a second, coarser quantization MXQuant's format-only path does not have. MXQuant does have
that level-2 LUT step, but `lut_quantization.py` applies it to **weights only** and "explicitly skips
activations", whereas the hardware forces it on both operands.

That one cannot be closed by a constant. Either MXQuant would need to LUT-quantize activations to
predict this datapath, or the hardware would need an 8-bit-operand FP6 mode. Recorded rather than
resolved.

### 4.12 Step 6a/6b/6d — the host runtime is in, and validated (2026-09-09)

**6a.** `mx_host.h` ported to `compiler/targets/mx_gemmini_rocket/backend/runtime/`, with `runner.py`
putting that directory on the compiler's include path. It is the backend's C **runtime**: the fp32
host side of a layer (RMSNorm, SiLU, SwiGLU, RoPE, softmax, the transpose) plus the MX quantizer
that hands a host result back to the mesh. Self-contained — only `stdint/stddef/math.h`.

**6d, pulled early** because everything in 6c stands on it. `tests/selftest_mx_host.py` compiles the
header with the NATIVE compiler and compares its quantizer against the Python twin:

| case | codes | scales |
|---|---|---|
| A 64×64 (rows) | **0/4096** | **0/128** |
| A 32×128 (rows) | **0/4096** | **0/128** |
| B 64×64 (cols) | **0/4096** | **0/128** |
| B 128×96 (cols) | **0/12288** | **0/384** |

Byte-identical. Worth having as a standing test rather than a one-off: two implementations of one
quantizer is a drift risk, and the split is unavoidable (one runs on the device, one has to run in
the reference pipeline).

*A false alarm from that test, kept because the reasoning generalizes:* it first reported 0/4096
codes but ~40% scale mismatches on the row side. Codes agreeing while scales disagree is
**impossible** — the codes ARE `v/scale` — so the test had to be wrong, not the header. It was
transposing one side: `mx_quantize_rows` writes `scales_a[g*M + m]`, i.e. `[GK][M]`, already the
layout the A-side scale memory indexes and exactly what `quantize_operand(side="a")` returns.

**6b.** `HostStage` is now declarative: `op` names an entry in `app/mxhost.OPS`, which carries the
Python twin and the `mx_host.h` function to call. A closure can be *run* but not *compiled*, so a
kernel built from closures can only ever execute on the host BETWEEN ELFs — which is the thing D4
abolishes. `fn=` still works for an op with no C twin, and `HostStage.emittable` says which case a
stage is, so an un-emittable kernel is confined to the per-stage path rather than silently mis-fused.

`app/mxhost.py` holds the twins (`softmax`, `rmsnorm`, `silu`, `swiglu`, `transpose`). `get()`
validates parameter names and **fails closed on a typo** — silently ignoring an unknown `scale=` is
how a kernel quietly computes the wrong thing.

Two things this step also fixed:

* **`mx_softmax_rows`** generalizes the C softmax to `[M][N]` with a `causal` flag;
  `mx_softmax_causal` is kept as the M×M wrapper so Step 7's llama kernel is unchanged. Our
  synthetic `attention` is non-causal and non-square, and the shipped C only did causal M×M.
* **`transpose` is registered as a host op**, with the reason recorded: the MX loop path IGNORES
  `A_transpose`/`B_transpose` (`mx_loop_ws_spad` does `(void)rs1;`), so `S = Q@Kᵀ` needs a real byte
  transpose on the scalar core.

All kernels bit-identical to MXQuant afterwards; `selftest_grade`, `selftest_mx_host` (4/4),
`selftest_formats` (12/12) green.

### 4.13 Step 6c — attention is ONE ELF (2026-09-09)

```
[lower   ] 7 step(s) -> ONE command buffer via the GRAPH path (6 mesh, 1 host, 5 baked operands)
[host    ] 4 (P) (64, 64)  softmax(S/sqrt(d)) -- no reduction on mesh
VERDICT  PASS  hardware == MXQuant/rtl_exact  (4096/4096 identical, max|d| 0)
           lowering=graph   elfs=1
```

**Bit-identical to the per-stage path** — same output hash `273f3febceef73cf`, same 13.6248% vs
fp32 — which is the right bar: fusing changes how many programs run, not what is computed. **Every
kernel in the repo is now a single ELF**, and numpy never sits between two stages. D4 is met.

#### A separate emitter, on purpose

`mxgraph_emit.py` is new rather than a generalization of `mxgemm_emit`. The chain emitter keeps
intermediates in the scratchpad and has bit-exact gates against three reference chain tests; growing
a DAG into it would have put all of that at risk for no gain. `generate_driver` dispatches on the
`graph` side channel, so the backend still has one entry point.

Its structure is transcribed from `bareMetalC/llama_attention.c` — reusable helpers (`mvin_A`,
`mvin_B`, `mvout_bf16`, `mesh_matmul`) plus a linear sequence of calls, one per step. That shape is
what makes an arbitrary DAG tractable: each step names its own buffers, so nothing depends on
stage *i+1* consuming stage *i*.

#### Every edge goes through host fp32 memory, and that is the hardware's answer

A mesh output is drained as bf16, converted, and re-quantized before its next use
(`_emit_uses`). For attention this is not a shortcut: every seam except `P@V -> O@Wo` carries a host
op (softmax here, RoPE too in the llama version), so those values must reach the scalar core anyway.
`llama_attention.c` is built the same way and uses the resident seam only where no host op
intervenes. The remaining opportunity — using residency for `O@Wo` — is real but is Step 7's
business, since it needs the requant-to-spad path this graph emitter does not yet emit.

A value used on **both** sides is quantized twice, differently (A blocks along K by rows, B by
columns), which is why the side is part of every buffer's name. `K` is used as `b.T`, so it gets a
host byte transpose — the MX loop path ignores `B_transpose` (`mx_loop_ws_spad` does `(void)rs1;`).

#### Two traps, both already documented by the reference

* **`expf` undefined at link time.** `-lm` sat inside CFLAGS, i.e. BEFORE the sources, so it
  resolved nothing. Exactly `llama_layer_hw_plan.md` §8.4, fixed the same way: `-lm -lgcc` after the
  sources, and `mx_host.h` stubs the `__errno` newlib wants under `-DBAREMETAL`.
* **The run record lied.** It reported `lowering=per_stage, elfs=13` for a run that had gone through
  the graph path, because both fields were derived from `fused` alone. A metadata bug is worse than
  it looks — every later comparison reads that record — so `lowering` now names all three paths and
  `elfs` counts DISTINCT ELF paths rather than stage records.

Cycles are 2670 against the per-stage total of 1692, and that is honest rather than a regression:
the fused window spans all six matmuls **and** the softmax and every re-quantization, where the
per-stage sum counted only the six accelerator regions and did the glue in numpy for free. Mesh
steps also read 445 rather than 282 because each now moves its own operands in.

### 4.14 Step 7 — a real TinyLlama layer, from PyTorch to one ELF (2026-09-09)

Both real kernels run on spike, and both reproduce the hand-written reference kernels they replace:

| kernel | steps | ours | hand-written reference | vs MXQuant/`rtl_exact` |
|---|---|---|---|---|
| `llama_mlp` | 3 mesh, 2 host | **11.5928%** | `llama_mlp.c` 11.5933% | **65536/65536, max\|d\| 0** |
| `llama_attention` | 6 mesh, 4 host | **13.4157%** | `llama_attention.c` 13.4161%| **65536/65536, max\|d\| 0** |

Within **5 ppm** of the reference, and bit-identical to MXQuant. This is the goal of the whole port
stated concretely: a real decoder layer, defined by data captured from a real forward pass, compiled
through the merlin path into **one ELF**, graded against the model the quantization work is done in.

```
llama_mlp        [32][2048]  Xn(host) -> G -> U -> H(host) -> Y      (3 mesh, 2 host)  1 ELF
llama_attention  [32][2048]  Xn(host) -> Q,K,V -> Qr,Kr(host)
                             -> S -> P(host) -> O -> Y               (6 mesh, 4 host)  1 ELF
```

`d_model = 2048` is FULL, so RMSNorm and every projection input is an exact real llama quantity; the
slice is on the output side (64 of 5632 FFN neurons, 1 of 32 heads), and the fp32 reference is
truncated the same way, so it grades what the device actually computes (D1 of
`llama_layer_hw_plan.md`).

#### What Step 7 needed beyond 6c

* **Multi-input host ops.** SwiGLU takes gate AND up. `HostOp.arity`, `HostStage.src` as a tuple,
  and `HostStage` now rejects a mismatch between the two early rather than at emission.
* **N-chunking.** `H @ Wd` is `[32,64]x[64,2048]`, which needs 16512 scratchpad rows against 16384
  — 128 over. The emitter now halves N until it fits and emits a chunk loop, pasting each drained
  chunk into place on the host. The paste is deliberately a host copy, not a strided mvout: a
  strided de-tiling mvout makes the writer DMA emit whole cache lines and zero-fill the gaps on RTL.
* **fp32 constants.** RMSNorm's weight and RoPE's cos/sin tables are DATA the driver must carry but
  are not mesh operands, so they are baked verbatim as `const float` rather than quantized.
  Array-valued params are split out of `params` into `const_names` — leaving them in would make the
  op's own parameter validation reject the emitter's bookkeeping.
* **`mx_rmsnorm_f32` / `mx_rope_f32`.** `mx_host.h`'s versions take bf16 inputs and (for RoPE)
  cos/sin as uint32 bit patterns punned through a union — right for the reference kernel, wrong for
  a graph whose edges are already fp32 and whose tables are baked as floats. Same arithmetic, same
  indexing, so the Python twins still hold.

#### Two bugs worth keeping

* **`0f` is not a C literal.** `%.9g` drops the decimal point for whole numbers, so a baked 0.0 came
  out as `0f` — an integer with an invalid suffix, reported 4000 lines into a generated file.
  `_cfloat` now guarantees a decimal point or exponent.
* **RoPE's table indexing.** The Python twin sliced `cos[:, :half]` for both halves while the C
  reads `cos[m][i]` across the full width. Those agree only if the tables are duplicated across
  halves — they ARE in llama (checked on the capture, exactly equal) but depending on it would be a
  silent trap for any model where they are not. The twin now mirrors the C's indexing directly.

#### Cost

`llama_mlp` 15767 cycles, `llama_attention` 21400, dominated by the two `[32,2048]x[2048,64]`
projections at ~5250 each. Wall clock on spike: ~80 s and ~90 s, mostly the scalar host glue —
which is `llama_layer_hw_plan.md` §8.3's point restated: at this shape the accelerator is not the
bottleneck, the glue is.

### 4.15 The last runtime dependency is gone (2026-09-09)

Asked whether all of those scripts' functionality now lives here, the answer was **no**:
`rtl_exact/rtl_datapath.py` imported `fp8_matmul_model` from `gemmini-rocc-tests` for five
exact-dyadic primitives, and it is on the graded path — so **every** run depended on that tree at
runtime.

Fixed by **extracting** rather than transcribing. `app/mxarith.py` holds the transitive closure of
`mx_product_quantize_trunc`, `fp_quantize_rne`, `fp_add_exact`, `q_bf16_rne` and `bf16_accum_add`
— 15 definitions, 237 lines, pulled out by an AST walk in source order.

The previous cross-tree import bought a real property: one implementation, so it could not drift.
Copying it away would have lost that, so it is preserved as a **test**.
`tests/selftest_mxarith.py`:

* re-runs the extraction against the current upstream file and requires a **textual** match;
* runs both implementations over edge cases (subnormals, ±0, ±inf, NaN, exact ties, 2^-23, 448, 480)
  plus 4096 random values across six decades, requiring **elementwise identity**, NaN patterns
  included.

9 checks, 0 failures. `--update` re-extracts after a deliberate upstream change.

**Measured result:** a graded run now imports **0** Python modules from `gemmini-rocc-tests`
(checked by walking `sys.modules` after a build). `rtl_exact` still reports
`0.00% vs hardware, 65536/65536 identical`, and all nine kernels remain bit-identical to MXQuant.

#### What is still external, and why that is right

* `include/gemmini.h`, `rocc-software/`, `riscv-tests/` — the ABI and the baremetal build
  environment. D1's sanctioned dependency.
* the shipped `matmul_*.h` headers — the reference goldens `selftest_quantizer` and
  `selftest_formats` gate against. A dependency of the TESTS, deliberately: gating against someone
  else's proven artifact is the point.
* `software/libgemmini` — spike itself.

#### What was NOT ported, and the honest consequence

The per-format bit-exact mesh models (`fp4_matmul_model.py`, `lut_fp8_matmul_model.py`,
`lut_golden_model.py`, `golden_model.py`) and the header generators (`gen_matmul_llama.py`,
`lut_mapping_demo.py`, `gen_fp6_chain.py`, `gen_asym_*.py`) stay where they are, because nothing
here needs them — we do not emit those headers.

The consequence is real and worth stating: **the MXQuant correctness tier covers FP8 only.** The
other five formats are gated against the shipped goldens instead (`selftest_formats`, 12 cases).
Extending `rtl_exact` per format would need those models, and that is a research question — "does
MXQuant model FP4 the way this mesh does" — not a porting one.

`app/mxquant.py` remains as a deprecated shim so the generators still living there keep importing.

### 4.16 rtl_exact extended to every format (2026-09-09)

**Goal (user, 2026-09-09):** *"the goal is to get also an MXQuant implementation that matches the
hardware ... so that we can make it as analog to fp8 as possible."*

**Single matmuls: all six formats bit-identical to MXQuant/`rtl_exact`.**

| dtype | before | after |
|---|---|---|
| `fp8_e4m3` | 4096/4096 | 4096/4096 |
| `fp4_e2m1` | 4096/4096 (already) | 4096/4096 |
| `fp6_e3m2` | 64/4096 | **4096/4096** |
| `fp6_e2m3` | 77/4096 | **4096/4096** |
| `fp8_e5m2` | 56/4096 | **4096/4096** |
| `fp8_e4m3_quad` | 66/4096 | **4096/4096** |

The first finding was free: **the three `rtl_exact` behaviours are format-independent.** `prod_e/
prod_m` and the `acc_e[]/acc_m[]` schedule are set BEFORE `gemmini.cc`'s per-format branches, and
the K-window is `DIM = 16` for every format — the nibble formats' 32x32 tiling changes which
elements are grouped, not the arithmetic per output. So FP4 needed nothing but `mx_fmt="MXFP4"`.

The codebook formats needed a **fourth** behaviour: `_wire_operands`. `MXLinearSim` re-quantizes A
and B to element codes; the hardware instead projects them through a 16-entry codebook that is
**compile output**, fitted to the data. MXQuant cannot re-derive it — its k-means seeds from
`torch.multinomial` — so the model is now handed the exact `(P, X)` the device received, decoded
from our wire bytes by `mxq_golden.wire_to_px`. For the direct formats this changes nothing
(verified bit-identical either way), which is why it is a general mechanism rather than a LUT patch.

**MXQuant itself is untouched.** `git status` in `MXQuant/` shows only `__pycache__/*.pyc` (tracked
upstream, so any import dirties them) and the untracked capture directory. The hook lives in OUR
`rtl_exact/rtl_datapath.py`, which already rebinds one method at runtime by design, and it is
inert unless a caller sets it — the "as shipped" tier still runs the unpatched path.

#### Edge provenance is DECLARED by the lowering, not inferred

Chained operands exposed a design trap worth recording. Whether a chained A operand came from the
hardware requantizer or from a host re-quantization is **a property of the lowering, not the graph**:

* **fused chain** — the hardware requantizer writes it, on device (`VIA_REQUANT`);
* **graph / per-stage** — it is drained to bf16 and re-quantized by `mx_quantize_rows`
  (`VIA_HOST`, MXQuant's convention).

An earlier draft inferred it from graph shape ("lhs is a previous mesh output"). That is right for
the fused chain and **wrong for the graph path**, where an identical-looking edge goes through the
host — and it would have passed today only because attention and llama are FP8, where the two
conventions coincide. An FP4 attention would have been silently wrong: the same failure mode this
repo has hit three times.

So each lowering now RETURNS its edges (`{"L0": {"via": "requant", "books": ...}}`) and the
reference honours them. Undeclared defaults to `VIA_HOST`, the conservative case. A future emitter
declares its own edges instead of someone remembering to update an inference rule.

With that, `mlp2` chained bit-identically in **fp8_e4m3** and **fp4_e2m1** — and, once the
element encoders below were fixed, in all six formats.

#### The codebook chain, closed (2026-09-09)

All six formats now chain bit-identically, to **eight stages**:

| dtype | mlp2 | mlp4 | mlp8 |
|---|---|---|---|
| `fp8_e4m3` | 4096/4096 | 4096/4096 | 4096/4096 |
| `fp8_e4m3_quad` | **4096/4096** | **4096/4096** | **4096/4096** |
| `fp8_e5m2` | **4096/4096** | **4096/4096** | **4096/4096** |
| `fp6_e3m2` | **4096/4096** | **4096/4096** | **4096/4096** |
| `fp6_e2m3` | **4096/4096** | **4096/4096** | **4096/4096** |
| `fp4_e2m1` | 4096/4096 | 4096/4096 | 4096/4096 |

**What was wrong, and why three attempts missed it.** Every attempt had been comparing the FINAL
OUTPUT of a two-stage chain — 27/4096 — which says a chain is broken and nothing about where. The
requant path has five steps, and a wrong one anywhere produces the same symptom. So instead of a
fourth guess, `tests/oracle/requant_oracle.cc` lifts `gemmini.cc:1445-1509` verbatim, links it
against `libgemmini/mx_fp_math.h` (the real encoders, the real finders), and prints the three
intermediates the Python could not otherwise be compared on: **scale codes, element codes,
indices**.

The first diff located it in one run. Scales: 128/128 — already right. Indices, given the same
element codes: right too, so the finder work in §4.14 was correct. **Element codes: 3924/4096.**

The element encoder was the whole failure, in three separate ways, none of which a generic
"round to the nearest grid point" gets right:

* **Signed zero.** A negative value that rounds to zero keeps its sign — code `0x20`, not `0`. Only
  a zero *input* is canonicalized. ~4% of elements on its own.
* **The tie rule is per format.** E2M3 rounds half to EVEN (`nearbyintf`); E5M2 and E4M3 round half
  AWAY from zero (`round_half_away`). One shared rule is wrong for half the formats.
* **E3M2 does not round to its own grid at all.** It goes BF16 → E4M2 (RNE with `sticky|lsb`) → a
  deterministic E4M2→FP6 map → code. The intermediate FLUSHES small values that a direct E3M2
  rounding keeps as subnormals.

`tensor_to_custom_fp_codes`, from the extracted mesh model, gets all three wrong — correctly, for
its own purpose. It encodes an OPERAND, where the value is already on the grid. A REQUANT OUTPUT is
an arbitrary accumulator value being rounded by hardware. Reusing the operand encoder for it was the
category error underneath all three attempts.

So the four encoders are transcribed into `app/mxwire.ENCODERS`, next to the decoders — the natural
home, since the decoders were already the single place a format's semantics are written down. All
three steps then matched 4096/4096 for all four formats, and `requantize_chained` became a
statement-for-statement transcription of the device block instead of a reconstruction from parts.

**Generalizable, not chain-specific.** Per the standing instruction, three things make this hold for
formats that do not exist yet:

* `encode_requant` **refuses** a dtype with no entry, naming `mx_fp_math.h` — a new format cannot
  arrive at the requantizer and be silently rounded by a generic rule.
* `tests/selftest_requant.py` is a permanent gate that compares the PROCEDURE, not a number: it
  builds the oracle from the live header and diffs all three steps across four formats and three
  magnitude decades. If `mx_fp_math.h` changes, it fails at the step that changed. Where there is no
  C++ compiler it SKIPS loudly rather than passing vacuously.
* The bf16 rounding of the requantizer's input is done inside `_requantize_codebook` rather than
  assumed of the caller. Under `rtl_exact` the input is already bf16 and it is a no-op — but the
  self-test found it immediately when fed raw fp32, which is what a future caller will do.

`chain_refusal` was re-examined and left in place, refusing nothing: with `log2_pmax = 0` everywhere
its aliasing condition cannot fire, and its docstring now says so rather than carrying a table of
`out_pmax` values that no longer exist. It stays as a live invariant for a format that reintroduces
one.

### 4.17 MXQuant's own element quantizers do NOT match the hardware (2026-09-09)

Raised by the user after §4.16: *"mxquant also has a requantization implementation and those needs
to match with this golden implementation."* Correct, and it is a bigger gap than expected.

§4.16 concluded "MXQuant does not need to change" — that answer was about the requantizer as a
STEP, which MXQuant genuinely does not model. It missed that the element quantizers MXQuant does
have are the same functions the hardware applies inside that step.

**Two implementations live in MXQuant, disagreeing with the hardware by different amounts.**
Exhaustive over all 3330 bf16 values with `|v|` in `[2^-12, 2)` — the whole live range after a
p=0 block scale — comparing against `app.mxwire.ENCODERS` (oracle-verified in §4.16):

| format | prodacc `quantize_to_*` | microxcaling `_quantize_elemwise` |
|---|---|---|
| `fp8_e4m3` | **34.11%** | 0.00% |
| `fp8_e5m2` | not implemented | 0.00% |
| `fp6_e3m2` | **19.58%** | 9.97% |
| `fp6_e2m3` | **31.41%** | 0.48% |

The grader runs the PRODACC column: `eval_complete.py` (which holds `MXLinearSim`) imports
`quantize_to_fp8_e4m3` / `_fp6_e3m2` / `_fp6_e2m3` directly.

Three separate causes:

* **prodacc flushes every subnormal to zero by construction** (`mask_subnormal = abs < MIN_NORMAL
  -> 0`). Neither the hardware nor microxcaling does this. It is the bulk of all three numbers —
  1024 of E4M3's 1136 mismatches.
* **E3M2's hardware path is not a rounding to the E3M2 grid at all** — BF16 -> E4M2 -> a
  deterministic map, whose intermediate flushes everything below 0.0546875. That is microxcaling's
  residual 9.97%, and it is the HARDWARE being lossier, not the model.
* **The E2M3 tie rule** — half-to-even in spike (`nearbyintf`), half-away in microxcaling. Exactly
  16 values, all exact ties. See the open question below.

**FP4 is the proof this matters, and the proof of how to fix it.** Someone already hit this for FP4
and fixed it the right way: `hw_bf16_to_e2m1`, written to match `BF16ScaleRoundToTiny.scala`,
selected by `fmt == "MXFP4_HW"` at `eval_complete.py:403`, leaving the generic quantizer untouched.
On a single 64x64x64 matmul, MXQuant-as-shipped against silicon:

| dtype | shipped vs hardware |
|---|---|
| `fp4_e2m1` — **has a HW encoder** | **0.2640%** (4085/4096 identical) |
| `fp8_e4m3` | 7.05% |
| `fp6_e2m3` | 13.97% |
| `fp8_e5m2` | 15.85% |
| `fp6_e3m2` | 15.92% |

FP4 is two orders of magnitude closer than the rest, and the only difference is a hardware-accurate
encoder.

**Scope.** The `rtl_exact` tier is unaffected — it bypasses MXQuant's quantizers entirely via
`_wire_operands`. What this affects is the `mxquant_shipped` tier (where it is the number being
measured, correctly) and **MXQuant's own accuracy research**, whose perplexity numbers are computed
on a grid the silicon does not have.

**Not done, and deliberately so:** MXQuant is untouched per the standing instruction. The change
would be four `hw_*` encoders in `mx_quantization.py` plus `_HW` format selectors, mirroring the FP4
precedent exactly; the verified implementations already exist in `app/mxwire.py` to port across, and
`tests/selftest_requant.py` would gate them.

**Correction (same day):** an earlier draft of this section reported microxcaling as 9.97% / 0.48%
off and called `mx_fp_math.h:537`'s "bit-exact vs `mx._quantize_elemwise` (0/20000)" claim into
question. That was OUR mis-parameterization, not their error. `_quantize_elemwise` takes a `round=`
argument whose names are counter-intuitive — **`'nearest'` means half AWAY from zero, `'even'` means
RNE** — and the sweep had been run with `'nearest'` throughout. Re-run per format:

| format | vs `round='nearest'` (half-away) | vs `round='even'` (RNE) |
|---|---|---|
| `fp8_e4m3` | **0.00%** | 1.92% |
| `fp8_e5m2` | **0.00%** | 1.56% |
| `fp6_e2m3` | 0.48% | **0.00%** |
| `fp6_e3m2` | 9.97% | 9.49% |

The claim at `mx_fp_math.h:537` is correct. Three of the four formats match microxcaling EXACTLY
under the right mode.

#### The real finding: the hardware's tie rule is inconsistent between formats

E4M3 and E5M2 round half AWAY from zero (`round_half_away`, `mx_fp_math.h:197`); E2M3 rounds half to
EVEN (`nearbyintf`, `mx_fp_math.h:556`). Each matches microxcaling exactly — but not under the same
mode, so no single MXQuant configuration reproduces all three. That is a datapath inconsistency, not
a modelling gap.

#### E3M2's 9.49% residual is entirely its E4M2 intermediate

`bf16_bits_to_fp6_e3m2_code` does not round to the E3M2 grid. It goes BF16 -> E4M2 (RNE) ->
`_e4m2_to_fp6_float` -> code, and that intermediate causes both residual effects:

* **222 values flushed to zero** — `_e4m2_to_fp6_float` returns 0 for `ax <= 0.0546875`. The flushed
  band is `|v|` in `[0.0315, 0.0583]`, and E3M2 *can* represent that: its subnormal quantum is
  0.0625, so 0.0583 should encode to 0.0625, not 0.
* **94 values rounded to the FARTHER grid point.** 0.0859375 -> hw 0.125, where true-nearest is
  0.0625 (distance 0.023 against 0.039). Double rounding: E4M2 takes 0.0859 to 0.09375, and
  `_e4m2_to_fp6_float` maps everything in (0.078125, 0.15625] to 0.125.

This one IS a candidate RTL fix — encode straight to the E3M2 grid and the format matches
microxcaling like the other three.

#### Decomposing the shipped gap: encoder vs datapath

`A` = rtl_exact datapath + hardware wire operands (our golden); `B` = rtl_exact datapath +
MXQuant's own operand prep; `C` = shipped datapath + MXQuant's own operand prep. Single 64x64x64
matmul, relative Frobenius:

| dtype | operand prep (A..B) | datapath (B..C) | both (A..C) |
|---|---|---|---|
| `fp8_e4m3` | 0.00% | 7.16% | 7.16% |
| `fp8_e5m2` | 18.95% | 7.53% | 20.46% |
| `fp6_e3m2` | 19.08% | 7.59% | 20.75% |
| `fp6_e2m3` | 14.46% | 6.86% | 16.83% |
| `fp4_e2m1` | 0.00% | 0.00% | 0.00% |

Two things this says that the grid-point percentages do not:

* **For E4M3 the element encoder costs nothing in practice (0.00%)** even though prodacc is 34% wrong
  across the grid. After block normalization the data does not reach E4M3's subnormal range, which
  is where prodacc's flush lives. Grid-point percentages OVERSTATE impact; this is the number to
  quote.
* **The 14-19% on the three codebook formats is dominated by CODEBOOK fitting, not element
  rounding** — MXQuant re-derives the 16-entry table with k-means seeded from `torch.multinomial`,
  where the hardware uses the table our compiler emitted (§4.16). Do not read that column as a
  rounding gap.

`fp4_e2m1`'s 0.00%/0.00% is unexplained and worth a look — the three rtl_exact behaviours should
change something. Its `mxq` name is `MXFP4`, not the `MXFP4_HW` that selects `hw_bf16_to_e2m1`
at `eval_complete.py:403`, so the earlier guess that FP4 is close *because* it has a hardware
encoder is NOT established.

#### Implemented on the SPIKE/GOLDEN model (2026-09-10) — RNE chosen, RTL deferred

Per the user, the fix was applied to the spike/golden element quantizers
(`software/libgemmini/mx_fp_math.h`) only; the RTL (`mxgen`) is a separate follow-up. Tie-rule
decision: **RNE / `round='even'`** (OCP MX spec) — the single mode under which all four formats match
`microxcaling._quantize_elemwise`. (The grader's prodacc `quantize_to_*` defaults to `round='nearest'`
= half-away *and* flushes subnormals; that subnormal flush is a prodacc operand-prep artifact and is
deliberately NOT put into hardware, so the target is the OCP reference — which keeps denormals — not
prodacc as-shipped.)

* `fp8_e4m3_to_code`: `round_half_away` → `nearbyintf` (RNE), both sites; helper retired.
* `bf16_bits_to_e5m2_code`: normal + subnormal round bits → RNE.
* `bf16_bits_to_fp6_e3m2_code`: **rewritten to round straight to the E3M2 grid** (RNE, denormals kept),
  mirroring the E2M3 encoder; the BF16→E4M2→`_e4m2_to_fp6_float` double-rounding path and its three
  now-dead C helpers were removed. Fixes both defects above (the 222 flushed and 94 farther-grid values).
* `bf16_bits_to_fp6_e2m3_code`: unchanged (already RNE).

**Verified bit-exact:** exhaustive over all 3328 bf16 values with `|v|` in `[2^-12, 2)`,
`0/3328` mismatches vs `_quantize_elemwise(round='even')` for **all four** formats (fp8_e4m3, fp8_e5m2,
fp6_e3m2, fp6_e2m3). This makes the golden the RTL is checked against, so E4M3/E5M2/E3M2 RTL tests will
mismatch until the RTL rounding is brought to RNE + straight-E3M2-grid (the deferred step).

### 4.18 Spike moves to RNE; the reference follows (2026-09-10)

Spike was updated with both fixes proposed in §4.17. `mx_fp_math.h`: `round_half_away` **deleted**
(E4M3 and E5M2 now use `nearbyintf`), and `bf16_bits_to_fp6_e3m2_code` **rewritten** to round
straight to the E3M2 grid — `bf16_bits_to_e4m2_rne`, `e4m2_to_fp6_float` and `fp6_value_to_code`
are gone with it.

**The hardware is now exactly the OCP MX reference, under one configuration.** Exhaustive over all
3330 bf16 values with `|v|` in `[2^-12, 2)`:

| format | vs `round='nearest'` | vs `round='even'` (RNE) |
|---|---|---|
| `fp8_e4m3` | 1.92% | **0.00%** |
| `fp8_e5m2` | 1.56% | **0.00%** |
| `fp6_e3m2` | 0.48% | **0.00%** (was 9.49%) |
| `fp6_e2m3` | 0.48% | **0.00%** |

The per-format tie inconsistency is gone and so is E3M2's double rounding. `mx._quantize_elemwise
(round='even')` is now THE reference for all four.

#### The stale `.so`, and a build rule that hides it

`libgemmini.so` was built 09-09 18:26 against a header edited 09-10 16:59. Running before noticing
would have exercised the OLD datapath and reported a confident, wrong result. The cause is in the
Makefile: `libgemmini.so: gemmini.cc` does not list `mx_fp_math.h`, so editing the header never
triggers a rebuild. **Check the `.so` mtime against the header's before trusting any spike run.**
Adding the header to that prerequisite list would fix it for good (not done — other repo).

#### What the update broke, and why it is the interesting part

After rebuilding, one thing failed: the **`fp8_e4m3` fused chain, 1242/4096.** Everything else —
all six single matmuls, the four codebook chains, the FP4 chain — passed.

`fp8_e4m3` was the only format still routed to `quantize_operand` in `requantize_chained`, on the
documented grounds that *"for FP8 the requantizer WAS migrated to MXQuant's convention, so the two
agree."* That was true when written. The header revision moved the hardware to RNE while MXQuant's
default stayed half-away, and the shared convention stopped being shared.

**A convention shared with another codebase is a dependency, not a simplification.** The four
codebook formats survived the same header change with a one-line encoder edit, because they go
through a transcription of the device block; the one format that leaned on an agreement broke. So
E4M3-single now has its own transcription too, `_requantize_direct_e4m3`, and no chained format
reaches MXQuant for its requant rounding.

It is a SEPARATE routine from `_requantize_codebook`, deliberately — `gemmini.cc:1303-1378` differs
from the codebook block in three ways that a shared parameterized version would have to carry
anyway:

* **no bf16 rounding of the scaled value** — it calls `fp8_e4m3_to_code(v / scale)` on the raw
  float, where the codebook path stores `f32_to_bf16_rne(scaled)` and encodes those bits;
* **an epsilon clamp on the block max** — `amax = max(amax, FLT_EPSILON)`, so an all-zero block
  gets code 104, not 0. `gemmini.cc:1341` explains why: code 0 used to mean both "all-zero block"
  and "the accumulator overflowed";
* **NaN/Inf propagate out of band** as E8M0 `0xFF`, with the scale itself set to NaN/Inf so element
  codes match byte for byte.

With that, `mlp2 fp8_e4m3` is back to 4096/4096.

#### The gate paid for itself the day after it was written

`tests/selftest_requant.py` failed immediately on the new header and named the step and the format:
`{'element': 463, 'index': 118}` for E3M2, with `fp6_e2m3` still passing untouched because it was
already RNE. Updating `app/mxwire.ENCODERS` was mechanical rather than an investigation — the
contrast with §4.16's three blind attempts is the argument for comparing procedures, not numbers.

It did NOT cover the direct 8-bit path, which is how `fp8_e4m3` reached a kernel run before failing.
**That gap is now closed** — the oracle gained `fmt 4` (E4M3-single: epsilon-clamped scale, no bf16
pre-round, no finder, so its "index" is its element code) and `selftest_requant` runs 15 cases
instead of 12.

Closing it found a SECOND bug the same hour, one that every kernel run had missed:
`_requantize_direct_e4m3` was not bf16-rounding its INPUT. The device reads the accumulator out of
smem, which is bf16, so both requant paths must; only the codebook one did. Under `rtl_exact` the
input is already bf16 and it is a no-op, which is exactly why 21 passing kernel runs said nothing —
the same latent defect the codebook path had, found the same way, one day apart.

Worth keeping the two roundings distinct, because they are easy to conflate:

* the **input** is bf16-rounded in BOTH paths (smem is bf16);
* the **scaled value** is bf16-rounded only in the CODEBOOK path — the direct path encodes the raw
  float via `fp8_e4m3_to_code(v / scale)`.

#### The shipped baremetal goldens are now stale (action for gemmini-rocc-tests)

`tests/selftest_formats.py` fails two chain cases after the update. This one is NOT ours: that test
runs SPIKE against the goldens shipped in `gemmini-rocc-tests/include`, and our Python computes
neither side of the comparison.

| case | vs shipped golden |
|---|---|
| `fp8_e4m3` chain | 250/4096 codes differ, **0/128 scales** |
| `fp6_e3m2` chain | 56/2048 codes differ, **0/128 scales** |
| `fp4_e2m1` chain | PASS -- its encoder did not change |

Proven by rebuilding the OLD header under a forced `make clean`:

| header | md5 | result |
|---|---|---|
| old (`round_half_away`) | `ae67e077` | **4096/4096 codes, 128/128 scales -- exact** |
| new (RNE) | `bba2b93d` | 250 differ, all `odd -> even`, delta -1 |

`matmul_fp8_64x64_chain.h` and `matmul_fp6_64x64_chain.h` need REGENERATING against the updated
spike. `matmul_fp4_64x64_chain.h` does not.

#### Two method failures in this investigation, worth more than the result

**1. `make` reports success while doing nothing.** The Makefile bug is worse than "editing the header
does not trigger a rebuild": `make` exits 0 having skipped the build, so a test silently exercises
the WRONG datapath and reports a confident number. The first attempt at the old-header experiment
did exactly that -- `git checkout mx_fp_math.h && make` re-ran the NEW binary, produced identical
failures, and appeared to refute a conclusion that was correct. **`make clean` is mandatory**, or
add `mx_fp_math.h` to the prerequisite list. Checking the `.so` mtime is NOT sufficient.

**2. The direct evidence was available first and was ignored.** 250 differing codes, all ADJACENT,
every one `odd -> even`, 58 distinct pairs -- that is the fingerprint of half-away -> RNE and it
settles the question on its own. It was gathered only after asserting a conclusion, then running a
broken experiment, then reversing. Characterize the divergence BEFORE reaching for a bisect: it is
cheaper and it does not depend on a build system behaving.

#### Unchanged, and correctly so

The §4.17 shipped-gap decomposition re-ran byte-identical (18.95% / 19.08% / 14.46%). Not a stale
result: that script measures a SINGLE matmul, whose operands go through `quantize_operand`, never
`encode_requant`. The requant encoders only run on a chained intermediate, so the RNE fix cannot
move those numbers by construction.

It does settle §4.17's open question, though. That column was suspected to be codebook FITTING
rather than element rounding. Element rounding is now provably exact against the OCP reference and
the column did not move, so it is k-means seeding — confirmed rather than assumed.

## 5. Known unknowns, recorded rather than assumed

* **E5M2 / E2M3 / E4M3-quad are not in any format table yet.** `gen_matmul_llama.py:95-100` covers
  fp8/fp6/fp4 only; the LUT formats come from `lut_mapping_demo.py` and `llama_operands.py`, a third
  generator family with its own k-means codebook construction. Step 5 has to unify all three.
* **The LUT codebook is data-derived** (`llama_operands.build_luts`, k-means per group of `2^G`
  rows), so a LUT format's operands are not a pure function of the tensor — the codebook is part of
  the compile output. That has no analogue in the current command buffer.
* **FP4/FP6 requant rounding is not MXQuant** (§9.4): every disagreement is one step on the value
  ladder, a tie-break difference. Step 1 must keep `out_requant="model"` for them rather than
  forcing D3 where the hardware does not follow it — D3 governs the *quantization strategy*
  (operands, block scales), not the hardware's own output rounding.
* **Two B mvin layouts both pass** at square shapes (§9.6). Step 5 must pick one and prove it on a
  non-square case per format, or it re-introduces the §10.2 class of bug.
* RTL execution is unverified for everything (D6).

## 6. Related plans

`npu_exploration_bridge_plan.md` (how the bridge got here), `chain_seam_hw_notes.md` (why the seam
existed and what killed it), `llama_layer_hw_plan.md` (the real-layer kernels this absorbs),
`../planning/mxgemmini_rocket_standalone_plan.md` (the RTL-side output modes).
