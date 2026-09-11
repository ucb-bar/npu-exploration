# config — hardware recipes

A recipe is one JSON file = one machine. `[hashed]` fields are the hardware itself:
changing one changes `build_id` and forces a new spike build. All other fields bind
per-instruction (`runtime`) or in Python before the hardware runs (`software`).
`formats.*` is derived by the loader; an explicit block may only confirm it.

| attribute | description |
|---|---|
| `name` | recipe identifier; referenced by `--config` |
| `description` | free text |
| `array.meshRows` | systolic mesh rows (= `dim`) `[hashed]` |
| `array.meshColumns` | systolic mesh columns; must equal `meshRows` `[hashed]` |
| `array.tileRows` | rows per PE tile `[hashed]` |
| `array.tileColumns` | columns per PE tile `[hashed]` |
| `types.meshProdPrecisionList` | 16 MxFloat entries: per-lane product precision; must be uniform (spike declares one scalar `prod_e/prod_m`) `[hashed]` |
| `types.meshAccPrecisionList` | 16 MxFloat entries: the accumulator ladder down the column `[hashed]` |
| `types.*[].expWidth` | exponent bits of that lane's format |
| `types.*[].sigWidth` | significand bits incl. the implicit leading bit (C mantissa = `sigWidth − 1`) |
| `types.*[].count` | how many consecutive lanes use this format |
| `types.*[].isRecoded` | Chisel recoded-format flag (pass-through) |
| `types.*[].pad` | Chisel padding flag (pass-through) |
| `mx.scaleSize` | input block-scale group: elements per E8M0 code (spike: only 32 wired) `[hashed]` |
| `mx.scaleSizeOut` | requantizer output group size `[hashed]` |
| `mx.enable_lut` | LUT decode hardware present `[hashed]` |
| `runtime.operand_fmt` | element format per instruction: `fp8` \| `fp6` \| `fp4` (fp6/fp4 rejected at load until the encoder is wired) |
| `runtime.out_dtype` | commit format: `bf16` \| `f8E4M3FN` |
| `runtime.use_lut` | LUT decode enable bit (CONFIG_EX) |
| `software.block` | quantizer group size along K |
| `software.target_code_exp` | peak-code exponent the encoder targets (accumulator-overflow headroom) |
| `software.seam` | chain-seam strategy: `weight` \| `rescale` |
| `software.intermediate_dtype` | commit format of non-final chained stages |
| `supported_backends` | simulators this recipe may run on: `spike` \| `verilator` |
| `provenance.rtl` | Scala file(s) each hardware fact was taken from |
| `provenance.spike` | C file(s) each hardware fact was taken from |
| `provenance.note` | anything a future reader must know about the mapping |
| `formats.<fmt>.tile` | `[M, N, K]` hardware tile for that operand format (derived: `[dim·pack, dim·pack, dim]`) |
| `formats.<fmt>.codes_per_byte` | operand packing density (derived: fp4 = 2, else 1) |
| `formats.<fmt>.prod_frac_bits` | exact product fraction width (derived: `2·m + 1`) |
| `formats.<fmt>.via_lut` | decode goes through the LUT SRAMs (derived) |
| `formats.fp6` | fail-closed: rejected at load until pinned against `lut_golden_model.py` |
