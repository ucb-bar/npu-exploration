# `mx_gemmini_rocket` — out-of-tree merlin target

The MX (microscaling) Gemmini datapath driven as a **real RoCC accelerator from a Rocket scalar
host** — functionally modelled by `software/libgemmini`, elaborated as `MxGemminiRocketConfig`.

This is an **out-of-tree package**: merlin resolves it from `MERLIN_TARGET_PATH`, which takes
precedence over everything in-tree (`target_registry.py:38-46`). The merlin submodule stays unforked.

```bash
export MERLIN_TARGET_PATH=<repo>/npu-exploration/compiler/targets
python -c "from merlin.targetgen.target_registry import resolve; print(resolve('mx_gemmini_rocket'))"
```

One path entry resolves a whole *directory* of packages, so future targets drop in beside this one.

## Why a separate target

merlin already knows this datapath in two other forms. Keeping three names is deliberate:

| target | datapath | transport | host |
|---|---|---|---|
| `gemmini` (merlin in-tree) | int8 systolic mesh | RoCC | Rocket scalar |
| `mx_gemmini` (capsule_bench) | **MX** | MMIO command buffer | Muon SIMT (radiance) |
| **`mx_gemmini_rocket`** (this) | **MX** | **RoCC** | **Rocket scalar** |

Rows 2 and 3 share their entire command vocabulary — same funct codes, same rs1/rs2 packing, same
programming sequence. They differ only in transport and in how scales/LUTs/outputs move. The RTL says
so itself (`ConfigsFP.scala:328`): *"Faithful standalone twin of the Radiance MX flow."*

## Layout

```
contracts/target_contract.yaml   the capability manifest — ABI, encoding, datapath, oracle ladder
contracts/dialect_plan.yaml      (Step 3+, deferred — see plan Q4)
backend/                         (Step 4-6) command buffer -> MX RoCC C driver, + the spike runner
```

`backend/` does not exist yet, which is why the contract carries **no `plugin:` block**: merlin
requires every plugin reference to resolve inside the package, and an unresolvable pointer is an
error, not a no-op. The block lands in Step 4 alongside the module it names.

## Two conventions this package follows

**Provenance tags.** Every contract field is tagged `[RTL]`, `[ABI]`, or `[COMPILE]`. The long-term
direction is a single JSON config artifact driving both compilation and RTL elaboration; `[RTL]`
marks the fields that artifact will own. Several are currently written twice — in `ConfigsFP.scala`
*and* in `gemmini_mx_rocket.h` — and kept in sync by hand. See plan §8.

**No radiance-kernels dependency.** `radiance-kernels/` is read for ideas and cited in comments, but
nothing here includes, links, or resolves a path into it. Every *fact* in the contract is grounded in
`generators/gemmini/` — headers, the spike model, or the RTL Scala. See plan §2.3.

## Status

Prototype, **uncertified**. The contract declares an ABI surface and a datapath; it makes no
performance or correctness claim. Spike is a hand-written functional model
(`derived_from_rtl: false`) and is bootstrap only — only Verilator counts as RTL-certified.

Full plan: [`../../../planning/npu_exploration_bridge_plan.md`](../../../planning/npu_exploration_bridge_plan.md)
