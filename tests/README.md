# tests — the self-tests, and the claim each one defends

Every test here exists because some claim in this repo would otherwise be unchecked. The right way
to read the table is right-hand column first: if that claim matters to you, that test is the one that
holds it up.

| file | the claim it defends |
|---|---|
| `selftest_grade.py` | metrics, telemetry and reporting agree — including that events logged before the results directory exists are not dropped |
| `selftest_quantizer.py` | our quantizer produces the same bytes as the baremetal headers |
| `selftest_formats.py` | all 6 MX formats and 3 chains match their shipped goldens |
| `selftest_mx_host.py` | the C host runtime (`mx_host.h`) equals its Python twin |
| `selftest_requant.py` | the chained requantizer, per step, against a C oracle |
| `selftest_extracted.py` | `app/mxmesh/` still equals the models it was extracted from |
| `selftest_mx_rocket_build.py` | one C source builds for **both** spike and `MX_ROCKET` |
| `test_recipe_drift.py` | a recipe JSON equals the Chisel it elaborates (see [`../config/`](../config/README.md)) |
| `oracle/` | fixtures the above compare against |

## Using it

```bash
.venv/bin/python tests/selftest_grade.py
.venv/bin/python rtl_exact/verify_rtl_exact.py     # lives with the config it verifies
```

Each is a plain script: exit `0` on pass, non-zero on failure. There is no runner — run the one whose
claim you are about to depend on, or all of them before a commit.

## What is not covered here

A convention shared between a *generated golden* and a *hand-written library* is not held up by any
test in this directory, because both sides can go stale together and still agree. That is exactly
what happened to the E4M3 tie-breaking rule in September 2026 —
[`../planning/llama_layer_hw_plan.md`](../planning/llama_layer_hw_plan.md) §9.1 has the post-mortem. When you
change a quantization convention, regenerate every header, not just the ones a test names.

See [`../README.md`](../README.md) for install and the run command.
