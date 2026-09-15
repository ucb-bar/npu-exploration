# compiler — the out-of-tree merlin target

`targets/mx_gemmini_rocket/` is an out-of-tree merlin target package, resolved via
`MERLIN_TARGET_PATH`. It consumes a **command buffer** and emits + builds + runs a bare-metal ELF.

| file | role |
|---|---|
| `contracts/target_contract.yaml` | the capability manifest merlin reads |
| `backend/mxgemm_emit.py` | command buffer → C driver |
| `backend/runner.py` | toolchain, compile, run on spike, parse OUT/METRIC/DONE |
| `backend/__init__.py` | self-registers the backend with merlin |

Nothing here knows about a model, a tensor, or a file — that is
[`app/`](../app/README.md). It receives a command buffer and knows nothing about
tensors, models, or files.

See [`targets/mx_gemmini_rocket/README.md`](targets/mx_gemmini_rocket/README.md).

See [`../README.md`](../README.md) for install and the run command.
