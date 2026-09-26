# compiler — lowering and backend

`lower.py` turns a KernelSpec and an operand format into ONE **command buffer** (a chain of
matmuls: leaf tensors plus four commands per matmul and the wire operands; anything else: the graph
on a side channel). It builds the buffer directly, with no merlin: `tests/selftest_lower.py` holds
it to the dicts merlin's parser used to produce (`tests/oracle/command_buffers.json`). Both
`run_kernel.py` and `compile_kernel.py` call it. `targets/mx_gemmini_rocket/` is the backend: it
consumes a command buffer and emits + builds + runs a bare-metal ELF.

| file | role |
|---|---|
| `lower.py` | KernelSpec + format → command buffer (fused chain, graph, or per-stage records) and edges |
| `contracts/target_contract.yaml` | the capability manifest merlin reads |
| `backend/mxgemm_emit.py` | command buffer → C driver |
| `backend/runner.py` | toolchain, compile, run on spike, parse OUT/METRIC/DONE |
| `backend/mxgraph_emit.py` | graph command buffer → C driver (one ELF with host ops) |
| `backend/__init__.py` | the package; registers with merlin only if merlin is importable (compile_kernel.py never loads it) |

Nothing here knows about a model, a tensor, or a file — that is
[`app/`](../app/README.md). It receives a command buffer and knows nothing about
tensors, models, or files.

See [`targets/mx_gemmini_rocket/README.md`](targets/mx_gemmini_rocket/README.md).

See [`../README.md`](../README.md) for install and the run command.
