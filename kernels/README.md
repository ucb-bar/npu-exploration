# kernels — what to run

A kernel is **data, not code**: a PyTorch model plus an input, expressed as an ordered list of
stages. The backend lowers exactly one matmul per command buffer, so a model of N `nn.Linear` layers
becomes N stages.

| file | role |
|---|---|
| `spec.py` | `Stage`, `KernelSpec`, `from_module()` — the machinery; rarely edited |
| `registry.py` | the catalogue: name → builder. **Add kernels here.** |

Two stage kinds:

| kind | runs on | notes |
|---|---|---|
| `Stage` | the **mesh** | one weight-stationary MX matmul, `lhs[M][K] @ rhs[K][N]` |
| `HostStage` | the **host** | anything the mesh cannot do — softmax, activations |

A `Stage`'s operands are names, so they can reference earlier stages rather than only resident
weights: `"x"` (the model input), a stage name, or either with a `.T` suffix. That is what lets
attention express `S = Q @ K^T` and `O = P @ V`. `lhs` defaults to the previous stage, so a straight
chain needs no operand named at all.

## Adding one

A few lines in `registry.py` and no other change. What a kernel *cannot* express is an op the
**backend** would lower differently: a fused epilogue, or a convolution.

### Shape rules, checked before anything is built

* `M`, `K`, `N` multiples of the PE tile (`dim`)
* `K` a multiple of the block-scale group (32)
* in a chain, each stage's `K` equals the previous stage's `N`
* a stage feeding another needs `N` a multiple of 32 — the requantizer emits one E8M0 code per 32
  output columns

`KernelSpec.validate()` enforces these *before* anything is built, mirroring `mxgemm_emit._validate`
so failures are cheap and name the offending stage. `KernelSpec.reference()` computes the FP32 answer
for the whole chain.

## Two emitters, on purpose

One fuses a straight **chain** and keeps intermediates in the scratchpad, never touching the host;
the other handles any **graph** — multiple live values, computed operands, host ops — with every edge
through host fp32 memory, which is what the hardware requires wherever a host op sits in a seam.
Both produce one ELF.

## What is refused, and why

`from_module()` raises on bias and on activations rather than approximating them: both need a COMMIT
epilogue, which the backend explicitly refuses. They *could* be added as a `HostStage`, but that
credits the host with work the accelerator did not do, so it is never done silently.

Softmax is a `HostStage` for the same reason it is not a compiler op: the target contract declares
one compute unit, `mx_systolic_mesh`, `ops: [matmul]`, and **there is no reduction hardware**.
`merlin_iface` has no softmax op either. Marking host stages explicitly is what lets the report
separate accelerator cycles from work that never reached the device.

**Honest limit:** host stages execute in the harness process, not on the simulated Rocket core, so
their cost is not counted anywhere. That matters more than it sounds — in the hand-written baremetal
equivalents, where the glue *does* run on Rocket, it is 99.9% of the cycles
([`../planning/llama_layer_hw_plan.md`](../planning/llama_layer_hw_plan.md) §8.3).

See [`../README.md`](../README.md) for install and the run command.
