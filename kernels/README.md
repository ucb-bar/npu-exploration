# kernels — what to run

A kernel is **data, not code**: a PyTorch model plus an input, expressed as an ordered list
of stages. The backend lowers exactly one matmul per command buffer, so a model of N
`nn.Linear` layers becomes N stages.

Two stage kinds:

| kind | runs on | notes |
|---|---|---|
| `Stage` | the **mesh** | one weight-stationary MX matmul, `lhs[M][K] @ rhs[K][N]` |
| `HostStage` | the **host** | anything the mesh cannot do — softmax, activations |

A `Stage`'s operands are names, so they can reference earlier stages rather than only
resident weights: `"x"` (the model input), a stage name, or either with a `.T` suffix.
That is what lets attention express `S = Q @ K^T` and `O = P @ V`. `lhs` defaults to the
previous stage, so a straight chain still needs neither operand named.

| file | role |
|---|---|
| `spec.py` | `Stage`, `KernelSpec`, `from_module()` — the machinery; rarely edited |
| `registry.py` | the catalogue: name → builder. **Add kernels here.** |

Adding one is a few lines in `registry.py` and no other change — see "Adding a kernel" in
the root `README.md`.

`KernelSpec.reference()` computes the FP32 answer for the whole chain, and
`KernelSpec.validate()` checks shape legality *before* anything is built, mirroring
`mxgemm_emit._validate` so failures are cheap and name the offending stage.

`from_module()` raises on bias and on activations rather than approximating them: both need a
COMMIT epilogue, which the backend explicitly refuses. They *could* be added as a `HostStage`,
but that credits the host with work the accelerator did not do, so it is never done silently.

Softmax is a `HostStage` for the same reason it is not a compiler op: the target contract
declares one compute unit, `mx_systolic_mesh`, `ops: [matmul]`, and **there is no reduction
hardware**. `merlin_iface` has no softmax op either. Marking host stages explicitly is what lets
the report separate accelerator cycles from work that never reached the device.

**Honest limit:** a kernel is still N ELFs and N spike runs, not one program — `mxgemm_emit._plan`
takes one matmul per command buffer, and host stages currently execute in the harness process
rather than on the simulated Rocket core, so their cost is not counted anywhere.
