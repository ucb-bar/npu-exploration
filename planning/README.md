# planning — decisions, and the measurements behind them

Not a backlog. Each file records what was decided, what was measured, and what turned out to be
wrong — so a conclusion can be re-derived instead of re-discovered. Corrections are written into the
document next to the claim they correct, rather than the claim being quietly deleted.

| file | what it covers |
|---|---|
| `merlin_glue_port_plan.md` | **the live one** — the port itself, every step's gate |
| `npu_exploration_bridge_plan.md` | how this repo bridges the model side and the compiler side |
| `chain_seam_hw_notes.md` | the hardware facts a chained matmul depends on: scale layouts, the resident seam, what the MX loop path does and does not honour |
| `llama_layer_hw_plan.md` | the real TinyLlama decoder layer — capture, both kernels, the error decomposition, and the small-D variant for RTL |

## Worth reading even if you are not touching that work

* `llama_layer_hw_plan.md` §8.2b — MXQuant's own model reproduces this datapath **bit-exactly**
  after exactly three changes, and they do not decompose. An earlier draft measured them one at a
  time and got both the magnitude and the sign wrong; the correction is kept in place.
* `llama_layer_hw_plan.md` §9.1 — a quantization convention lived in two copies (a generated golden
  and a hand-written C header) and both went stale together, so every test still passed.
* `chain_seam_hw_notes.md` §2 — the scale transpose that the chain seam turns out to get for free.

## Adding one

State the decision, who made it and when, then the evidence. Convert relative dates to absolute. If
a later measurement contradicts an earlier section, correct that section in place and say what the
earlier reasoning got wrong — that is the part worth keeping.

See [`../README.md`](../README.md) for install and the run command.
