# Perf-model handoff: loop spad store made RTL-exact, and the missing pending-bank gate (2026-10-06)

From the Llama-7B flash-attention tuning session to the session that owns the perf model (`perf_model_plan.md`).
I made **one** change to the model (below), kept it at the user's request, and stopped there. The second item
(pending-bank gate) is **not implemented**; it is the reason the total got worse and should be next.

## 1. What I changed: `L_STSPAD` (loop store into the scratchpad) follows `LoopMatmulStCSpad`

Files (uncommitted in the libgemmini working tree; ~28 lines):
* `perf/control/loop_matmul.h`: `sts_group_tiles(J, g)`, `sts_chunks(J, g)`, `sts_per_i(J)`.
* `perf/control/loop_matmul.cc`: `total[L_STSPAD] = I * sts_per_i(J)`; in `make()` the `L_STSPAD` branch now
  1. iterates **j groups outer, i inner, chunk innermost**: one command per 2-tile chunk of a 4-tile group
     (`chunks_this_j = fp8_tiles_this_j / 2`, LoopMatmul.scala:849-850, 959-971; `max_block_len = 1`, :1158, :842);
  2. releases a command by the RTL's `ex_ahead` (:944-949): the group's last-k tile `(i, ej_high)` has left
     (`need = (K-1)*J*I + group_last*I + i + 1`); the loop's **last** group, when there is more than one, waits for
     every compute (`terminal_needs_drain`);
  3. gives it the RS range of its **group**: `rs_span = make_span(acc_tile(i, 4g), DIM)` (`src_addr` = group base,
     `cols` = the group's packed cols -> one mat -> DIM rows, :913-920 + ReservationStation.scala:278-289);
  4. `cols = tiles * DIM / chunks` per command, so the data moved per matmul is unchanged.
* `perf/model.cc`: `r.a = c.rs_span` for both `L_STC` and `L_STSPAD` (was `store_.src_span(s)` for `L_STSPAD`).

Why: the old `L_STSPAD` stored per tile, i-major, with a DIM-row range from the **tile's** address. In the packed
layout (tile = DIM/4 address units) that range covered the next three tiles, so every tile-column step of every
matmul's last k made the next preload wait for the store: ~52-cycle mesh bubbles, ~160 per `attn_flash_llama_2h_fused`
pass. **The RTL does not have them** (VCS waveform below).

### Also in the working tree: the `rs.packed_exact` knob is removed
I added a what-if knob `rs.packed_exact` (exact DIM/4 ranges) while chasing those bubbles; it was committed to HEAD
while it was in my working tree. My working tree removes it again (`perf/params/config.h`, two uses in `model.cc`),
because it described a non-existent RTL stall. Keep or drop as you prefer.

## 2. Validation on the exact binary

* Binary: `attn_flash_llama_2h_fused` built with `-DATTN_STATS_OUT=1` (new kernel flag in `attn_flash.c`: keeps the
  pipelined pass's debug stats mvout, which current builds drop). Its `objdump -D` equals the VCS run's `.dump`
  (`sims/vcs/output/chipyard.harness.TestHarness.MxE4M3VpuGemminiRocketConfig/attn_flash_llama_2h_fused.dump`,
  0 differing lines). Copy: `/tmp/claude-200740/-bwrcq-scratch-nicorakela-radiance-cy-dev-generators-gemmini/8546cac3-2dd4-4e86-b676-3fb919d98131/scratchpad/val/attn_flash_llama_2h_fused`.
  (The vb / vb_fused / 2h VCS runs are from older source states and do not match the current build.)
* VCS: **156,475** cycles (`PERF pipelined` in the `.log`).

| | total (e4m3_vpu) | short mesh idle runs (last 150k cycles) |
|---|---|---|
| RTL (FSDB) | 156,475 | 28 x ~35 (980 cycles) |
| model before | 155,652 (-0.5 %) | ~160 x 52, 29 x 45, 16 x 72 |
| model after | 151,478 (-3.2 %) | 32 x 22, 29 x 45, 32 x 74, 16 x 117 |

The old -0.5 % was two errors cancelling: ~8k of non-existent short stalls vs ~12k of real long stalls the model
lacks (section 3). After the fix only the second error is left.

## 3. Missing in the model: the vector-issue gate on spad banks with pending stores (NOT implemented)

RTL:
* `Scratchpad.scala:954-977`: `io.vpu_pending_banks` = banks with rows buffered in `requant_q` (`requant_pending`)
  **or** a scratchpad store between entering `write_norm_q` and leaving `write_issue_q` (`store_pending`; an
  acc-source store retires at its acc read, before its rows land). DRAM stores (dest 0) never count.
* `Controller.scala:1501` wires it to the RS; `ReservationStation.scala:574-578` `unit_ok`: a **vector** entry
  (VPU_EXEC or SPAD_REQUANT) may not issue while `entry_sp_banks(e)` (banks of opa, opb, opc, opd: sources AND
  destination) intersects `vec_pending_banks`.

Model today: `reservation_station.cc` checks only row conflicts, shared read banks and SR ordering for `Q_VEC`.
Suggested shape: per-bank counters in `store_unit_t` for `TO_SPAD` stores (from `process()` until the spad
write-port callback for its last beat; plus SPAD_REQUANT queued rows if they are modelled separately), a callback the
RS queries in `try_issue(Q_VEC)` against the union of the entry's span banks, and a `kick(Q_VEC)` when a count drops.

RTL evidence (`attn_flash_llama_2h_fused` FSDB, `waveform-debug/npi/mesh_gap_hist.py <fsdb> 160000`, which lists
long idle runs with their position): **727-cycle mesh stalls every 17,324 cycles**, at tiles 2321 / 3350 / 4378 / 5407
/ 6435 / 7464, i.e. every 2nd key block (512 tiles/block). With `ATTN_ST_BANK2` the softmax stats live in bank 2
together with S0 / O_0, so every other block the stats ops wait for S/O_j stores draining into bank 2. Plus 962 near
the start and 2,520 / 5,433 at the end of the pass. Total ~12k cycles: about the remaining -3.2 %.

## 4. Why it matters downstream

The flash-attention tuning (`attn_flash_llama7b*`, see `planning/mxgemmini_rocket_standalone_plan.md`, 2026-10-06
entries) was done on the old model. Its choices (V double-buffered, 32/96 first/last key blocks, stats store dropped)
are built and pass on Spike, awaiting RTL. `ATTN_ST_BANK2` in particular looked free in the old model and is exactly
what this gate penalises, so re-check it once the gate is in. Sweep harness used:
`/tmp/claude-200740/.../scratchpad/sweep7b.sh TAG "-D..."` (builds `src/attn_flash_llama7b.c`, runs perf mode).

## 5. Reproduce

```
cd generators/gemmini
GEMMINI_MODE=perf GEMMINI_PERF_CONFIG=e4m3_vpu GEMMINI_PERF_TRACE=/tmp/tr.csv \
  spike --extlib=software/libgemmini/libgemmini.so --extension=gemmini <scratchpad>/val/attn_flash_llama_2h_fused
python3 <scratchpad>/perf_gaps.py /tmp/tr.csv 10     # mesh idle gaps in the pipelined pass (model)
cd waveform-debug/npi && bash run_npi.sh mesh_gap_hist.py <fsdb> 160000   # the same from the RTL waveform
```
