"""The hardware recipe: one file that defines the MX-Gemmini we are running on.

A recipe is consumed by THREE models of the same machine:

  * ``grade/golden.py``            — the Python model (what we expect)
  * ``config/build_spike.py``      — patches libgemmini's C model, seconds to build
  * ``config/scala/JsonGemminiConfig.scala`` — Chisel elaboration, tens of minutes

so the schema is the one the Scala parser already enforces (``array`` / ``types`` /
``mx`` / ``accumulator``), plus two sections it is told to ignore (``software`` /
``runtime``) that never reach the hardware.
"""
