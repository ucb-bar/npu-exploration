"""The hardware recipe: one file that defines the MX-Gemmini we are running on.

A recipe is consumed by THREE models of the same machine:

  * ``models/mxquant/mxquant.py``  — the Python model on mxq (the bits we expect)
  * ``models/spike/build_spike.py`` — patches libgemmini's C model, seconds to build
  * ``config/scala/JsonGemminiConfig.scala`` — Chisel elaboration, tens of minutes

so the schema is the one the Scala parser already enforces (``array`` / ``types`` /
``mx`` / ``accumulator``), plus two sections it is told to ignore (``software`` /
``runtime``) that never reach the hardware.
"""
