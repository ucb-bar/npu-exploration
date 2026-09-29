"""The two recipes a run takes.

  * ``hardware/*.json`` (``--hw``): one MX-Gemmini. Every number is the chip; ``build_id`` is its digest.
  * ``run/*.json`` (``--run``): how software drives it (operand format, rounding, scale floor, reducer,
    the kernel path's threshold). ``run_id`` is its digest.

``config.recipe`` loads and checks both; ``config.scheme`` turns the pair into mxq's Scheme.
"""
