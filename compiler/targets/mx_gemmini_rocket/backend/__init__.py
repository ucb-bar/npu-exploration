"""MX-Gemmini (Rocket host) backend — the out-of-tree package merlin loads.

The target contract declares ``plugin.backend: backend``, so
:func:`merlin.runtime.backends.base._load_oot_backend` imports THIS directory as
``merlin._oot_backends.mx_gemmini_rocket`` with ``submodule_search_locations`` set to it — which is
why the siblings use relative imports (``from .mxgemm_emit import ...``). Importing the package runs
the ``register(...)`` below, so ``get_backend("mx_gemmini_rocket")`` resolves here with no name ->
module map in the core.

Registration is BEST-EFFORT: this package is also used standalone (an ``app/`` script driving spike
directly, with no merlin on the path). A missing merlin makes the backend unregistered, not broken.
"""
from __future__ import annotations

from .mxgemm_emit import (  # noqa: F401
    MxEmitError,
    MxGemmPlan,
    SpikeSmemTransport,
    Transport,
    generate_driver,
)
from .runner import (  # noqa: F401
    ORACLE,
    MxRunnerError,
    available,
    compile_command_buffer,
    parse_output,
    run_command_buffer,
    run_elf,
)

try:
    from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register
except ImportError:  # standalone use — no merlin on the path
    pass
else:
    register(BackendInfo("mx_gemmini_rocket", TargetClass.NPU, BackendKind.KERNEL, __name__))
