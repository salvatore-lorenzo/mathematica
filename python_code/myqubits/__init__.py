"""myqubits - quantum information and open quantum systems (Python port of MyQubits 0.0.4).

The package is split into independent modules; import only what you need::

    from myqubits.core import sx, sz, kron, ket2dm      # basic definitions
    from myqubits.entropy import von_neumann
    import myqubits.channels as ch

==================  =========================================================
``core``            constants, tensor products, vectorization, matrix functions
``subsystems``      partial trace / transpose, subsystem permutation, reshuffling
``measures``        fidelity, trace distance, purity
``entropy``         von Neumann, Rényi, relative entropy, Shannon, mutual information
``entanglement``    concurrence, entanglement of formation, negativity
``discord``         quantum and geometric discord of two qubits
``bases``           Gell-Mann and Pauli bases, generalized Bloch vectors
``channels``        Kraus / superoperator / Choi / process-matrix representations
``lindblad``        Lindblad and cascaded master equations, solvers
``qubits``          qubit registers, XX chain, Bloch vectors
``random_states``   random kets, density matrices, unitaries, POVMs, channels
``bloch``           Bloch-sphere plots (needs matplotlib)
``symbolic``        SymPy tools: symbolic matrices, closed-form master equations
==================  =========================================================

The contents of :mod:`myqubits.core` are also available at the top level, and
the other modules are imported lazily on first access (``myqubits.entropy``).
"""

from importlib import import_module

from . import core
from .core import *  # noqa: F401,F403

__version__ = "0.0.4"

_SUBMODULES = ("subsystems", "measures", "entropy", "entanglement", "discord", "bases",
               "channels", "lindblad", "qubits", "random_states", "bloch", "symbolic")

__all__ = list(core.__all__) + list(_SUBMODULES) + ["core", "__version__"]


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))
