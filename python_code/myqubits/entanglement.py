"""Entanglement measures: concurrence, entanglement of formation, negativity."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import ATOL, _dense, as_dm, psd_sqrt, sy
from .entropy import _out, binary_entropy
from .subsystems import partial_transpose

__all__ = ["concurrence", "concurrence_x", "entanglement_of_formation",
           "negativity", "log_negativity"]

_YY = np.kron(sy, sy)


def concurrence(state) -> float:
    """Wootters concurrence of a two-qubit density matrix or ket.

    For a ket it is ``2 |psi_00 psi_11 - psi_01 psi_10|``.  For a density
    matrix the ``lambda_i`` (square roots of the eigenvalues of
    ``rho (sy x sy) rho* (sy x sy)``) are obtained as the singular values of
    ``sqrt(rho) sqrt(rho_tilde)``, which avoids the non-Hermitian
    eigenproblem and is accurate also for rank-deficient states.
    """
    a = _dense(state)
    if a.ndim == 1:
        if a.shape != (4,):
            raise ValueError("concurrence needs a two-qubit ket")
        return float(2 * abs(a[0] * a[3] - a[1] * a[2]))
    if a.shape != (4, 4):
        raise ValueError("concurrence needs a 4x4 density matrix")
    rho_tilde = _YY @ a.conj() @ _YY
    lam = np.linalg.svd(psd_sqrt(a) @ psd_sqrt(rho_tilde), compute_uv=False)  # descending
    return float(max(0.0, lam[0] - lam[1:].sum()))


def concurrence_x(rho) -> float:
    """Concurrence of a two-qubit X state (nonzero only on the diagonal and anti-diagonal):

    ``2 max(0, |rho_12| - sqrt(rho_00 rho_33), |rho_03| - sqrt(rho_11 rho_22))`` (0-based indices).
    """
    r = _dense(rho)
    k1 = abs(r[1, 2]) - np.sqrt(abs(r[0, 0] * r[3, 3]))
    k2 = abs(r[0, 3]) - np.sqrt(abs(r[1, 1] * r[2, 2]))
    return float(2 * max(0.0, k1, k2))


def entanglement_of_formation(state, base: float = 2) -> float:
    """Two-qubit entanglement of formation ``h((1 + sqrt(1 - C^2))/2)``."""
    c = min(concurrence(state), 1.0)
    return float(binary_entropy((1 + np.sqrt(1 - c * c)) / 2, base))


def _pt_spectrum(rho, dims, sys):
    rho = as_dm(rho) if np.ndim(rho) < 3 else np.asarray(rho)
    return np.linalg.eigvalsh(partial_transpose(rho, dims, sys))


def negativity(rho, dims: Sequence[int] = (2, 2), sys=0):
    """Sum of the absolute values of the negative eigenvalues of the partial transpose.

    Accepts stacks of density matrices; ``dims`` defaults to two qubits.
    """
    w = _pt_spectrum(rho, dims, sys)
    n = np.clip(-w, 0.0, None).sum(-1)
    return _out(np.where(n < ATOL, 0.0, n))


def log_negativity(rho, dims: Sequence[int] = (2, 2), sys=0):
    """``log2 ||rho^{T_sys}||_1``."""
    return _out(np.log2(np.abs(_pt_spectrum(rho, dims, sys)).sum(-1)))
