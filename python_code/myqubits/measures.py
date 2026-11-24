"""Distance and similarity measures between quantum states.

States may be given as density matrices or as kets; pure-state shortcuts are
used whenever a ket is supplied.
"""

from __future__ import annotations

import numpy as np

from .core import _dense, as_dm, is_hermitian, psd_sqrt

__all__ = ["root_fidelity", "fidelity", "trace_norm", "trace_distance",
           "purity", "hilbert_schmidt_distance"]


def root_fidelity(rho, sigma) -> float:
    """Root fidelity ``Tr sqrt(sqrt(rho) sigma sqrt(rho))`` (Mathematica ``Fidelity``).

    Computed as the nuclear norm of ``sqrt(rho) sqrt(sigma)``, which is
    numerically stable for rank-deficient states.  :func:`fidelity` is its square.
    """
    a, b = _dense(rho), _dense(sigma)
    if a.ndim == 1 and b.ndim == 1:
        return float(abs(np.vdot(a, b)))
    if a.ndim == 1 or b.ndim == 1:
        psi, m = (a, b) if a.ndim == 1 else (b, a)
        return float(np.sqrt(max(np.vdot(psi, m @ psi).real, 0.0)))
    return float(np.linalg.svd(psd_sqrt(a) @ psd_sqrt(b), compute_uv=False).sum())


def fidelity(rho, sigma) -> float:
    """Uhlmann fidelity ``(Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2`` (``QuantumFidelity``)."""
    return root_fidelity(rho, sigma) ** 2


def trace_norm(a) -> float:
    """Sum of the singular values (absolute eigenvalues for Hermitian input)."""
    a = _dense(a)
    if is_hermitian(a):
        return float(np.abs(np.linalg.eigvalsh(a)).sum())
    return float(np.linalg.svd(a, compute_uv=False).sum())


def trace_distance(rho, sigma) -> float:
    """``||rho - sigma||_1 / 2``."""
    return 0.5 * trace_norm(as_dm(rho) - as_dm(sigma))


def hilbert_schmidt_distance(rho, sigma) -> float:
    """Frobenius norm ``||rho - sigma||_2``."""
    return float(np.linalg.norm(as_dm(rho) - as_dm(sigma)))


def purity(rho):
    """``Tr rho^2`` in O(d^2) operations; works on stacks ``(..., d, d)``."""
    rho = _dense(rho)
    if rho.ndim == 1:
        return float(np.vdot(rho, rho).real ** 2)
    p = np.einsum("...ij,...ji->...", rho, rho)
    if np.allclose(np.imag(p), 0):
        p = np.real(p)
    return p.item() if np.ndim(p) == 0 else p
