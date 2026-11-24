"""Entropies and information measures, quantum and classical.

All logarithms are base 2 unless ``base`` is given (``base=np.e`` gives nats).
Eigenvalues and probabilities not larger than ``tol`` contribute nothing
(``0 log 0 = 0``).  Quantum functions take density matrices or kets and also
accept stacks of density matrices ``(..., d, d)``, returning an array.
Density matrices are assumed Hermitian (``numpy.linalg.eigvalsh`` is used).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import ATOL, as_dm
from .subsystems import _resolve_dims, _resolve_sys, partial_trace

__all__ = [
    "von_neumann", "renyi", "relative_entropy",
    "shannon_entropy", "binary_entropy", "marginal",
    "classical_mutual_information", "conditional_entropy",
    "mutual_information", "tripartite_information", "quantum_conditional_entropy",
]


def _scale(base) -> float:
    if base == 2:
        return 1 / np.log(2)
    if base is None or base == np.e:
        return 1.0
    return 1 / np.log(base)


def _out(x):
    x = np.asarray(x)
    return x.item() if x.ndim == 0 else x


def _entropy_terms(p, tol):
    """``-p log p`` elementwise, zero where ``p <= tol``."""
    safe = np.where(p > tol, p, 1.0)
    return np.where(p > tol, -p * np.log(safe), 0.0)


def _spectrum(rho) -> np.ndarray:
    return np.linalg.eigvalsh(as_dm(rho))


# --------------------------------------------------------------------------
# Quantum entropies
# --------------------------------------------------------------------------

def von_neumann(rho, base: float = 2, tol: float = ATOL):
    """Von Neumann entropy ``-Tr rho log rho``."""
    if np.ndim(rho) == 1:
        return 0.0
    return _out(_entropy_terms(_spectrum(rho), tol).sum(-1) * _scale(base))


def renyi(rho, alpha: float, base: float = 2, tol: float = ATOL):
    """Rényi entropy ``log(Tr rho^alpha) / (1 - alpha)`` over the support of ``rho``.

    ``alpha = 1`` gives the von Neumann entropy, ``alpha = np.inf`` the
    min-entropy ``-log(lambda_max)`` and ``alpha = 0`` the log of the rank.
    """
    if alpha == 1:
        return von_neumann(rho, base, tol)
    w = _spectrum(rho)
    if np.isinf(alpha):
        return _out(-np.log(w.max(-1)) * _scale(base))
    powers = np.where(w > tol, np.where(w > tol, w, 1.0) ** alpha, 0.0)
    return _out(np.log(powers.sum(-1)) / (1 - alpha) * _scale(base))


def relative_entropy(rho, sigma, base: float = 2, tol: float = ATOL) -> float:
    """Quantum relative entropy ``S(rho||sigma) = Tr rho (log rho - log sigma)`` (``KullbackLeibler``).

    It is ``inf`` when the support of ``rho`` is not contained in that of ``sigma``.
    """
    rho, sigma = as_dm(rho), as_dm(sigma)
    lam, v = np.linalg.eigh(sigma)
    w = np.einsum("ji,jk,ki->i", v.conj(), rho, v).real      # <v_i|rho|v_i>
    if np.any((w > tol) & (lam <= tol)):
        return np.inf
    mask = w > tol
    cross = -(w[mask] * np.log(lam[mask])).sum() * _scale(base)
    return float(cross - von_neumann(rho, base, tol))


# --------------------------------------------------------------------------
# Classical information
# --------------------------------------------------------------------------

def shannon_entropy(probs, base: float = 2, tol: float = ATOL):
    """Shannon entropy of a probability vector (of each row, for a 2-D input)."""
    return _out(_entropy_terms(np.asarray(probs, dtype=float), tol).sum(-1) * _scale(base))


def binary_entropy(p, base: float = 2, tol: float = ATOL):
    """``h(p) = -p log p - (1 - p) log(1 - p)``, elementwise."""
    p = np.asarray(p, dtype=float)
    return _out((_entropy_terms(p, tol) + _entropy_terms(1 - p, tol)) * _scale(base))


def _joint(probs, sizes) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if sizes is None:
        return p if p.ndim > 1 else p.reshape(2, 2)
    return p.reshape(tuple(sizes))


def marginal(probs, sizes: Sequence[int] | None, keep: int | Sequence[int]) -> np.ndarray:
    """Marginal distribution of the variables ``keep`` (``marginalProbabilityDistribution``).

    ``probs`` is the joint distribution, either as an array with one axis per
    variable or flattened in row-major order with the variable sizes in
    ``sizes``.  Variables are numbered from 0.
    """
    p = np.asarray(probs, dtype=float)
    if sizes is not None:
        p = p.reshape(tuple(sizes))
    keep = _resolve_sys(keep, p.ndim)
    return p.sum(axis=tuple(i for i in range(p.ndim) if i not in keep))


def classical_mutual_information(probs, sizes: Sequence[int] | None = None,
                                 base: float = 2) -> float:
    """``H(X) + H(Y) - H(X, Y)`` of a joint distribution ``P[x, y]``.

    ``probs`` is a 2-D table or a flat row-major list (``sizes`` defaults to (2, 2)).
    """
    p = _joint(probs, sizes)
    return float(shannon_entropy(p.sum(1), base) + shannon_entropy(p.sum(0), base)
                 - shannon_entropy(p.ravel(), base))


def conditional_entropy(probs, sizes: Sequence[int] | None = None, given: int = 1,
                        base: float = 2) -> float:
    """Conditional Shannon entropy ``H(X|Y) = H(X, Y) - H(Y)`` (``given=1``) or ``H(Y|X)`` (``given=0``).

    ``probs`` as in :func:`classical_mutual_information`.
    """
    p = _joint(probs, sizes)
    return float(shannon_entropy(p.ravel(), base) - shannon_entropy(p.sum(1 - given), base))


# --------------------------------------------------------------------------
# Quantum information
# --------------------------------------------------------------------------

def _chop(x, tol=1e-12):
    x = np.asarray(x, dtype=float)
    return _out(np.where(np.abs(x) < tol, 0.0, x))


def _entropy_of(rho, dims, keep, base):
    if len(keep) == len(dims):
        return von_neumann(rho, base)
    return von_neumann(partial_trace(rho, dims, keep=keep), base)


def mutual_information(rho, dims: Sequence[int] = (2, 2), a=0, b=None, base: float = 2):
    """Quantum mutual information ``I(A:B) = S(A) + S(B) - S(AB)``.

    ``a`` and ``b`` are disjoint (lists of) subsystems; by default ``a = 0`` and
    ``b`` is everything else, so for ``dims=(dA, dB)`` it is the usual bipartite
    mutual information.  Subsystems outside ``a`` and ``b`` are traced out.
    """
    dims = _resolve_dims(dims, np.shape(rho)[-1])
    n = len(dims)
    a = _resolve_sys(a, n)
    b = tuple(i for i in range(n) if i not in a) if b is None else _resolve_sys(b, n)
    if set(a) & set(b):
        raise ValueError(f"parts {a} and {b} overlap")
    ab = tuple(sorted(a + b))
    return _chop(_entropy_of(rho, dims, a, base) + _entropy_of(rho, dims, b, base)
                 - _entropy_of(rho, dims, ab, base))


def tripartite_information(rho, dims: Sequence[int] = (2, 2, 2), base: float = 2):
    """``I(A:B) + I(A:C) - I(A:BC) = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC``."""
    dims = _resolve_dims(dims, np.shape(rho)[-1])
    if len(dims) != 3:
        raise ValueError("tripartite_information needs three subsystems")
    s = lambda *k: _entropy_of(rho, dims, k, base)
    return _chop(s(0) + s(1) + s(2) - s(0, 1) - s(0, 2) - s(1, 2) + s(0, 1, 2))


def quantum_conditional_entropy(rho, dims: Sequence[int] = (2, 2), given=1, base: float = 2):
    """``S(A|B) = S(AB) - S(B)`` where ``B`` are the subsystems ``given``."""
    dims = _resolve_dims(dims, np.shape(rho)[-1])
    given = _resolve_sys(given, len(dims))
    return _out(von_neumann(rho, base) - _entropy_of(rho, dims, given, base))
