"""Operator bases: generalized Gell-Mann matrices, orthonormal Hermitian bases,
Pauli-string bases, matrix units and generalized Bloch vectors.

The bases are returned as read-only stacks of shape ``(d**2, d, d)`` and are
cached, so repeated calls (e.g. inside channel conversions) cost nothing.
Indices are 0-based: ``gell_mann_symmetric(j, k, n)`` needs ``0 <= j < k < n``
and ``gell_mann_diagonal(l, n)`` needs ``1 <= l < n``.
"""

from __future__ import annotations

from functools import lru_cache, reduce
from itertools import product

import numpy as np

from .core import _dense, _frozen, _square_dim, as_dm, id2, paulis

__all__ = [
    "gell_mann_symmetric", "gell_mann_antisymmetric", "gell_mann_diagonal",
    "gell_mann", "hermitian_basis", "pauli_basis", "matrix_units",
    "basis_coefficients", "from_coefficients", "state_to_bloch", "bloch_to_state",
]


def gell_mann_symmetric(j: int, k: int, n: int) -> np.ndarray:
    """``|j><k| + |k><j|`` (``Lambda1``), for ``0 <= j < k < n``."""
    if not 0 <= j < k < n:
        raise ValueError("need 0 <= j < k < n")
    out = np.zeros((n, n), dtype=complex)
    out[j, k] = out[k, j] = 1
    return out


def gell_mann_antisymmetric(j: int, k: int, n: int) -> np.ndarray:
    """``-i (|j><k| - |k><j|)`` (``Lambda2``), for ``0 <= j < k < n``."""
    if not 0 <= j < k < n:
        raise ValueError("need 0 <= j < k < n")
    out = np.zeros((n, n), dtype=complex)
    out[j, k], out[k, j] = -1j, 1j
    return out


def gell_mann_diagonal(l: int, n: int) -> np.ndarray:
    """``sqrt(2/(l(l+1))) diag(1, ..., 1, -l, 0, ..., 0)`` with ``l`` ones (``Lambda3[l+1, n]``), ``1 <= l < n``."""
    if not 1 <= l < n:
        raise ValueError("need 1 <= l < n")
    diag = np.zeros(n)
    diag[:l], diag[l] = 1, -l
    return np.diag(np.sqrt(2 / (l * (l + 1))) * diag).astype(complex)


@lru_cache(maxsize=None)
def gell_mann(n: int) -> np.ndarray:
    """The ``n**2 - 1`` generalized Gell-Mann matrices (``GeneralizedPauliMatrices``).

    Traceless, Hermitian, ``Tr(l_i l_j) = 2 delta_ij``; ordered as all the
    symmetric ones, then the antisymmetric ones, then the diagonal ones.
    For ``n = 2`` they are ``(sx, sy, sz)``.
    """
    j, k = np.triu_indices(n, 1)
    m = np.arange(len(j))
    sym = np.zeros((len(j), n, n), dtype=complex)
    sym[m, j, k] = sym[m, k, j] = 1
    anti = np.zeros((len(j), n, n), dtype=complex)
    anti[m, j, k], anti[m, k, j] = -1j, 1j
    diag = np.array([gell_mann_diagonal(l, n) for l in range(1, n)]).reshape(-1, n, n)
    return _frozen(np.concatenate([sym, anti, diag]))


@lru_cache(maxsize=None)
def hermitian_basis(n: int) -> np.ndarray:
    """Orthonormal Hermitian basis ``{I/sqrt(n), l_1/sqrt(2), ...}`` of ``n x n`` matrices (``GEN``)."""
    return _frozen(np.concatenate([np.eye(n, dtype=complex)[None] / np.sqrt(n),
                                   gell_mann(n) / np.sqrt(2)]))


@lru_cache(maxsize=None)
def pauli_basis(n_qubits: int) -> np.ndarray:
    """Orthonormal basis of normalized Pauli strings ``P_1 x ... x P_n / sqrt(2^n)``.

    Ordered lexicographically in ``(I, X, Y, Z)``; in this basis the process
    matrix of an ``n``-qubit channel is its Pauli transfer matrix.
    """
    single = np.concatenate([np.asarray(id2, dtype=complex)[None], paulis])
    ops = [reduce(np.kron, combo) for combo in product(single, repeat=n_qubits)]
    return _frozen(np.array(ops) / np.sqrt(2 ** n_qubits))


@lru_cache(maxsize=None)
def matrix_units(n: int) -> np.ndarray:
    """The ``n**2`` matrix units ``|i><j|`` in row-major order (``TAU``)."""
    return _frozen(np.eye(n * n).reshape(n * n, n, n))


def _basis(d: int, basis=None) -> np.ndarray:
    if basis is None:
        return hermitian_basis(d)
    b = np.asarray(basis)
    if b.shape != (d * d, d, d):
        raise ValueError(f"basis must be a stack of {d * d} matrices of size {d}x{d}, got {b.shape}")
    return b


def _real_if_close(x):
    x = np.asarray(x)
    return x.real if np.iscomplexobj(x) and np.allclose(x.imag, 0, atol=1e-13) else x


def basis_coefficients(a, basis=None) -> np.ndarray:
    """Coefficients ``Tr(G_k† A)`` of ``A`` (or of a stack of matrices) in an orthonormal basis (``Vec2``).

    The default basis is :func:`hermitian_basis`; Hermitian ``A`` gives real coefficients.
    """
    a = _dense(a)
    b = _basis(a.shape[-1], basis)
    return _real_if_close(np.einsum("kij,...ij->...k", b.conj(), a))


def from_coefficients(c, basis=None) -> np.ndarray:
    """``sum_k c_k G_k``, the inverse of :func:`basis_coefficients` (``UnVec2``)."""
    c = np.asarray(c)
    b = _basis(_square_dim(c.shape[-1], "coefficient vector length"), basis)
    return np.einsum("...k,kij->...ij", c, b)


def state_to_bloch(state) -> np.ndarray:
    """Generalized Bloch vector ``sqrt(d) (Tr rho G_1, ..., Tr rho G_{d^2})`` (``StateToBloch``).

    The first component is ``Tr rho``; for a qubit the result is
    ``(1, <sx>, <sy>, <sz>)``.  Kets are accepted.
    """
    rho = as_dm(state)
    return np.sqrt(rho.shape[-1]) * basis_coefficients(rho)


def bloch_to_state(v) -> np.ndarray:
    """Inverse of :func:`state_to_bloch` (``BlochToState``).

    A vector of length ``d**2 - 1`` is taken to omit the leading component 1,
    so ``bloch_to_state([x, y, z])`` is the qubit state ``(I + r.sigma)/2``.
    """
    v = np.asarray(v)
    n = v.shape[-1]
    d = int(round(np.sqrt(n)))
    if n == 0:
        raise ValueError("empty Bloch vector")
    if d * d != n:
        d = int(round(np.sqrt(n + 1)))
        if d * d != n + 1:
            raise ValueError(f"length {n} is neither d**2 nor d**2 - 1")
        v = np.concatenate([np.ones(v.shape[:-1] + (1,)), v], axis=-1)
    return from_coefficients(v, hermitian_basis(d)) / np.sqrt(d)
