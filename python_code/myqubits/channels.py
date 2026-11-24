"""Quantum channels and their representations.

Representations of a linear map ``Phi`` on ``d x d`` matrices:

* **Kraus** operators: a stack ``(k, d_out, d_in)`` with ``Phi(X) = sum_k K X K†``.
* **superoperator** ``L`` acting on column-stacked vectors:
  ``vec(Phi(X)) = L @ vec(X)``; for Kraus operators ``L = sum_k conj(K) x K``.
* **Choi** matrix ``S = sum_ij Phi(|i><j|) x |i><j|`` (output space first, as
  in MyQubits); ``Phi`` is completely positive iff ``S >= 0``.
* **process matrix** ``F[a, b] = Tr(G_a† Phi(G_b))`` in an orthonormal
  operator basis ``G`` (default :func:`myqubits.bases.hermitian_basis`; for a
  qubit it is the Pauli transfer matrix).

All conversions go through the superoperator, which is related to the
Choi matrix by an index permutation and to the process matrix by a unitary
change of basis, so they cost a reshape or a matrix product instead of
``d**2`` applications of the map.  The :class:`Channel` class wraps them.
"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np

from .bases import _basis
from .core import (ATOL, _dense, _operator_stack, _square_dim, id2, paulis,
                   sm, superop_from_map, unvec, vec)

__all__ = [
    # standard channels
    "depolarizing", "amplitude_damping", "amplitude_damping_kraus",
    "pauli_channel", "pauli_kraus",
    # applying a channel
    "apply_kraus", "apply_superop", "apply_choi", "apply_ptm",
    # conversions
    "kraus_to_superop", "kraus_to_choi", "kraus_to_ptm",
    "choi_to_kraus", "choi_to_superop", "choi_to_ptm",
    "superop_to_choi", "superop_to_ptm", "superop_to_kraus",
    "ptm_to_superop", "ptm_to_choi", "ptm_to_kraus",
    "map_to_superop", "map_to_choi", "map_to_ptm", "map_to_kraus",
    "Channel",
]


# --------------------------------------------------------------------------
# Standard channels (they act on single matrices or on stacks of matrices)
# --------------------------------------------------------------------------

def depolarizing(rho, p: float):
    """``(1 - p) rho + p Tr(rho) I/d`` (``Depo``)."""
    rho = _dense(rho)
    d = rho.shape[-1]
    tr = np.trace(rho, axis1=-2, axis2=-1)[..., None, None]
    return (1 - p) * rho + p * tr * np.eye(d) / d


def amplitude_damping_kraus(gamma: float) -> np.ndarray:
    """Kraus operators ``diag(1, sqrt(1 - gamma))`` and ``sqrt(gamma) sm`` (decay ``|1> -> |0>``)."""
    return np.array([[[1.0, 0.0], [0.0, np.sqrt(1 - gamma)]], np.sqrt(gamma) * sm])


def amplitude_damping(rho, gamma: float):
    """Qubit amplitude damping with decay probability ``gamma`` (``AD``)."""
    return apply_kraus(amplitude_damping_kraus(gamma), rho)


def pauli_kraus(px: float, py: float, pz: float) -> np.ndarray:
    """Kraus operators of :func:`pauli_channel`."""
    w = np.sqrt(np.array([1 - (px + py + pz) / 4, px / 4, py / 4, pz / 4]))
    return w[:, None, None] * np.concatenate([np.asarray(id2)[None], paulis])


def pauli_channel(rho, probs: Sequence[float]):
    """``(1 - (px+py+pz)/4) rho + (px sx rho sx + py sy rho sy + pz sz rho sz)/4`` (``Pauli``).

    ``p_k/4`` is the probability of a ``sigma_k`` error, so that
    ``pauli_channel(rho, (p, p, p)) == depolarizing(rho, p)``.
    """
    px, py, pz = probs
    rho = _dense(rho)
    out = (1 - (px + py + pz) / 4) * rho
    for pk, s in zip((px, py, pz), paulis):
        out = out + pk / 4 * (s @ rho @ s)
    return out


# --------------------------------------------------------------------------
# Applying a channel
# --------------------------------------------------------------------------

def apply_kraus(kraus, rho):
    """``sum_k K_k rho K_k†`` (``KrausToPhi``); ``rho`` may be a stack of matrices."""
    k = _operator_stack(kraus)
    return np.einsum("kij,...jl,kml->...im", k, _dense(rho), k.conj(), optimize=True)


def apply_superop(superop, rho, d_out: int | None = None):
    """``unvec(L @ vec(rho))``; ``rho`` may be a stack of matrices."""
    out = vec(rho) @ _dense(superop).T
    return unvec(out, d_out)


def apply_choi(choi, rho, dims: tuple[int, int] | None = None):
    """``Phi(rho) = Tr_2[S (I x rho^T)] = sum_ij rho_ij S[:, i, :, j]``."""
    s = _dense(choi)
    d_out, d_in = _choi_dims(s, dims)
    return np.einsum("ricj,...ij->...rc", s.reshape(d_out, d_in, d_out, d_in), _dense(rho))


def apply_ptm(ptm, rho, basis=None):
    """Apply the channel with process matrix ``F`` (``FToPhi``)."""
    rho = _dense(rho)
    b = _basis(rho.shape[-1], basis)
    c = np.einsum("kij,...ij->...k", b.conj(), rho)
    return np.einsum("...k,kij->...ij", c @ np.asarray(ptm).T, b)


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------

def _choi_dims(s, dims):
    if dims is not None:
        return tuple(dims)
    d = _square_dim(s.shape[0], "Choi matrix size")
    return d, d


def _superop_dims(lsup):
    return (_square_dim(lsup.shape[0], "superoperator rows"),
            _square_dim(lsup.shape[1], "superoperator columns"))


def kraus_to_superop(kraus) -> np.ndarray:
    """``sum_k conj(K_k) x K_k`` (Kraus operators may be rectangular)."""
    k = _operator_stack(kraus)
    _, m, n = k.shape
    return np.einsum("kab,kcd->acbd", k.conj(), k).reshape(m * m, n * n)


def kraus_to_choi(kraus) -> np.ndarray:
    """``sum_k res(K_k) res(K_k)†`` (``KrausToS``)."""
    k = _operator_stack(kraus)
    r = k.reshape(k.shape[0], -1)
    return r.T @ r.conj()


def kraus_to_ptm(kraus, basis=None) -> np.ndarray:
    """Process matrix of a Kraus channel (``KrausToF``)."""
    return superop_to_ptm(kraus_to_superop(kraus), basis)


def choi_to_kraus(choi, dims: tuple[int, int] | None = None, tol: float = ATOL) -> np.ndarray:
    """Minimal Kraus decomposition from the Choi matrix (``SToKraus``).

    Each eigenvector ``v`` with eigenvalue ``|lambda| > tol`` gives
    ``sqrt(|lambda|) unres(v)``; ``dims = (d_out, d_in)`` defaults to square.
    A warning is issued if the map is not completely positive.
    """
    s = _dense(choi)
    d_out, d_in = _choi_dims(s, dims)
    w, v = np.linalg.eigh(s)
    if (w < -max(tol, 1e-10 * abs(w).max())).any():
        warnings.warn("the Choi matrix has negative eigenvalues: the map is not completely "
                      "positive; absolute values of the eigenvalues are used", stacklevel=2)
    keep = np.flatnonzero(np.abs(w) > tol)[::-1]          # largest eigenvalue first
    return (np.sqrt(np.abs(w[keep])) * v[:, keep]).T.reshape(-1, d_out, d_in)


def superop_to_choi(superop) -> np.ndarray:
    """Choi matrix from the column-stacking superoperator (an index permutation)."""
    lsup = _dense(superop)
    d_out, d_in = _superop_dims(lsup)
    l4 = lsup.reshape(d_out, d_out, d_in, d_in)            # l4[c, r, j, i] = Phi(|i><j|)[r, c]
    return l4.transpose(1, 3, 0, 2).reshape(d_out * d_in, d_out * d_in)


def choi_to_superop(choi, dims: tuple[int, int] | None = None) -> np.ndarray:
    """Inverse of :func:`superop_to_choi`."""
    s = _dense(choi)
    d_out, d_in = _choi_dims(s, dims)
    s4 = s.reshape(d_out, d_in, d_out, d_in)
    return s4.transpose(2, 0, 3, 1).reshape(d_out * d_out, d_in * d_in)


def _basis_columns(d, basis):
    return vec(_basis(d, basis)).T            # column b is vec(G_b)


def superop_to_ptm(superop, basis=None) -> np.ndarray:
    """``F = V† L V`` with ``V`` the matrix whose columns are ``vec(G_b)``."""
    lsup = _dense(superop)
    d = _square_dim(lsup.shape[1], "superoperator columns")
    v = _basis_columns(d, basis)
    return _real(v.conj().T @ lsup @ v)


def ptm_to_superop(ptm, basis=None) -> np.ndarray:
    """``L = V F V†``, the inverse of :func:`superop_to_ptm`."""
    f = np.asarray(ptm)
    v = _basis_columns(_square_dim(f.shape[0], "process matrix size"), basis)
    return v @ f @ v.conj().T


def superop_to_kraus(superop, tol: float = ATOL) -> np.ndarray:
    lsup = _dense(superop)
    return choi_to_kraus(superop_to_choi(lsup), _superop_dims(lsup), tol)


def choi_to_ptm(choi, basis=None) -> np.ndarray:
    """``SToF``."""
    return superop_to_ptm(choi_to_superop(choi), basis)


def ptm_to_choi(ptm, basis=None) -> np.ndarray:
    """``FToS``."""
    return superop_to_choi(ptm_to_superop(ptm, basis))


def ptm_to_kraus(ptm, basis=None, tol: float = ATOL) -> np.ndarray:
    """``FToKraus``."""
    return choi_to_kraus(ptm_to_choi(ptm, basis), tol=tol)


def map_to_superop(fn: Callable, d: int) -> np.ndarray:
    """Superoperator of a Python function acting on ``d x d`` matrices."""
    return superop_from_map(fn, d)


def map_to_choi(fn: Callable, d: int) -> np.ndarray:
    return superop_to_choi(superop_from_map(fn, d))


def map_to_ptm(fn: Callable, d: int | None = None, basis=None) -> np.ndarray:
    """Process matrix of a Python function (``PhiToF``); ``d`` may be inferred from ``basis``."""
    if d is None:
        if basis is None:
            raise ValueError("give the dimension d or an operator basis")
        d = np.shape(basis)[-1]
    return superop_to_ptm(superop_from_map(fn, d), basis)


def map_to_kraus(fn: Callable, d: int, tol: float = ATOL) -> np.ndarray:
    """Minimal Kraus operators of a completely positive Python function (``PhiToKraus``)."""
    return choi_to_kraus(map_to_choi(fn, d), tol=tol)


def _real(x):
    return x.real if np.allclose(x.imag, 0, atol=1e-13) else x


# --------------------------------------------------------------------------
# Channel object
# --------------------------------------------------------------------------

class Channel:
    """A linear map on matrices, stored as its column-stacking superoperator.

    >>> ch = Channel.from_kraus(amplitude_damping_kraus(0.3))
    >>> ch(rho)                      # apply
    >>> (ch @ ch).choi()             # composition, then Choi matrix
    >>> ch.ptm(), ch.kraus(), ch.is_cptp()
    """

    __slots__ = ("superop", "dims")

    def __init__(self, superop, dims: tuple[int, int] | None = None):
        self.superop = _dense(superop)
        self.dims = tuple(dims) if dims is not None else _superop_dims(self.superop)  # (d_out, d_in)

    # constructors -----------------------------------------------------
    @classmethod
    def from_kraus(cls, kraus) -> "Channel":
        k = _operator_stack(kraus)
        return cls(kraus_to_superop(k), k.shape[1:])

    @classmethod
    def from_choi(cls, choi, dims: tuple[int, int] | None = None) -> "Channel":
        s = _dense(choi)
        return cls(choi_to_superop(s, dims), _choi_dims(s, dims))

    @classmethod
    def from_ptm(cls, ptm, basis=None) -> "Channel":
        return cls(ptm_to_superop(ptm, basis))

    @classmethod
    def from_map(cls, fn: Callable, d: int) -> "Channel":
        return cls(superop_from_map(fn, d))

    @classmethod
    def from_unitary(cls, u) -> "Channel":
        return cls.from_kraus(np.asarray(u)[None])

    @classmethod
    def identity(cls, d: int) -> "Channel":
        return cls(np.eye(d * d), (d, d))

    # behaviour ---------------------------------------------------------
    def __call__(self, rho):
        return apply_superop(self.superop, rho, self.dims[0])

    def __matmul__(self, other: "Channel") -> "Channel":
        """Composition: ``(self @ other)(rho) == self(other(rho))``."""
        if self.dims[1] != other.dims[0]:
            raise ValueError("dimension mismatch in channel composition")
        return Channel(self.superop @ other.superop, (self.dims[0], other.dims[1]))

    def __pow__(self, n: int) -> "Channel":
        return Channel(np.linalg.matrix_power(self.superop, n), self.dims)

    def __repr__(self) -> str:
        return f"Channel(d_in={self.dims[1]}, d_out={self.dims[0]})"

    # representations ---------------------------------------------------
    def choi(self) -> np.ndarray:
        return superop_to_choi(self.superop)

    def kraus(self, tol: float = ATOL) -> np.ndarray:
        return choi_to_kraus(self.choi(), self.dims, tol)

    def ptm(self, basis=None) -> np.ndarray:
        return superop_to_ptm(self.superop, basis)

    # properties --------------------------------------------------------
    def is_cp(self, atol: float = 1e-10) -> bool:
        return bool(np.linalg.eigvalsh(self.choi()).min() >= -atol)

    def is_tp(self, atol: float = 1e-10) -> bool:
        """``Tr Phi(X) = Tr X``, i.e. ``vec(I)^T L = vec(I)^T``."""
        d_out, d_in = self.dims
        return np.allclose(vec(np.eye(d_out)) @ self.superop, vec(np.eye(d_in)), atol=atol)

    def is_cptp(self, atol: float = 1e-10) -> bool:
        return self.is_cp(atol) and self.is_tp(atol)

    def is_unital(self, atol: float = 1e-10) -> bool:
        d_out, d_in = self.dims
        return np.allclose(self.superop @ vec(np.eye(d_in)), vec(np.eye(d_out)), atol=atol)
