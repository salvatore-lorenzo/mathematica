"""Operations on composite systems: partial trace, partial transpose,
subsystem permutation and reshuffling (realignment).

Subsystems are numbered from 0 and negative indices count from the end, as
in NumPy.  Everything is done with reshapes and axis permutations, so there
is a single algorithm for any number of subsystems and any dimensions.  Dense
input may be a stack of operators ``(..., D, D)``; sparse input stays sparse.
The functions only reshape, permute and add entries, so they also work on
NumPy object arrays holding SymPy expressions (see :mod:`myqubits.symbolic`).
"""

from __future__ import annotations

from math import prod
from typing import Sequence

import numpy as np
import scipy.sparse as sps

from .core import _check_square, _square_dim

__all__ = ["partial_trace", "partial_transpose", "permute_subsystems", "reshuffle"]


def _resolve_dims(dims, size: int) -> tuple[int, ...]:
    if dims is None:
        d = _square_dim(size, "size")
        return (d, d)
    dims = tuple(int(x) for x in np.atleast_1d(dims))
    if any(x <= 0 for x in dims):
        raise ValueError(f"subsystem dimensions must be positive, got {dims}")
    if prod(dims) != size:
        raise ValueError(f"subsystem dimensions {dims} are not compatible with size {size}")
    return dims


def _resolve_sys(sys, n: int) -> tuple[int, ...]:
    sys = [int(s) for s in np.atleast_1d(np.asarray(sys, dtype=int))]
    bad = [s for s in sys if not -n <= s < n]
    if bad:
        raise ValueError(f"a {n}-partite system has no subsystems {bad}")
    return tuple(sorted({s % n for s in sys}))


def _split(dims, sys, keep):
    n = len(dims)
    if (sys is None) == (keep is None):
        raise TypeError("give exactly one of `sys` (subsystems to trace out) or `keep`")
    if keep is not None:
        keep = _resolve_sys(keep, n)
        sys = tuple(i for i in range(n) if i not in keep)
    else:
        sys = _resolve_sys(sys, n)
        keep = tuple(i for i in range(n) if i not in sys)
    return sys, keep


def partial_trace(a, dims: Sequence[int] | None = None, sys=None, *, keep=None):
    """Trace out the subsystems ``sys`` (an index or a list of indices).

    Parameters
    ----------
    a : ket (1-D), operator ``(D, D)``, stack of operators ``(..., D, D)`` or sparse array
    dims : subsystem dimensions ``(d0, d1, ...)``; ``None`` means two subsystems of equal dimension
    sys : subsystems to trace out
    keep : alternatively, the subsystems to keep (exactly one of ``sys``/``keep``)

    A ket gives the reduced density matrix directly from the amplitudes,
    without building ``|psi><psi|``.  Tracing out every subsystem gives the
    trace.  ``partial_trace(rho, [2, 2], 0)`` is ``Tr_A rho``.
    """
    if sps.issparse(a):
        return _partial_trace_sparse(a, dims, sys, keep)
    a = np.asarray(a)
    dims = _resolve_dims(dims, a.shape[-1])
    sys, keep = _split(dims, sys, keep)
    dk = prod(dims[i] for i in keep)
    dt = prod(dims[i] for i in sys)

    if a.ndim == 1:                                   # ket
        m = a.reshape(dims).transpose(keep + sys).reshape(dk, dt)
        rho = m @ m.conj().T
        return rho if keep else rho[0, 0]

    _check_square(a, "operator")
    if not sys:
        return a
    if not keep:
        return np.trace(a, axis1=-2, axis2=-1)
    batch = a.shape[:-2]
    nb, n = len(batch), len(dims)
    order = keep + sys
    perm = list(range(nb)) + [nb + i for i in order] + [nb + n + i for i in order]
    t = a.reshape(batch + dims + dims).transpose(perm).reshape(batch + (dk, dt, dk, dt))
    return np.trace(t, axis1=-3, axis2=-1)


def _partial_trace_sparse(a, dims, sys, keep):
    coo = sps.coo_array(a)
    _check_square(coo, "operator")
    dims = _resolve_dims(dims, coo.shape[0])
    sys, keep = _split(dims, sys, keep)
    r = np.unravel_index(coo.coords[0], dims)
    c = np.unravel_index(coo.coords[1], dims)
    mask = np.ones(coo.nnz, dtype=bool)
    for s in sys:
        mask &= r[s] == c[s]
    data = coo.data[mask]
    if not keep:
        return data.sum()
    kd = tuple(dims[i] for i in keep)
    rows = np.ravel_multi_index(tuple(r[i][mask] for i in keep), kd)
    cols = np.ravel_multi_index(tuple(c[i][mask] for i in keep), kd)
    dk = prod(kd)
    return sps.csr_array((data, (rows, cols)), shape=(dk, dk))   # duplicates are summed


def partial_transpose(a, dims: Sequence[int] | None = None, sys=0):
    """Partial transpose with respect to the subsystems ``sys`` (default: the first)."""
    if sps.issparse(a):
        coo = sps.coo_array(a)
        _check_square(coo, "operator")
        dims = _resolve_dims(dims, coo.shape[0])
        sys = _resolve_sys(sys, len(dims))
        r = list(np.unravel_index(coo.coords[0], dims))
        c = list(np.unravel_index(coo.coords[1], dims))
        for s in sys:
            r[s], c[s] = c[s], r[s]
        return sps.csr_array((coo.data, (np.ravel_multi_index(r, dims),
                                         np.ravel_multi_index(c, dims))), shape=coo.shape)
    a = np.asarray(a)
    _check_square(a, "operator")
    dims = _resolve_dims(dims, a.shape[-1])
    sys = _resolve_sys(sys, len(dims))
    if not sys:
        return a
    batch = a.shape[:-2]
    nb, n = len(batch), len(dims)
    perm = list(range(nb + 2 * n))
    for s in sys:
        perm[nb + s], perm[nb + n + s] = perm[nb + n + s], perm[nb + s]
    return a.reshape(batch + dims + dims).transpose(perm).reshape(a.shape)


def permute_subsystems(a, dims: Sequence[int], perm: Sequence[int]):
    """Reorder the tensor factors: subsystem ``k`` of the result is subsystem ``perm[k]`` of ``a``.

    Works on kets and on (stacks of) dense operators.  For instance
    ``permute_subsystems(rho, [2, 3], [1, 0])`` swaps a qubit and a qutrit.
    """
    a = np.asarray(a)
    dims = _resolve_dims(dims, a.shape[-1])
    n = len(dims)
    perm = [int(p) % n for p in perm]
    if sorted(perm) != list(range(n)):
        raise ValueError(f"{perm} is not a permutation of {n} subsystems")
    if a.ndim == 1:
        return a.reshape(dims).transpose(perm).reshape(-1)
    _check_square(a, "operator")
    batch = a.shape[:-2]
    nb = len(batch)
    axes = list(range(nb)) + [nb + p for p in perm] + [nb + n + p for p in perm]
    return a.reshape(batch + dims + dims).transpose(axes).reshape(a.shape)


def reshuffle(a, dims=None):
    """Realignment ``R[(i,j),(k,l)] = A[(i,k),(j,l)]`` (``Reshuffle``).

    ``dims=((n1, n2), (m1, m2))`` reshuffles an ``(n1 n2) x (m1 m2)`` matrix
    made of ``n2 x m2`` blocks into an ``(n1 m1) x (n2 m2)`` one: the
    row-flattened block ``(i, k)`` becomes row ``i m1 + k``.  By default both
    the row and the column dimension are split into two equal factors.
    """
    a = a.toarray() if sps.issparse(a) else np.asarray(a)
    r, c = a.shape[-2:]
    if dims is None:
        n1 = n2 = _square_dim(r, "number of rows")
        m1 = m2 = _square_dim(c, "number of columns")
    else:
        (n1, n2), (m1, m2) = dims
        if n1 * n2 != r or m1 * m2 != c:
            raise ValueError(f"dims {dims} do not match a matrix of shape {(r, c)}")
    batch = a.shape[:-2]
    nb = len(batch)
    axes = list(range(nb)) + [nb, nb + 2, nb + 1, nb + 3]
    return a.reshape(batch + (n1, n2, m1, m2)).transpose(axes).reshape(batch + (n1 * m1, n2 * m2))
