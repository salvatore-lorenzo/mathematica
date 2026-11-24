"""Multi-qubit registers, spin-chain Hamiltonians and single-qubit Bloch vectors.

Qubits are numbered from 0 (leftmost tensor factor).  Register operators are
CSR sparse arrays by default, since they have only ``2**n`` nonzero entries.
"""

from __future__ import annotations

from functools import cached_property
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sps

from . import core
from .core import _dense, as_dm, identity, kron

__all__ = ["local_operator", "qubit_operator", "multi_qubit_operator",
           "QubitRegister", "tight_binding", "bloch_vector"]


def local_operator(op, i: int, dims: Sequence[int], sparse: bool = True):
    """``I x ... x op x ... x I`` with ``op`` on subsystem ``i`` of a system with dimensions ``dims``."""
    dims = list(dims)
    n = len(dims)
    if not -n <= i < n:
        raise ValueError(f"subsystem {i} out of range for {n} subsystems")
    i %= n
    left, right = int(np.prod(dims[:i])), int(np.prod(dims[i + 1:]))
    op = sps.csr_array(op) if sparse else _dense(op)
    return kron(identity(left, sparse), op, identity(right, sparse))


def qubit_operator(op, i: int, n: int, sparse: bool = True):
    """Single-qubit operator ``op`` acting on qubit ``i`` of an ``n``-qubit register."""
    return local_operator(op, i, [2] * n, sparse)


def multi_qubit_operator(ops: Mapping[int, object], n: int, sparse: bool = True):
    """Tensor product with ``ops[i]`` on qubit ``i`` and the identity elsewhere.

    ``multi_qubit_operator({0: sx, 2: sz}, 3)`` is ``sx x I x sz``; one Kronecker
    chain instead of a product of full-size matrices.
    """
    ops = {int(i) % n: o for i, o in ops.items()}
    factors = [ops.get(i, core.id2) for i in range(n)]
    if sparse:
        factors = [sps.csr_array(f) for f in factors]
    return kron(*factors)


class QubitRegister:
    """Pauli and ladder operators of an ``n``-qubit register (``NQubits``).

    >>> reg = QubitRegister(3)
    >>> H = reg.sz[0] @ reg.sz[1] + reg.sx[2]

    ``reg.sx[i]``, ``reg.sy[i]``, ``reg.sz[i]``, ``reg.sp[i]`` and ``reg.sm[i]``
    are built on first access and cached.  Unlike Mathematica's ``NQubits``,
    several registers can coexist.
    """

    def __init__(self, n: int, sparse: bool = True):
        if n < 1:
            raise ValueError("n must be positive")
        self.n = n
        self.sparse = sparse

    def __repr__(self) -> str:
        return f"QubitRegister(n={self.n}, sparse={self.sparse})"

    @property
    def dim(self) -> int:
        return 2 ** self.n

    def op(self, a, i: int):
        """``a`` acting on qubit ``i``."""
        return qubit_operator(a, i, self.n, self.sparse)

    def product(self, ops: Mapping[int, object]):
        """See :func:`multi_qubit_operator`."""
        return multi_qubit_operator(ops, self.n, self.sparse)

    def identity(self):
        return identity(self.dim, self.sparse)

    def _all(self, a):
        return tuple(self.op(a, i) for i in range(self.n))

    @cached_property
    def sx(self):
        return self._all(core.sx)

    @cached_property
    def sy(self):
        return self._all(core.sy)

    @cached_property
    def sz(self):
        return self._all(core.sz)

    @cached_property
    def sp(self):
        return self._all(core.sp)

    @cached_property
    def sm(self):
        return self._all(core.sm)


def tight_binding(J, n: int, periodic: bool = False, sparse: bool = True):
    """XX chain ``sum_i J_i (sx_i sx_{i+1} + sy_i sy_{i+1})`` on ``n`` qubits (``TightBinding``).

    ``J`` is a scalar or a sequence of bond couplings (``n - 1`` of them, or
    ``n`` with ``periodic=True``, which adds the bond ``(n-1, 0)`` for
    ``n > 2``).  Since ``sx sx + sy sy = 2(|01><10| + |10><01|)``, the matrix
    is assembled directly from bit flips in ``O(n 2^n)`` operations, without
    any Kronecker product.
    """
    bonds = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        bonds.append((n - 1, 0))
    couplings = np.broadcast_to(np.asarray(J), (len(bonds),)) if np.ndim(J) == 0 else np.asarray(J)
    if len(couplings) != len(bonds):
        raise ValueError(f"expected {len(bonds)} couplings, got {len(couplings)}")
    states = np.arange(2 ** n)
    rows, cols, data = [], [], []
    for (i, j), c in zip(bonds, couplings):
        bi, bj = n - 1 - i, n - 1 - j                   # qubit 0 is the most significant bit
        s = states[(((states >> bi) ^ (states >> bj)) & 1) == 1]
        rows.append(s)
        cols.append(s ^ ((1 << bi) | (1 << bj)))
        data.append(np.full(len(s), 2 * c))
    shape = (2 ** n, 2 ** n)
    if not bonds:
        h = sps.csr_array(shape)
    else:
        h = sps.csr_array((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=shape)
    return h if sparse else h.toarray()


def bloch_vector(state) -> np.ndarray:
    """``(Tr rho sx, Tr rho sy, Tr rho sz)`` of a qubit ket, density matrix or stack ``(..., 2, 2)`` (``bv``)."""
    r = as_dm(state) if np.ndim(state) < 3 else np.asarray(state)
    if r.shape[-2:] != (2, 2):
        raise ValueError("a qubit state is required")
    v = np.stack([r[..., 0, 1] + r[..., 1, 0],
                  1j * (r[..., 0, 1] - r[..., 1, 0]),
                  r[..., 0, 0] - r[..., 1, 1]], axis=-1)
    return np.real_if_close(v, tol=1000)
