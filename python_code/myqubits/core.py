"""Basic definitions shared by every other ``myqubits`` module.

Conventions (the same as the Mathematica package MyQubits 0.0.4, except for
0-based numbering):

* Kets are 1-D arrays; operators and density matrices are square 2-D arrays,
  either ``numpy.ndarray`` or, where sparsity pays off, ``scipy.sparse`` arrays.
* Computational basis: ``ket0 = |0> = [1, 0]`` and ``ket1 = |1> = [0, 1]``;
  ``sm = |0><1|`` maps ``|1>`` to ``|0>`` and ``sp = |1><0|`` maps ``|0>`` to ``|1>``.
* Tensor products are ordered left to right: in ``kron(A, B)`` the operator
  ``A`` acts on subsystem 0.
* Subsystems and qubits are numbered from 0 (Python convention); the
  Mathematica package numbered them from 1.
* ``vec``/``unvec`` stack columns and ``res``/``unres`` stack rows.  Every
  superoperator matrix in the package acts on ``vec`` (column stacking), for
  which ``vec(A @ X @ B) == kron(B.T, A) @ vec(X)``.
* Most dense functions accept stacks of matrices with shape ``(..., d, d)``,
  so a whole trajectory of states can be processed in one call.
* The module-level constants are read-only arrays: copy them
  (``sx.copy()``) before modifying them in place.
"""

from __future__ import annotations

from functools import reduce
from math import isqrt
from typing import Callable, Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
from numpy.typing import ArrayLike, NDArray

__all__ = [
    # numerical tolerance
    "ATOL",
    # kets and operators
    "ket0", "ket1", "sx", "sy", "sz", "sp", "sm", "id2", "paulis",
    "wh", "hadamard", "cnot",
    "phi_plus", "phi_minus", "psi_plus", "psi_minus", "w3", "ghz3",
    "basis", "ket", "ghz_state", "w_state", "identity",
    # algebra
    "dag", "comm", "acomm", "kron", "tensor_power", "outer", "ket2dm",
    "as_dm", "direct_sum", "tridiagonal",
    # matrix functions
    "hermitian_function", "psd_sqrt", "matrix_sqrt", "matrix_abs",
    # checks
    "is_hermitian", "is_unitary", "is_psd", "is_density_matrix",
    # vectorization and superoperators
    "vec", "unvec", "res", "unres", "spre", "spost", "sprepost",
    "superop_from_map",
]

#: Absolute threshold below which eigenvalues and probabilities count as zero.
ATOL = 1e-12


# --------------------------------------------------------------------------
# Small helpers used across the package
# --------------------------------------------------------------------------

def _frozen(a: ArrayLike, dtype=None) -> NDArray:
    arr = np.array(a, dtype=dtype)
    arr.setflags(write=False)
    return arr


def _dense(a) -> NDArray:
    """Return ``a`` as a NumPy array, densifying sparse input."""
    return a.toarray() if sps.issparse(a) else np.asarray(a)


def _square_dim(n: int, what: str = "length") -> int:
    """Integer ``d`` with ``d**2 == n``; raise ``ValueError`` otherwise."""
    d = isqrt(n)
    if d * d != n:
        raise ValueError(f"{what} {n} is not a perfect square")
    return d


def _check_square(a, name: str = "matrix") -> int:
    shape = a.shape
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(f"{name} must be square, got shape {shape}")
    return shape[-1]


def _operator_list(ops) -> list:
    """Normalize one operator, a sequence of operators or a 3-D stack to a list.

    ``None`` and empty sequences give ``[]`` (e.g. no jump operators).
    """
    if ops is None:
        return []
    if sps.issparse(ops):
        return [ops]
    if isinstance(ops, np.ndarray):
        if ops.ndim == 2:
            return [ops]
        if ops.ndim == 3:
            return list(ops)
        raise ValueError(f"expected a matrix or a stack of matrices, got shape {ops.shape}")
    ops = list(ops)
    if not ops:
        return []
    if not sps.issparse(ops[0]) and np.ndim(ops[0]) < 2:
        # a single matrix given as a nested list
        return [np.asarray(ops)]
    return [o if sps.issparse(o) else np.asarray(o) for o in ops]


def _operator_stack(ops) -> NDArray:
    """Dense ``(k, m, n)`` stack from one operator or a collection of them."""
    lst = _operator_list(ops)
    if not lst:
        raise ValueError("at least one operator is required")
    return np.stack([_dense(o) for o in lst])


# --------------------------------------------------------------------------
# Kets, gates and standard states
# --------------------------------------------------------------------------

_SQRT2 = np.sqrt(2.0)

ket0 = _frozen([1.0, 0.0])                         #: ``|0>`` (``m`` in Mathematica)
ket1 = _frozen([0.0, 1.0])                         #: ``|1>`` (``p`` in Mathematica)
sx = _frozen([[0.0, 1.0], [1.0, 0.0]])             #: Pauli X
sy = _frozen([[0.0, -1.0j], [1.0j, 0.0]])          #: Pauli Y
sz = _frozen([[1.0, 0.0], [0.0, -1.0]])            #: Pauli Z
sm = _frozen([[0.0, 1.0], [0.0, 0.0]])             #: lowering ``|0><1| = (sx + i sy)/2``
sp = _frozen([[0.0, 0.0], [1.0, 0.0]])             #: raising ``|1><0| = (sx - i sy)/2``
id2 = _frozen(np.eye(2))                           #: 2x2 identity (``id``)
paulis = _frozen(np.stack([sx, sy, sz]))           #: stack ``(sx, sy, sz)``, shape (3, 2, 2)
wh = _frozen([[1.0, 1.0], [1.0, -1.0]])            #: unnormalized Walsh-Hadamard
hadamard = _frozen(np.asarray(wh) / _SQRT2)        #: Hadamard gate ``wh/sqrt(2)``
cnot = _frozen([[1.0, 0, 0, 0], [0, 1, 0, 0],      #: CNOT, control = qubit 0
                [0, 0, 0, 1], [0, 0, 1, 0]])

phi_plus = _frozen([1.0, 0, 0, 1] / _SQRT2)        #: ``(|00> + |11>)/sqrt(2)``
phi_minus = _frozen([1.0, 0, 0, -1] / _SQRT2)      #: ``(|00> - |11>)/sqrt(2)``
psi_plus = _frozen([0, 1.0, 1, 0] / _SQRT2)        #: ``(|01> + |10>)/sqrt(2)``
psi_minus = _frozen([0, 1.0, -1, 0] / _SQRT2)      #: ``(|01> - |10>)/sqrt(2)``


def basis(dims: int | Sequence[int], index: int | Sequence[int]) -> NDArray:
    """Computational basis ket.

    ``basis(d, k)`` is ``|k>`` in dimension ``d``; ``basis((2, 3), (1, 2))`` is
    ``|1> (x) |2>`` on a qubit-qutrit system.
    """
    dims_t = tuple(np.atleast_1d(dims).tolist())
    flat = int(np.ravel_multi_index(tuple(np.atleast_1d(index).tolist()), dims_t))
    out = np.zeros(int(np.prod(dims_t)))
    out[flat] = 1.0
    return out


def ket(label: str) -> NDArray:
    """Multi-qubit computational basis ket from a bit string, e.g. ``ket("010")``."""
    if not label or set(label) - {"0", "1"}:
        raise ValueError(f"label must be a non-empty string of 0s and 1s, got {label!r}")
    out = np.zeros(2 ** len(label))
    out[int(label, 2)] = 1.0
    return out


def ghz_state(n: int) -> NDArray:
    """``n``-qubit GHZ state ``(|0...0> + |1...1>)/sqrt(2)``."""
    out = np.zeros(2 ** n)
    out[[0, -1]] = 1 / _SQRT2
    return out


def w_state(n: int) -> NDArray:
    """``n``-qubit W state, the uniform superposition of the single-excitation kets."""
    out = np.zeros(2 ** n)
    out[1 << np.arange(n)] = 1 / np.sqrt(n)
    return out


w3 = _frozen(w_state(3))          #: ``(|001> + |010> + |100>)/sqrt(3)``
ghz3 = _frozen(ghz_state(3))      #: ``(|000> + |111>)/sqrt(2)``


def identity(n: int, sparse: bool = False):
    """``n x n`` identity, dense or as a CSR sparse array."""
    return sps.eye_array(n, format="csr") if sparse else np.eye(n)


# --------------------------------------------------------------------------
# Basic algebra
# --------------------------------------------------------------------------

def dag(a):
    """Conjugate transpose; works on sparse arrays and on stacks ``(..., m, n)``."""
    if sps.issparse(a):
        return a.conj().T
    a = np.asarray(a)
    return a.conj() if a.ndim < 2 else np.conj(np.swapaxes(a, -1, -2))


def comm(a, b):
    """Commutator ``[A, B] = A B - B A``."""
    return a @ b - b @ a


def acomm(a, b):
    """Anticommutator ``{A, B} = A B + B A``."""
    return a @ b + b @ a


def kron(*ops):
    """Kronecker (tensor) product of any number of kets or matrices.

    Kets give a ket and matrices give a matrix; if any factor is sparse the
    result is a CSR sparse array.  Pass a list as ``kron(*ops)``.
    """
    if not ops:
        raise ValueError("kron needs at least one factor")
    if any(sps.issparse(o) for o in ops):
        out = reduce(lambda a, b: sps.kron(a, b, format="csr"), ops)
        return sps.csr_array(out)
    return reduce(np.kron, (np.asarray(o) for o in ops))


def tensor_power(op, n: int):
    """``op (x) op (x) ... (x) op`` with ``n`` factors."""
    return kron(*([op] * n))


def outer(u: ArrayLike, v: ArrayLike) -> NDArray:
    """Outer product ``u_i v_j`` *without* conjugation (``CircleDot``).

    ``|psi><phi|`` is ``outer(psi, phi.conj())``.
    """
    return np.outer(u, v)


def ket2dm(psi: ArrayLike) -> NDArray:
    """Projector ``|psi><psi|`` (not normalized); a 2-D input is a stack of kets."""
    psi = np.asarray(psi)
    return psi[..., :, None] * psi.conj()[..., None, :]


def as_dm(state) -> NDArray:
    """Density matrix of ``state``: kets (1-D or column) are turned into projectors."""
    a = _dense(state)
    if a.ndim == 1:
        return ket2dm(a)
    if a.ndim == 2 and a.shape[1] == 1 and a.shape[0] > 1:
        return ket2dm(a[:, 0])
    _check_square(a, "density matrix")
    return a


def direct_sum(*ops):
    """Block-diagonal direct sum ``A (+) B (+) ...``; empty arrays are neutral."""
    ops = [o if sps.issparse(o) else np.atleast_2d(o) for o in ops]
    ops = [o for o in ops if o.shape[0] * o.shape[1] > 0]
    if not ops:
        return np.zeros((0, 0))
    if any(sps.issparse(o) for o in ops):
        return sps.csr_array(sps.block_diag(ops, format="csr"))
    return sla.block_diag(*ops)


def _bands(x, count: int, k: int, name: str) -> NDArray:
    """Turn a tridiagonal band specification into a ``(count, k, k)`` block stack."""
    x = np.asarray(x)
    if x.ndim == 0:
        return np.broadcast_to(x * np.eye(k), (count, k, k))
    if x.ndim == 1:
        if k != 1:
            raise ValueError(f"{name}: a list of scalars cannot be mixed with {k}x{k} blocks")
        if len(x) < count:
            raise ValueError(f"{name} needs at least {count} elements, got {len(x)}")
        return x[:count, None, None]
    if x.ndim == 2:
        return np.broadcast_to(x, (count, k, k))
    if x.ndim == 3:
        if len(x) < count:
            raise ValueError(f"{name} needs at least {count} blocks, got {len(x)}")
        return x[:count]
    raise ValueError(f"{name}: unsupported shape {x.shape}")


def tridiagonal(n: int, up, diag, down, *, periodic: bool = False, sparse: bool = True):
    """(Block-)tridiagonal ``n x n`` matrix (``Trid``, ``Tridopen``, ``Tridclosed``).

    Each of ``up`` (upper diagonal), ``diag`` and ``down`` (lower diagonal) can be

    * a scalar, repeated along the band (times the identity for blocks);
    * a 1-D sequence of position-dependent values;
    * a ``k x k`` matrix, giving a block-tridiagonal ``(k n) x (k n)`` matrix;
    * a ``(m, k, k)`` stack of position-dependent blocks.

    Only the first ``n - 1`` off-diagonal elements are used, unless
    ``periodic=True``: then the ``n``-th element of ``up`` is placed in the
    bottom-left corner (block ``(n-1, 0)``) and that of ``down`` in the
    top-right corner (block ``(0, n-1)``), closing the chain into a ring.
    Returns a CSR sparse array, or a dense array with ``sparse=False``.
    """
    if n < 1:
        raise ValueError("n must be positive")
    shapes = [np.shape(x) for x in (up, diag, down)]
    ks = {s[-1] for s in shapes if len(s) >= 2}
    if len(ks) > 1:
        raise ValueError(f"inconsistent block sizes {sorted(ks)}")
    k = ks.pop() if ks else 1
    m = n if periodic else n - 1
    site = np.arange(m)
    bands = [
        (_bands(diag, n, k, "diag"), np.arange(n), np.arange(n)),
        (_bands(up, m, k, "up"), site, (site + 1) % n),
        (_bands(down, m, k, "down"), (site + 1) % n, site),
    ]
    ii, jj = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    rows, cols, data = [], [], []
    for blocks, r, c in bands:
        rows.append((r[:, None, None] * k + ii).ravel())
        cols.append((c[:, None, None] * k + jj).ravel())
        data.append(np.asarray(blocks).ravel())
    data = np.concatenate(data)
    out = sps.coo_array((data, (np.concatenate(rows), np.concatenate(cols))),
                        shape=(k * n, k * n)).tocsr()   # duplicates are summed
    out.eliminate_zeros()
    return out if sparse else out.toarray()


# --------------------------------------------------------------------------
# Matrix functions
# --------------------------------------------------------------------------

def hermitian_function(a: ArrayLike, f: Callable[[NDArray], NDArray]) -> NDArray:
    """``f(A) = V f(Λ) V†`` for a Hermitian matrix (or stack) ``A = V Λ V†``."""
    w, v = np.linalg.eigh(_dense(a))
    return (v * f(w)[..., None, :]) @ dag(v)


def psd_sqrt(a: ArrayLike) -> NDArray:
    """Square root of a positive semidefinite matrix; tiny negative eigenvalues are clipped."""
    return hermitian_function(a, lambda w: np.sqrt(np.clip(w, 0.0, None)))


def matrix_sqrt(a: ArrayLike) -> NDArray:
    """Principal square root (``MatrixSqrt``).

    Hermitian input goes through an eigendecomposition (exact, fast and
    Hermitian-preserving); other matrices through ``scipy.linalg.sqrtm``.
    """
    a = _dense(a)
    if is_hermitian(a):
        w, v = np.linalg.eigh(a)
        w = np.where(np.abs(w) < ATOL, 0.0, w)
        s = np.sqrt(w.astype(complex)) if (w < 0).any() else np.sqrt(w)
        return (v * s) @ dag(v)
    return sla.sqrtm(a)


def matrix_abs(a: ArrayLike) -> NDArray:
    """``|A| = sqrt(A A†)`` (``MatrixAbs``), computed from the SVD ``A = U Σ V†`` as ``U Σ U†``."""
    u, s, _ = np.linalg.svd(_dense(a))
    return (u * s[..., None, :]) @ dag(u)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def is_hermitian(a, atol: float = 1e-10) -> bool:
    a = _dense(a)
    return a.ndim >= 2 and a.shape[-1] == a.shape[-2] and np.allclose(a, dag(a), rtol=0, atol=atol)


def is_unitary(u, atol: float = 1e-10) -> bool:
    u = _dense(u)
    n = _check_square(u)
    return np.allclose(dag(u) @ u, np.eye(n), rtol=0, atol=atol)


def is_psd(a, atol: float = 1e-10) -> bool:
    """Hermitian with no eigenvalue below ``-atol``."""
    return is_hermitian(a, atol) and bool(np.linalg.eigvalsh(_dense(a)).min() >= -atol)


def is_density_matrix(rho, atol: float = 1e-10) -> bool:
    """Positive semidefinite with unit trace."""
    rho = _dense(rho)
    return is_psd(rho, atol) and abs(np.trace(rho) - 1) < atol


# --------------------------------------------------------------------------
# Vectorization and superoperators (column stacking)
# --------------------------------------------------------------------------

def vec(a) -> NDArray:
    """Column-stacking vectorization; a stack ``(..., m, n)`` gives ``(..., m n)``."""
    a = _dense(a)
    return np.swapaxes(a, -1, -2).reshape(a.shape[:-2] + (-1,))


def unvec(v: ArrayLike, rows: int | None = None) -> NDArray:
    """Inverse of :func:`vec`: fill columns of length ``rows`` (default: square matrix)."""
    v = np.asarray(v)
    n = v.shape[-1]
    if rows is None:
        rows = _square_dim(n, "vector length")
    if rows <= 0 or n % rows:
        raise ValueError(f"a vector of length {n} cannot be split into columns of length {rows}")
    return np.swapaxes(v.reshape(v.shape[:-1] + (n // rows, rows)), -1, -2)


def res(a) -> NDArray:
    """Row-stacking vectorization (``Flatten``); a stack ``(..., m, n)`` gives ``(..., m n)``."""
    a = _dense(a)
    return a.reshape(a.shape[:-2] + (-1,))


def unres(v: ArrayLike, cols: int | None = None) -> NDArray:
    """Inverse of :func:`res`: fill rows of length ``cols`` (default: square matrix)."""
    v = np.asarray(v)
    n = v.shape[-1]
    if cols is None:
        cols = _square_dim(n, "vector length")
    if cols <= 0 or n % cols:
        raise ValueError(f"a vector of length {n} cannot be split into rows of length {cols}")
    return v.reshape(v.shape[:-1] + (n // cols, cols))


def _eye_like(a):
    n = a.shape[0]
    return identity(n, sparse=sps.issparse(a))


def spre(a):
    """Superoperator of ``X -> A X``: ``kron(I, A)``."""
    return kron(_eye_like(a), a)


def spost(b):
    """Superoperator of ``X -> X B``: ``kron(B.T, I)``."""
    return kron(b.T, _eye_like(b))


def sprepost(a, b):
    """Superoperator of ``X -> A X B``: ``kron(B.T, A)``."""
    return kron(b.T, a)


def superop_from_map(fn: Callable[[NDArray], ArrayLike], d: int, d_out: int | None = None) -> NDArray:
    """Matrix of a linear map on ``d x d`` matrices acting on column-stacked vectors.

    Column ``i + d j`` is ``vec(fn(|i><j|))``, so that
    ``vec(fn(X)) == superop_from_map(fn, d) @ vec(X)``.
    """
    cols = [vec(_dense(fn(unvec(e)))) for e in np.eye(d * d)]
    out = np.column_stack(cols)
    if d_out is not None and out.shape[0] != d_out * d_out:
        raise ValueError(f"map output is not {d_out}x{d_out}")
    return out
