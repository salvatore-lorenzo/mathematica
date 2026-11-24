"""Symbolic computations with SymPy (optional dependency).

The rest of ``myqubits`` is numerical (NumPy/SciPy).  This module covers the
symbolic side of the Mathematica package: symbolic matrices, conjugation that
treats symbols as real, simplification, partial traces of symbolic operators
and closed-form solutions of Lindblad equations.

SymPy matrices are used throughout; kets are column matrices.

>>> import sympy as S
>>> from myqubits import symbolic as sym
>>> g, w = S.symbols("gamma omega", positive=True)
>>> rho0 = sym.sym_matrix("r", 2)
>>> rho_t = sym.mesolve(w / 2 * sym.sz, [S.sqrt(g) * sym.sm], rho0)   # function of sym.tau
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

try:
    import sympy
except ImportError as exc:                                    # pragma: no cover
    raise ImportError("myqubits.symbolic needs sympy: pip install sympy") from exc

from sympy import I, ImmutableMatrix, Matrix, MatrixBase

from . import subsystems as _sub

__all__ = [
    "tau", "ket0", "ket1", "sx", "sy", "sz", "sp", "sm", "id2", "hadamard",
    "sym_matrix", "sym_vector", "to_sympy", "conj_real", "simplify_real",
    "dag", "comm", "acomm", "kron", "ket2dm", "vec", "unvec",
    "partial_trace", "partial_transpose", "concurrence_x",
    "dissipator", "lindbladian", "liouvillian", "superop_from_map",
    "solve_linear", "mesolve",
]

#: Default time symbol (Mathematica's global ``\[Tau]``).
tau = sympy.Symbol("tau", nonnegative=True)

ket0 = ImmutableMatrix([1, 0])
ket1 = ImmutableMatrix([0, 1])
sx = ImmutableMatrix([[0, 1], [1, 0]])
sy = ImmutableMatrix([[0, -I], [I, 0]])
sz = ImmutableMatrix([[1, 0], [0, -1]])
sm = ImmutableMatrix([[0, 1], [0, 0]])            # |0><1|
sp = ImmutableMatrix([[0, 0], [1, 0]])            # |1><0|
id2 = ImmutableMatrix.eye(2)
hadamard = ImmutableMatrix([[1, 1], [1, -1]]) / sympy.sqrt(2)


# --------------------------------------------------------------------------
# Construction and elementary operations
# --------------------------------------------------------------------------

def sym_matrix(name: str, rows: int, cols: int | None = None, real: bool = True) -> Matrix:
    """Matrix of symbols ``name_ij`` (1-based, as ``SymMatrix``); real by default."""
    cols = rows if cols is None else cols
    sep = "_" if max(rows, cols) > 9 else ""
    return Matrix(rows, cols, lambda i, j: sympy.Symbol(f"{name}_{i + 1}{sep}{j + 1}", real=real))


def sym_vector(name: str, n: int, real: bool = True) -> Matrix:
    """Column of symbols ``name_i`` (``SymVector``)."""
    return Matrix(n, 1, lambda i, _: sympy.Symbol(f"{name}_{i + 1}", real=real))


def to_sympy(a, rational: bool = True) -> Matrix:
    """Convert a NumPy array to a SymPy matrix, turning floats into exact numbers if ``rational``."""
    m = Matrix(np.asarray(a).tolist())
    return m.applyfunc(sympy.nsimplify) if rational else m


def conj_real(expr):
    """Complex conjugate obtained by flipping the sign of ``I``, i.e. treating every symbol as real (``Conj``)."""
    return expr.xreplace({I: -I})


def simplify_real(expr):
    """``simplify(expand_complex(expr))`` elementwise (``cf``)."""
    f = lambda e: sympy.simplify(sympy.expand_complex(e))
    return expr.applyfunc(f) if isinstance(expr, MatrixBase) else f(expr)


def dag(m: MatrixBase) -> Matrix:
    return m.H


def comm(a, b):
    return a * b - b * a


def acomm(a, b):
    return a * b + b * a


def kron(*ms) -> Matrix:
    return ms[0] if len(ms) == 1 else sympy.kronecker_product(*ms)


def ket2dm(psi: MatrixBase, real_symbols: bool = False) -> Matrix:
    """``|psi><psi|``; with ``real_symbols=True`` conjugation is :func:`conj_real` (``Todm``)."""
    bra = conj_real(psi).T if real_symbols else psi.H
    return psi * bra


def vec(m: MatrixBase) -> Matrix:
    """Column stacking."""
    return Matrix(m).vec()


def unvec(v: MatrixBase, rows: int | None = None) -> Matrix:
    n = len(v)
    rows = int(sympy.sqrt(n)) if rows is None else rows
    if rows * (n // rows) != n:
        raise ValueError(f"a vector of length {n} cannot be split into columns of length {rows}")
    return Matrix(rows, n // rows, lambda i, j: v[i + rows * j])


# --------------------------------------------------------------------------
# Composite systems (reusing the NumPy algorithms on object arrays)
# --------------------------------------------------------------------------

def _obj(m) -> np.ndarray:
    m = Matrix(m)
    arr = np.array(m.tolist(), dtype=object)
    return arr[:, 0] if m.cols == 1 and m.rows > 1 else arr


def _back(x):
    return Matrix(x) if np.ndim(x) == 2 else sympy.sympify(x)


def partial_trace(m, dims: Sequence[int] | None = None, sys=None, *, keep=None):
    """Symbolic partial trace (same conventions as :func:`myqubits.subsystems.partial_trace`)."""
    return _back(_sub.partial_trace(_obj(m), dims, sys, keep=keep))


def partial_transpose(m, dims: Sequence[int] | None = None, sys=0) -> Matrix:
    return _back(_sub.partial_transpose(_obj(m), dims, sys))


def concurrence_x(rho):
    """Symbolic concurrence of a two-qubit X state (``ConcurrenceX``)."""
    k1 = sympy.Abs(rho[1, 2]) - sympy.sqrt(rho[0, 0] * rho[3, 3])
    k2 = sympy.Abs(rho[0, 3]) - sympy.sqrt(rho[1, 1] * rho[2, 2])
    return 2 * sympy.Max(0, k1, k2)


# --------------------------------------------------------------------------
# Master equations
# --------------------------------------------------------------------------

def dissipator(L, rho):
    ldl = L.H * L
    return L * rho * L.H - (ldl * rho + rho * ldl) / 2


def lindbladian(H, c_ops=()) -> Callable:
    """``rho -> -I [H, rho] + sum_k D[L_k](rho)`` (``ME[H, ops]``)."""
    ops = [c_ops] if isinstance(c_ops, MatrixBase) else list(c_ops)
    return lambda rho: -I * comm(H, rho) + sum((dissipator(L, rho) for L in ops), sympy.zeros(*rho.shape))


def liouvillian(H, c_ops=()) -> Matrix:
    """Liouvillian matrix acting on column-stacked vectors (``METoVec``)."""
    ops = [c_ops] if isinstance(c_ops, MatrixBase) else list(c_ops)
    e = sympy.eye(H.rows)
    out = -I * (kron(e, H) - kron(H.T, e))
    for L in ops:
        ldl = L.H * L
        out += kron(L.conjugate(), L) - kron(ldl.T, e) / 2 - kron(e, ldl) / 2
    return out


def superop_from_map(fn: Callable, d: int) -> Matrix:
    """Matrix of a linear map on ``d x d`` SymPy matrices, acting on column-stacked vectors."""
    cols = []
    for q in range(d * d):
        e = sympy.zeros(d * d, 1)
        e[q] = 1
        cols.append(vec(fn(unvec(e, d))))
    return Matrix.hstack(*cols)


def solve_linear(generator, rho0, t=tau, simplify: bool = True) -> Matrix:
    """Closed-form solution of ``drho/dt = generator(rho)`` with ``rho(0) = rho0`` (``MESolve``).

    ``generator`` is a linear function of a matrix or directly its
    superoperator matrix; the solution is ``unvec(exp(L t) vec(rho0))``.
    """
    rho0 = Matrix(rho0)
    d = rho0.rows
    lsup = generator if isinstance(generator, MatrixBase) else superop_from_map(generator, d)
    rho_t = unvec((lsup * t).exp() * vec(rho0), d)
    return rho_t.applyfunc(sympy.simplify) if simplify else rho_t


def mesolve(H, c_ops, rho0, t=tau, simplify: bool = True) -> Matrix:
    """Closed-form solution of the Lindblad equation (``MESolve[ME[H, ops], rho0]`` / ``METoVecSolve``)."""
    return solve_linear(liouvillian(H, c_ops), rho0, t, simplify)
