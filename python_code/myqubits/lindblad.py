"""Lindblad and cascaded master equations.

Two complementary representations are provided:

* **generators**: Python functions ``rho -> drho/dt`` (Mathematica's
  ``ME[H, ops]``).  :func:`lindbladian` precomputes the effective
  Hamiltonian ``H - i/2 sum L†L``, so each call costs ``1 + k`` matrix
  products, and it works on stacks of density matrices;
* **superoperators**: Liouvillian matrices acting on column-stacked vectors
  (``METoVec`` and friends), sparse when the inputs are sparse.

Solvers:

* :func:`mesolve` - Lindblad equation with constant ``H`` and jump operators,
  solved exactly with matrix exponentials (``method="expm"``) or with an ODE
  integrator (``method="ode"``);
* :func:`solve_generator` - any (possibly time-dependent, even nonlinear)
  generator, integrated with :func:`scipy.integrate.solve_ivp`
  (``MESolvet``);
* :func:`propagate` - exact evolution ``exp(L t) vec(rho0)`` for a given Liouvillian;
* :func:`steady_state` - stationary state of a Liouvillian.

The returned :class:`MESolution` holds the states on the requested time
grid and can be called at any other time, like an ``InterpolatingFunction``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp

from .core import _dense, _operator_list, as_dm, comm, dag, kron, spost, spre, unvec, vec

__all__ = [
    "dissipator", "adjoint_dissipator", "cascade_dissipator", "lindbladian",
    "hamiltonian_superop", "dissipator_superop", "cascade_superop", "liouvillian",
    "MESolution", "propagate", "mesolve", "solve_generator", "steady_state",
]


def _op(a):
    return a if sps.issparse(a) else np.asarray(a)


# --------------------------------------------------------------------------
# Generators acting on matrices
# --------------------------------------------------------------------------

def dissipator(L, rho):
    """``D[L](rho) = L rho L† - {L†L, rho}/2``."""
    L = _op(L)
    ld = dag(L)
    ldl = ld @ L
    return L @ rho @ ld - 0.5 * (ldl @ rho + rho @ ldl)


def adjoint_dissipator(L, x):
    """Heisenberg-picture dissipator ``L† X L - {L†L, X}/2`` (``Dc``)."""
    L = _op(L)
    ld = dag(L)
    ldl = ld @ L
    return ld @ x @ L - 0.5 * (ldl @ x + x @ ldl)


def cascade_dissipator(a, b, rho):
    """Cascaded-system generator ``[A rho, B†] + [B, rho A†] + D[A](rho) + D[B](rho)``.

    A source with jump operator ``A`` drives unidirectionally a target with
    jump operator ``B`` (``DcascAB``).
    """
    a, b = _op(a), _op(b)
    return (comm(a @ rho, dag(b)) + comm(b, rho @ dag(a))
            + dissipator(a, rho) + dissipator(b, rho))


def lindbladian(H, c_ops=()) -> Callable:
    """The generator ``rho -> -i[H, rho] + sum_k D[L_k](rho)`` as a function (``ME[H, ops]``).

    ``c_ops`` is one jump operator, a sequence of them or ``()`` (von Neumann
    equation).  The returned function accepts stacks ``(..., d, d)`` for dense input.
    """
    ops = _operator_list(c_ops)
    H = _op(H)
    heff = H
    for L in ops:
        heff = heff - 0.5j * (dag(L) @ L)
    heff_d = dag(heff)
    pairs = [(L, dag(L)) for L in ops]

    def generator(rho):
        out = -1j * (heff @ rho - rho @ heff_d)
        for L, ld in pairs:
            out = out + L @ rho @ ld
        return out

    return generator


# --------------------------------------------------------------------------
# Superoperators (column stacking)
# --------------------------------------------------------------------------

def hamiltonian_superop(H):
    """``-i (I x H - H^T x I)``: ``vec(-i[H, rho]) == hamiltonian_superop(H) @ vec(rho)`` (``HToVec``)."""
    H = _op(H)
    return -1j * (spre(H) - spost(H))


def dissipator_superop(L):
    """``conj(L) x L - (L†L)^T x I / 2 - I x L†L / 2`` (``DToVec``)."""
    L = _op(L)
    ldl = dag(L) @ L
    return kron(L.conj(), L) - 0.5 * spost(ldl) - 0.5 * spre(ldl)


def cascade_superop(a, b):
    """Superoperator of :func:`cascade_dissipator` (``DcascABToVec``)."""
    a, b = _op(a), _op(b)
    return (kron(b.conj(), a) - spre(dag(b) @ a) + kron(a.conj(), b) - spost(dag(a) @ b)
            + dissipator_superop(a) + dissipator_superop(b))


def liouvillian(H, c_ops=()):
    """Liouvillian matrix of :func:`lindbladian` (``METoVec``).

    Sparse if ``H`` or any jump operator is sparse.  ``H=None`` means no
    Hamiltonian part.
    """
    ops = _operator_list(c_ops)
    if H is None:
        if not ops:
            raise ValueError("give a Hamiltonian or at least one jump operator")
        d = ops[0].shape[0]
        H = sps.csr_array((d, d)) if any(sps.issparse(o) for o in ops) else np.zeros((d, d))
    out = hamiltonian_superop(H)
    for L in ops:
        out = out + dissipator_superop(L)
    return sps.csr_array(out) if sps.issparse(out) else np.asarray(out)


# --------------------------------------------------------------------------
# Solutions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MESolution:
    """States of an evolution on a time grid.

    ``times``: shape ``(nt,)``; ``states``: shape ``(nt, d, d)``.  Calling the
    solution, ``sol(t)``, evaluates the state at arbitrary times (exactly for
    :func:`propagate`/:func:`mesolve` with ``"expm"``, by dense interpolation
    for the ODE solvers).
    """

    times: np.ndarray
    states: np.ndarray
    _evaluate: Callable | None = field(default=None, repr=False)

    def __call__(self, t):
        if self._evaluate is None:
            raise ValueError("this solution has no continuous representation")
        return self._evaluate(t)

    def __len__(self) -> int:
        return len(self.times)

    def expect(self, op) -> np.ndarray:
        """``Tr(op rho(t))`` on the time grid (real for Hermitian ``op``)."""
        op = _dense(op)
        values = np.einsum("ij,tji->t", op, self.states)
        return values.real if np.allclose(op, dag(op)) else values


def _times(times) -> np.ndarray:
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if t.ndim != 1 or (np.diff(t) < 0).any():
        raise ValueError("times must be a non-decreasing 1-D sequence")
    return t


def propagate(liouv, rho0, times) -> MESolution:
    """Exact solution ``vec(rho(t)) = exp(L (t - t0)) vec(rho0)`` with ``t0 = times[0]``.

    Dense Liouvillians use :func:`scipy.linalg.expm`, computed once for
    evenly spaced times; sparse ones use :func:`scipy.sparse.linalg.expm_multiply`,
    which never forms the exponential.
    """
    times = _times(times)
    rho0 = as_dm(rho0)
    d = rho0.shape[0]
    v0 = vec(rho0).astype(complex)
    sparse = sps.issparse(liouv)
    L = sps.csr_array(liouv) if sparse else np.asarray(liouv)
    if L.shape != (d * d, d * d):
        raise ValueError(f"Liouvillian shape {L.shape} does not match a {d}x{d} state")

    def step(dt, v):
        if dt == 0:
            return v
        return spla.expm_multiply(L * dt, v) if sparse else sla.expm(L * dt) @ v

    dts = np.diff(times)
    out = np.empty((len(times), d * d), dtype=complex)
    out[0] = v0
    if len(dts) and np.allclose(dts, dts[0], rtol=1e-10, atol=0):
        if sparse:
            out[:] = spla.expm_multiply(L, v0, start=0.0, stop=times[-1] - times[0],
                                        num=len(times), endpoint=True)
        else:
            u = sla.expm(L * dts[0])
            for k in range(1, len(times)):
                out[k] = u @ out[k - 1]
    else:
        for k, dt in enumerate(dts, start=1):
            out[k] = step(dt, out[k - 1])

    def evaluate(t):
        t_arr = np.asarray(t, dtype=float)
        vs = np.array([step(tk - times[0], v0) for tk in t_arr.ravel()])
        states = unvec(vs)
        return states[0] if t_arr.ndim == 0 else states.reshape(t_arr.shape + (d, d))

    return MESolution(times, unvec(out), evaluate)


def solve_generator(generator: Callable, rho0, times, *, time_dependent: bool = False,
                    method: str = "DOP853", rtol: float = 1e-9, atol: float = 1e-11,
                    **ivp_kwargs) -> MESolution:
    """Integrate ``drho/dt = generator(rho)`` (or ``generator(t, rho)`` with
    ``time_dependent=True``) from ``rho(times[0]) = rho0`` (``MESolvet``).

    ``times`` is the output grid; the returned solution can also be called at
    any time in ``[times[0], times[-1]]`` (dense output), so ``(ti, tf)`` is
    enough to get an interpolating solution.  Extra keyword arguments go to
    :func:`scipy.integrate.solve_ivp`.
    """
    times = _times(times)
    rho0 = as_dm(rho0).astype(complex)
    d = rho0.shape[0]
    if time_dependent:
        def rhs(t, y):
            return np.asarray(_dense(generator(t, y.reshape(d, d)))).ravel()
    else:
        def rhs(t, y):
            return np.asarray(_dense(generator(y.reshape(d, d)))).ravel()

    sol = solve_ivp(rhs, (times[0], times[-1]), rho0.ravel(), method=method, t_eval=times,
                    dense_output=True, rtol=rtol, atol=atol, **ivp_kwargs)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    interp = sol.sol

    def evaluate(t):
        t_arr = np.asarray(t, dtype=float)
        y = interp(t_arr.ravel()).T.reshape(-1, d, d)
        return y[0] if t_arr.ndim == 0 else y.reshape(t_arr.shape + (d, d))

    return MESolution(sol.t, sol.y.T.reshape(-1, d, d), evaluate)


def mesolve(H, c_ops, rho0, times, *, method: str = "expm", **ivp_kwargs) -> MESolution:
    """Solve the Lindblad equation with Hamiltonian ``H`` and jump operators ``c_ops``.

    ``method="expm"`` (default) is exact; ``"ode"`` integrates the matrix-form
    generator with :func:`solve_generator` and is preferable for large Hilbert
    spaces, where the ``d**2 x d**2`` Liouvillian would be too large.
    """
    if method == "expm":
        return propagate(liouvillian(H, c_ops), rho0, times)
    if method == "ode":
        return solve_generator(lindbladian(H, c_ops), rho0, times, **ivp_kwargs)
    raise ValueError(f"unknown method {method!r}")


def steady_state(liouv) -> np.ndarray:
    """Stationary state ``L vec(rho) = 0`` with ``Tr rho = 1`` (assumed unique).

    The trace condition replaces the first equation, giving a square linear
    system solved directly (sparse LU for sparse Liouvillians).
    """
    n = liouv.shape[0]
    d = int(round(np.sqrt(n)))
    trace_row = vec(np.eye(d))
    rhs = np.zeros(n, dtype=complex)
    rhs[0] = 1.0
    if sps.issparse(liouv):
        a = sps.lil_array(liouv, dtype=complex)
        a[0, :] = trace_row
        v = spla.spsolve(sps.csc_array(a), rhs)
    else:
        a = np.array(liouv, dtype=complex)
        a[0, :] = trace_row
        v = np.linalg.solve(a, rhs)
    rho = unvec(v)
    return 0.5 * (rho + dag(rho))
