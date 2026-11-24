"""Quantum and geometric discord of two-qubit states.

Every two-qubit state is written in the Pauli (Fano) form

    rho = (t I + a.sigma x I + I x b.sigma + sum_ij T_ij sigma_i x sigma_j) / 4,

with ``t = Tr rho``, local Bloch vectors ``a``, ``b`` and correlation matrix
``T``.  A projective measurement of the first qubit along the unit vector
``n`` gives outcome ``+-`` with probability ``p = (t +- a.n)/2``, leaving the
other qubit with Bloch vector ``(b +- T^T n)/(2 p)``.  The conditional
entropies used by the discord are therefore closed-form, vectorized functions
of the measurement direction: the minimization over the Bloch sphere is a
dense NumPy grid followed by a Nelder-Mead polish, instead of thousands of
eigenvalue problems.

Measured subsystems are numbered 0 (A) and 1 (B).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .core import ATOL, as_dm, paulis
from .entropy import binary_entropy, von_neumann
from .subsystems import partial_trace

__all__ = ["two_qubit_correlations", "conditional_entropy_after_measurement",
           "quantum_discord", "geometric_discord"]


def two_qubit_correlations(rho):
    """``(a, b, T)``: ``a_i = Tr rho (s_i x I)``, ``b_j = Tr rho (I x s_j)``, ``T_ij = Tr rho (s_i x s_j)``."""
    rho = as_dm(rho)
    if rho.shape != (4, 4):
        raise ValueError("a 4x4 two-qubit state is required")
    r4 = rho.reshape(2, 2, 2, 2)          # r4[a, c, b, d] = rho[(a c), (b d)]
    a = np.einsum("acbc,iba->i", r4, paulis).real
    b = np.einsum("acad,jdc->j", r4, paulis).real
    t = np.einsum("acbd,iba,jdc->ij", r4, paulis, paulis).real
    return a, b, t


def _direction(theta, phi):
    st = np.sin(theta)
    return np.stack(np.broadcast_arrays(st * np.cos(phi), st * np.sin(phi), np.cos(theta)), -1)


def _avg_conditional_entropy(tr, a, b, t, n, measured, base, tol=ATOL):
    """``sum_k p_k S(rho_other|k)`` for the measurement of ``measured`` along ``n`` (..., 3)."""
    if measured == 1:
        a, b, t = b, a, t.T
    an = n @ a
    tn = n @ t                                   # (T^T n), shape (..., 3)
    total = 0.0
    for s in (1.0, -1.0):
        p = (tr + s * an) / 2
        ok = p > tol
        r = np.linalg.norm(b + s * tn, axis=-1) / (2 * np.where(ok, p, 1.0))
        h = binary_entropy((1 + np.clip(r, 0.0, 1.0)) / 2, base)
        total = total + np.where(ok, p * h, 0.0)
    return total


def _measured_index(measured) -> int:
    if measured not in (0, 1, -1, -2):
        raise ValueError("measured must be 0 (qubit A) or 1 (qubit B)")
    return measured % 2


def conditional_entropy_after_measurement(rho, theta, phi, measured: int = 0, base: float = 2):
    """Average entropy of the unmeasured qubit after measuring ``measured`` along ``(theta, phi)``.

    ``theta`` and ``phi`` broadcast, so a whole grid is evaluated at once.
    ``measured=0`` is Mathematica's ``ClAB[rho][theta, phi]``, ``measured=1`` is ``ClBA``.
    """
    measured = _measured_index(measured)
    a, b, t = two_qubit_correlations(rho)
    tr = np.trace(as_dm(rho)).real
    out = _avg_conditional_entropy(tr, a, b, t, _direction(theta, phi), measured, base)
    return out.item() if np.ndim(out) == 0 else out


def quantum_discord(rho, measured: int = 0, base: float = 2, *, grid: tuple[int, int] = (21, 41),
                    full_output: bool = False):
    """Quantum discord ``S(M) - S(AB) + min_n sum_k p_k S(rho_other|k)`` for projective measurements on ``measured``.

    ``measured=0`` is Mathematica's ``DiscZAB`` (measurement on A) and
    ``measured=1`` is ``DiscZBA``.  The minimum is found on a ``grid`` of
    ``(theta, phi)`` points and refined with Nelder-Mead from the three best
    grid points.  With ``full_output=True`` the optimal ``(theta, phi)`` are
    also returned.
    """
    measured = _measured_index(measured)
    rho = as_dm(rho)
    a, b, t = two_qubit_correlations(rho)
    tr = np.trace(rho).real

    def f(theta, phi):
        return _avg_conditional_entropy(tr, a, b, t, _direction(theta, phi), measured, base)

    th, ph = np.meshgrid(np.linspace(0, np.pi, grid[0]), np.linspace(0, 2 * np.pi, grid[1]),
                         indexing="ij")
    values = f(th, ph).ravel()
    best_val = values.min()
    best = (th.ravel()[values.argmin()], ph.ravel()[values.argmin()])
    for k in np.argsort(values)[:3]:
        res = minimize(lambda x: f(x[0], x[1]), (th.ravel()[k], ph.ravel()[k]), method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 4000})
        if res.fun < best_val:
            best_val, best = float(res.fun), (res.x[0] % (2 * np.pi), res.x[1] % (2 * np.pi))

    s_meas = von_neumann(partial_trace(rho, (2, 2), keep=measured), base)
    disc = float(s_meas - von_neumann(rho, base) + best_val)
    return (disc, *best) if full_output else disc


def geometric_discord(rho, measured: int = 0) -> float:
    """Geometric discord for measurements on ``measured``, normalized to 1 for Bell states.

    ``(|x|^2 + ||T||^2 - k_max)/2`` with ``x`` the Bloch vector of the measured
    qubit and ``k_max`` the largest eigenvalue of ``x x^T + T T^T`` (``measured=0``)
    or ``x x^T + T^T T`` (``measured=1``).  Note the mapping to Mathematica:
    ``GeomDiscordBA`` is ``measured=0`` and ``GeomDiscordAB`` is ``measured=1``.
    """
    measured = _measured_index(measured)
    a, b, t = two_qubit_correlations(rho)
    x, m = (a, t) if measured == 0 else (b, t.T)
    k_max = np.linalg.eigvalsh(np.outer(x, x) + m @ m.T).max()
    return float(0.5 * (x @ x + np.sum(t * t) - k_max))
