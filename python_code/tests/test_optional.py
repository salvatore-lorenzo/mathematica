"""Tests of the optional modules (skipped when sympy / matplotlib are missing)."""

import numpy as np
import pytest

from myqubits import lindblad, subsystems
from myqubits.core import ket0, ket1, phi_plus, sm, sz
from myqubits.random_states import ginibre, random_state


# ----------------------------------------------------------------- symbolic

def test_symbolic():
    sympy = pytest.importorskip("sympy")
    from myqubits import symbolic as sym

    a = sym.sym_matrix("a", 2)
    assert a.shape == (2, 2) and str(a[0, 1]) == "a_12"
    x, y = sympy.symbols("x y", real=True)
    assert sym.conj_real(x + sympy.I * y) == x - sympy.I * y
    assert sym.simplify_real(sympy.exp(sympy.I * x) * sympy.exp(-sympy.I * x)) == 1

    # partial trace / transpose agree with the numerical ones after substitution
    r = sym.sym_matrix("r", 4, real=False)
    values = ginibre(4, rng=0)
    subs = {r[i, j]: values[i, j] for i in range(4) for j in range(4)}
    for sys in (0, 1):
        num = np.array(sym.partial_trace(r, (2, 2), sys).subs(subs), dtype=complex)
        np.testing.assert_allclose(num, subsystems.partial_trace(values, (2, 2), sys))
    num = np.array(sym.partial_transpose(r, (2, 2), 1).subs(subs), dtype=complex)
    np.testing.assert_allclose(num, subsystems.partial_transpose(values, (2, 2), 1))
    psi = sym.kron(sym.ket0, sym.ket1) + sym.kron(sym.ket1, sym.ket0)
    assert sym.partial_trace(psi / sympy.sqrt(2), (2, 2), 0) == sympy.eye(2) / 2
    assert sym.partial_trace(sympy.eye(4), (2, 2), [0, 1]) == 4

    # Liouvillian agrees with the numerical one
    g, w = sympy.symbols("gamma omega", positive=True)
    h, ls = w / 2 * sym.sz, [sympy.sqrt(g) * sym.sm]
    lnum = np.array(sym.liouvillian(h, ls).subs({g: 0.3, w: 1.1}), dtype=complex)
    np.testing.assert_allclose(lnum, lindblad.liouvillian(1.1 / 2 * sz, np.sqrt(0.3) * sm))
    lsup_from_map = sym.superop_from_map(sym.lindbladian(h, ls), 2)
    assert (lsup_from_map - sym.liouvillian(h, ls)).applyfunc(sympy.simplify) == sympy.zeros(4)

    # closed-form amplitude damping
    rho0 = sym.sym_matrix("p", 2, real=False)
    rho_t = sym.mesolve(h, ls, rho0)
    t = sym.tau
    assert sympy.simplify(rho_t[1, 1] - rho0[1, 1] * sympy.exp(-g * t)) == 0
    assert sympy.simplify(rho_t[0, 1] - rho0[0, 1] * sympy.exp(-g * t / 2 - sympy.I * w * t)) == 0
    assert sympy.simplify(rho_t.trace() - rho0.trace()) == 0


# ------------------------------------------------------------------- bloch

def test_bloch_sphere(tmp_path):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from myqubits.bloch import BlochSphere, bloch_points

    assert bloch_points(ket0).shape == (1, 3)
    np.testing.assert_allclose(bloch_points([ket0, ket1]), [[0, 0, 1], [0, 0, -1]])
    np.testing.assert_allclose(bloch_points(np.array([[1, 0], [0, 0]])), [[0, 0, 1]])
    assert bloch_points(random_state(2, size=5, rng=1)).shape == (5, 3)
    assert bloch_points([0.1, 0.2, 0.3]).shape == (1, 3)
    traj = lindblad.mesolve(sz, np.sqrt(0.2) * sm, (ket0 + ket1) / np.sqrt(2), np.linspace(0, 10, 50))
    b = BlochSphere()
    b.add_points(ket0, color="red").add_vectors([ket1, random_state(2, rng=2)], color="blue")
    b.add_line(traj.states, color="green")
    b.save(tmp_path / "bloch.png")
    assert (tmp_path / "bloch.png").stat().st_size > 1000
    with pytest.raises(ValueError):
        bloch_points(phi_plus)
