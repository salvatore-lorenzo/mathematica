import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sps

from myqubits import lindblad as lb
from myqubits.core import ket2dm, sm, sx, sz, vec, unvec
from myqubits.qubits import QubitRegister, tight_binding
from myqubits.random_states import ginibre, random_state

RNG = np.random.default_rng(4)


def _rand_problem(d=3, k=2):
    g = ginibre(d, rng=RNG)
    return g + g.conj().T, [ginibre(d, rng=RNG) for _ in range(k)], random_state(d, rng=RNG)


def test_generators_match_superoperators():
    h, ops, rho = _rand_problem()
    np.testing.assert_allclose(vec(lb.lindbladian(h, ops)(rho)), lb.liouvillian(h, ops) @ vec(rho),
                               atol=1e-12)
    a, b = ops
    np.testing.assert_allclose(vec(lb.cascade_dissipator(a, b, rho)),
                               lb.cascade_superop(a, b) @ vec(rho), atol=1e-12)
    np.testing.assert_allclose(vec(lb.dissipator(a, rho)), lb.dissipator_superop(a) @ vec(rho),
                               atol=1e-12)
    # the adjoint dissipator is the Hilbert-Schmidt adjoint of the dissipator
    x = ginibre(3, rng=RNG)
    lhs = np.trace(x.conj().T @ lb.dissipator(a, rho))
    rhs = np.trace(lb.adjoint_dissipator(a, x).conj().T @ rho)
    assert lhs == pytest.approx(rhs)
    # stacks of states
    stack = random_state(3, size=4, rng=RNG)
    np.testing.assert_allclose(lb.lindbladian(h, ops)(stack)[1], lb.lindbladian(h, ops)(stack[1]))


def test_sparse_liouvillian():
    reg = QubitRegister(3)
    h = tight_binding(1.0, 3)
    ops = [0.3 * reg.sm[i] for i in range(3)]
    lsp = lb.liouvillian(h, ops)
    assert sps.issparse(lsp)
    np.testing.assert_allclose(lsp.toarray(), lb.liouvillian(h.toarray(), [o.toarray() for o in ops]),
                               atol=1e-12)


def test_amplitude_damping_analytic():
    w, g = 1.3, 0.4
    rho0 = np.array([[0.3, 0.2 - 0.1j], [0.2 + 0.1j, 0.7]])
    times = np.linspace(0, 5, 51)
    for method in ("expm", "ode"):
        sol = lb.mesolve(w / 2 * sz, np.sqrt(g) * sm, rho0, times, method=method)
        np.testing.assert_allclose(sol.states[:, 1, 1], 0.7 * np.exp(-g * times), atol=1e-8)
        np.testing.assert_allclose(sol.states[:, 0, 1], rho0[0, 1] * np.exp(-1j * w * times - g * times / 2),
                                   atol=1e-8)
        np.testing.assert_allclose(sol(2.345)[1, 1], 0.7 * np.exp(-g * 2.345), atol=1e-8)
    np.testing.assert_allclose(sol.expect(sz), 1 - 2 * 0.7 * np.exp(-g * times), atol=1e-8)


def test_propagation_paths_agree():
    h, ops, rho = _rand_problem(4)
    lsup = lb.liouvillian(h, ops)
    uniform = np.linspace(0, 2, 9)
    irregular = np.array([0, 0.1, 0.15, 0.7, 2.0])
    ref = lb.propagate(lsup, rho, uniform)
    for t in irregular:
        np.testing.assert_allclose(ref(t), unvec(sla.expm(lsup * t) @ vec(rho)),
                                   atol=1e-10)
    np.testing.assert_allclose(lb.propagate(sps.csr_array(lsup), rho, uniform).states, ref.states, atol=1e-10)
    np.testing.assert_allclose(lb.propagate(lsup, rho, irregular).states[-1], ref.states[-1], atol=1e-10)
    ode = lb.mesolve(h, ops, rho, uniform, method="ode")
    np.testing.assert_allclose(ode.states, ref.states, atol=1e-7)
    np.testing.assert_allclose(ode(np.array([0.3, 1.1])), ref(np.array([0.3, 1.1])), atol=1e-7)


def test_time_dependent_generator():
    rho0 = ket2dm(np.array([1, 1]) / np.sqrt(2))
    sol = lb.solve_generator(lambda t, r: -1j * np.cos(t) * (sz @ r - r @ sz), rho0, [0, 3],
                             time_dependent=True)
    np.testing.assert_allclose(sol(1.7)[0, 1], 0.5 * np.exp(-2j * np.sin(1.7)), atol=1e-8)


def test_steady_state():
    g, om = 0.5, 0.8
    lsup = lb.liouvillian(om * sx, np.sqrt(g) * sm)
    ss = lb.steady_state(lsup)
    np.testing.assert_allclose(lb.liouvillian(om * sx, np.sqrt(g) * sm) @ vec(ss), 0, atol=1e-12)
    late = lb.propagate(lsup, np.eye(2) / 2, [0, 200]).states[-1]
    np.testing.assert_allclose(ss, late, atol=1e-10)
    np.testing.assert_allclose(lb.steady_state(sps.csr_array(lsup)), ss, atol=1e-12)
    np.testing.assert_allclose(lb.steady_state(lb.liouvillian(None, sm)), ket2dm([1, 0]), atol=1e-12)


def test_von_neumann_equation_is_unitary():
    h, _, rho = _rand_problem()
    sol = lb.mesolve(h, [], rho, [0, 1.0])
    u = sla.expm(-1j * h)
    np.testing.assert_allclose(sol.states[-1], u @ rho @ u.conj().T, atol=1e-10)
