import numpy as np
import pytest

from myqubits import bases, random_states as rs
from myqubits.core import kron, paulis, sx, sy, sz, id2, ket2dm
from myqubits.qubits import QubitRegister, bloch_vector, multi_qubit_operator, tight_binding

RNG = np.random.default_rng(5)


def test_register_operators():
    reg = QubitRegister(3)
    np.testing.assert_allclose(reg.sx[1].toarray(), kron(id2, sx, id2))
    np.testing.assert_allclose((reg.sx[2] @ reg.sy[2] - reg.sy[2] @ reg.sx[2]).toarray(),
                               2j * reg.sz[2].toarray())
    np.testing.assert_allclose(reg.product({0: sx, 2: sz}).toarray(), kron(sx, id2, sz))
    np.testing.assert_allclose(multi_qubit_operator({-1: sy}, 2, sparse=False), kron(id2, sy))
    np.testing.assert_allclose((reg.sm[0] @ reg.sp[0]).toarray(), kron([[1, 0], [0, 0]], id2, id2))


@pytest.mark.parametrize("n, periodic", [(1, False), (2, False), (4, False), (4, True), (5, True)])
def test_tight_binding_matches_kron(n, periodic):
    reg = QubitRegister(n, sparse=False)
    bonds = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if periodic and n > 2 else [])
    couplings = RNG.normal(size=len(bonds))
    ref = sum((c * (reg.sx[i] @ reg.sx[j] + reg.sy[i] @ reg.sy[j]) for (i, j), c in zip(bonds, couplings)),
              np.zeros((2 ** n, 2 ** n)))
    np.testing.assert_allclose(tight_binding(couplings, n, periodic).toarray(), ref, atol=1e-12)
    if bonds:
        h = sum(reg.sx[i] @ reg.sx[j] + reg.sy[i] @ reg.sy[j] for i, j in bonds)
        np.testing.assert_allclose(tight_binding(0.7, n, periodic, sparse=False), 0.7 * h, atol=1e-12)


def test_bloch_vector():
    psi = rs.random_ket(2, rng=RNG)
    v = bloch_vector(psi)
    assert np.linalg.norm(v) == pytest.approx(1)
    np.testing.assert_allclose(v, [np.vdot(psi, s @ psi).real for s in paulis])
    stack = rs.random_state(2, size=7, rng=RNG)
    np.testing.assert_allclose(bloch_vector(stack)[3], bloch_vector(stack[3]))
    np.testing.assert_allclose(bases.state_to_bloch(stack[3])[1:], bloch_vector(stack[3]))


def test_random_objects():
    u = rs.random_unitary(5, rng=RNG)
    np.testing.assert_allclose(u @ u.conj().T, np.eye(5), atol=1e-12)
    su = rs.random_special_unitary(4, rng=RNG)
    assert np.linalg.det(su) == pytest.approx(1)
    o = rs.random_orthogonal(4, rng=RNG)
    assert np.isrealobj(o)
    np.testing.assert_allclose(o @ o.T, np.eye(4), atol=1e-12)
    for kind in ("hs", "bures", 7):
        states = rs.random_state(3, kind, size=(2, 3), rng=RNG)
        assert states.shape == (2, 3, 3, 3)
        np.testing.assert_allclose(np.trace(states, axis1=-2, axis2=-1), 1)
        assert np.linalg.eigvalsh(states).min() > -1e-12
    povm = rs.random_povm(3, 5, rng=RNG)
    np.testing.assert_allclose(povm.sum(0), np.eye(3), atol=1e-12)
    assert np.linalg.eigvalsh(povm).min() > -1e-12
    kraus = rs.random_kraus(4, 3, rng=RNG)
    np.testing.assert_allclose(sum(k.conj().T @ k for k in kraus), np.eye(4), atol=1e-12)
    assert rs.random_simplex(6, rng=RNG).sum() == pytest.approx(1)
    kets = rs.random_ket(4, size=10, rng=RNG)
    np.testing.assert_allclose(np.linalg.norm(kets, axis=1), 1)
    np.testing.assert_allclose(rs.random_state(3, rng=42), rs.random_state(3, rng=42))
    with pytest.raises(ValueError):
        rs.random_state(3, 2)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_bases_orthonormal(n):
    b = bases.hermitian_basis(n)
    gram = np.einsum("aij,bij->ab", b.conj(), b)
    np.testing.assert_allclose(gram, np.eye(n * n), atol=1e-12)
    np.testing.assert_allclose(b, np.conj(np.swapaxes(b, 1, 2)))
    if n > 1:
        g = bases.gell_mann(n)
        np.testing.assert_allclose(np.trace(g, axis1=1, axis2=2), 0, atol=1e-12)


def test_bases_specifics():
    np.testing.assert_allclose(bases.gell_mann(2), paulis)
    np.testing.assert_allclose(bases.gell_mann(3)[0], bases.gell_mann_symmetric(0, 1, 3))
    np.testing.assert_allclose(bases.gell_mann(3)[-1], bases.gell_mann_diagonal(2, 3))
    p = bases.pauli_basis(2)
    np.testing.assert_allclose(np.einsum("aij,bij->ab", p.conj(), p), np.eye(16), atol=1e-12)
    np.testing.assert_allclose(p[1 * 4 + 3], kron(sx, sz) / 2)
    np.testing.assert_allclose(bases.matrix_units(2)[1], [[0, 1], [0, 0]])
    rho = rs.random_state(3, rng=RNG)
    np.testing.assert_allclose(bases.from_coefficients(bases.basis_coefficients(rho)), rho, atol=1e-12)
    np.testing.assert_allclose(bases.bloch_to_state(bases.state_to_bloch(rho)), rho, atol=1e-12)
    np.testing.assert_allclose(bases.bloch_to_state([0, 0, 1]), ket2dm([1, 0]), atol=1e-12)
    assert np.isrealobj(bases.state_to_bloch(rho))
