import numpy as np
import pytest
import scipy.sparse as sps

from myqubits import core
from myqubits.core import (cnot, dag, direct_sum, ket, ket0, ket1, ket2dm, kron, matrix_abs,
                           matrix_sqrt, res, sm, sp, sprepost, spost, spre, superop_from_map,
                           sx, sy, sz, tridiagonal, unres, unvec, vec)
from myqubits.random_states import ginibre, random_state

RNG = np.random.default_rng(0)


def test_ladder_conventions():
    np.testing.assert_allclose(sm @ ket1, ket0)
    np.testing.assert_allclose(sp @ ket0, ket1)
    np.testing.assert_allclose(sm, (sx + 1j * sy) / 2)
    np.testing.assert_allclose(sp, (sx - 1j * sy) / 2)
    np.testing.assert_allclose(cnot @ ket("10"), ket("11"))


def test_constants_are_read_only():
    with pytest.raises(ValueError):
        sx[0, 0] = 5


def test_states():
    np.testing.assert_allclose(core.w3, (ket("001") + ket("010") + ket("100")) / np.sqrt(3))
    np.testing.assert_allclose(core.ghz3, (ket("000") + ket("111")) / np.sqrt(2))
    np.testing.assert_allclose(core.psi_minus, (kron(ket0, ket1) - kron(ket1, ket0)) / np.sqrt(2))
    np.testing.assert_allclose(core.basis((2, 3), (1, 2)), kron(ket1, [0, 0, 1]))


def test_kron_dense_and_sparse():
    a, b, c = ginibre(2, rng=RNG), ginibre(3, rng=RNG), ginibre(2, rng=RNG)
    dense = kron(a, b, c)
    np.testing.assert_allclose(dense, np.kron(np.kron(a, b), c))
    sparse = kron(sps.csr_array(a), b, c)
    assert sps.issparse(sparse)
    np.testing.assert_allclose(sparse.toarray(), dense)
    assert kron(ket0, ket1).shape == (4,)
    np.testing.assert_allclose(core.tensor_power(sx, 3), kron(sx, sx, sx))


def test_direct_sum():
    a, b = np.ones((2, 2)), 2 * np.ones((1, 1))
    out = direct_sum(a, np.zeros((0, 0)), b)
    np.testing.assert_allclose(out, [[1, 1, 0], [1, 1, 0], [0, 0, 2]])
    np.testing.assert_allclose(direct_sum(sps.csr_array(a), b).toarray(), out)


def test_tridiagonal_scalar_vector_block():
    t = tridiagonal(4, 1, 2, 3).toarray()
    np.testing.assert_allclose(t, np.diag([1] * 3, 1) + np.diag([2] * 4) + np.diag([3] * 3, -1))
    up, dg, dw = np.arange(1, 5.0), np.arange(5, 9.0), np.arange(9, 13.0)
    closed = tridiagonal(4, up, dg, dw, periodic=True, sparse=False)
    expected = np.diag(up[:3], 1) + np.diag(dg) + np.diag(dw[:3], -1)
    expected[3, 0] += up[3]
    expected[0, 3] += dw[3]
    np.testing.assert_allclose(closed, expected)
    blocks = tridiagonal(3, sx, sz, sy, sparse=False)
    ref = np.kron(np.eye(3, k=1), sx) + np.kron(np.eye(3), sz) + np.kron(np.eye(3, k=-1), sy)
    np.testing.assert_allclose(blocks, ref)
    # a scalar is broadcast to a multiple of the identity block
    np.testing.assert_allclose(tridiagonal(3, sx, 0.5, sx, sparse=False),
                               np.kron(np.eye(3, k=1) + np.eye(3, k=-1), sx) + 0.5 * np.eye(6))


def test_vectorization():
    a, x, b = ginibre(3, 2, rng=RNG), ginibre(2, 4, rng=RNG), ginibre(4, 3, rng=RNG)
    np.testing.assert_allclose(vec(a @ x @ b), np.kron(b.T, a) @ vec(x))
    np.testing.assert_allclose(unvec(vec(x), 2), x)
    np.testing.assert_allclose(unres(res(x), 4), x)
    np.testing.assert_allclose(vec(x), x.T.ravel())
    stack = ginibre(3, size=5, rng=RNG)
    np.testing.assert_allclose(unvec(vec(stack)), stack)


def test_superoperators():
    a, b, x = (ginibre(3, rng=RNG) for _ in range(3))
    np.testing.assert_allclose(unvec(sprepost(a, b) @ vec(x)), a @ x @ b)
    np.testing.assert_allclose(unvec(spre(a) @ vec(x)), a @ x)
    np.testing.assert_allclose(unvec(spost(b) @ vec(x)), x @ b)
    lsup = superop_from_map(lambda m: a @ m @ b + m.T, 3)
    np.testing.assert_allclose(unvec(lsup @ vec(x)), a @ x @ b + x.T)


def test_matrix_functions():
    rho = random_state(4, rng=RNG)
    s = matrix_sqrt(rho)
    np.testing.assert_allclose(s @ s, rho, atol=1e-12)
    h = rho - 0.3 * np.eye(4)                       # Hermitian with negative eigenvalues
    s = matrix_sqrt(h)
    np.testing.assert_allclose(s @ s, h, atol=1e-12)
    g = ginibre(4, rng=RNG)
    s = matrix_sqrt(g)
    np.testing.assert_allclose(s @ s, g, atol=1e-10)
    m = matrix_abs(g)
    np.testing.assert_allclose(m @ m, g @ dag(g), atol=1e-10)


def test_checks_and_as_dm():
    rho = random_state(3, rng=RNG)
    assert core.is_density_matrix(rho)
    assert not core.is_density_matrix(2 * rho)
    assert core.is_unitary(core.hadamard)
    np.testing.assert_allclose(core.as_dm(core.phi_plus), ket2dm(core.phi_plus))
    np.testing.assert_allclose(core.as_dm(core.phi_plus[:, None]), ket2dm(core.phi_plus))
