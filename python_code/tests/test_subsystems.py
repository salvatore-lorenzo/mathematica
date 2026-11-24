from itertools import combinations, product

import numpy as np
import pytest
import scipy.sparse as sps

from myqubits.core import ket2dm, kron, res
from myqubits.random_states import ginibre, random_ket, random_state
from myqubits.subsystems import partial_trace, partial_transpose, permute_subsystems, reshuffle

RNG = np.random.default_rng(1)
DIMS = (2, 3, 2)


def _reference_partial_trace(a, dims, sys):
    """Brute-force partial trace with explicit index loops."""
    keep = [i for i in range(len(dims)) if i not in sys]
    kd = [dims[i] for i in keep]
    out = np.zeros((int(np.prod(kd)),) * 2, dtype=complex)
    for r in product(*map(range, dims)):
        for c in product(*map(range, dims)):
            if all(r[s] == c[s] for s in sys):
                ri = np.ravel_multi_index([r[k] for k in keep], kd) if keep else 0
                ci = np.ravel_multi_index([c[k] for k in keep], kd) if keep else 0
                out[ri, ci] += a[np.ravel_multi_index(r, dims), np.ravel_multi_index(c, dims)]
    return out


@pytest.mark.parametrize("sys", [s for k in (1, 2) for s in combinations(range(3), k)])
def test_partial_trace_against_loops(sys):
    a = ginibre(12, rng=RNG)
    ref = _reference_partial_trace(a, DIMS, sys)
    np.testing.assert_allclose(partial_trace(a, DIMS, list(sys)), ref, atol=1e-12)
    np.testing.assert_allclose(partial_trace(sps.csr_array(a), DIMS, list(sys)).toarray(), ref,
                               atol=1e-12)
    keep = [i for i in range(3) if i not in sys]
    np.testing.assert_allclose(partial_trace(a, DIMS, keep=keep), ref, atol=1e-12)


def test_partial_trace_variants():
    psi = random_ket(12, rng=RNG)
    rho = ket2dm(psi)
    for sys in ([0], [1], [0, 2], [-1]):
        np.testing.assert_allclose(partial_trace(psi, DIMS, sys), partial_trace(rho, DIMS, sys),
                                   atol=1e-12)
    a = ginibre(12, rng=RNG)
    np.testing.assert_allclose(partial_trace(a, DIMS, [0, 1, 2]), np.trace(a))
    stack = random_state(12, size=4, rng=RNG)
    batch = partial_trace(stack, DIMS, 1)
    for k in range(4):
        np.testing.assert_allclose(batch[k], partial_trace(stack[k], DIMS, 1))
    # two equal subsystems by default
    a4 = ginibre(4, rng=RNG)
    np.testing.assert_allclose(partial_trace(a4, None, 0), partial_trace(a4, (2, 2), 0))
    with pytest.raises(ValueError):
        partial_trace(a, DIMS, 3)
    with pytest.raises(TypeError):
        partial_trace(a, DIMS)


def test_partial_transpose():
    a, b = ginibre(2, rng=RNG), ginibre(3, rng=RNG)
    np.testing.assert_allclose(partial_transpose(kron(a, b), (2, 3), 1), kron(a, b.T))
    np.testing.assert_allclose(partial_transpose(kron(a, b), (2, 3), [0, 1]), kron(a, b).T)
    m = ginibre(12, rng=RNG)
    np.testing.assert_allclose(partial_transpose(sps.csr_array(m), DIMS, [0, 2]).toarray(),
                               partial_transpose(m, DIMS, [0, 2]))
    np.testing.assert_allclose(partial_transpose(partial_transpose(m, DIMS, 1), DIMS, 1), m)


def test_permute_subsystems():
    a, b, c = ginibre(2, rng=RNG), ginibre(3, rng=RNG), ginibre(2, rng=RNG)
    np.testing.assert_allclose(permute_subsystems(kron(a, b, c), DIMS, [1, 2, 0]), kron(b, c, a))
    u, v = random_ket(2, rng=RNG), random_ket(3, rng=RNG)
    np.testing.assert_allclose(permute_subsystems(kron(u, v), (2, 3), [1, 0]), kron(v, u))


def test_reshuffle():
    a, b = ginibre(3, rng=RNG), ginibre(3, rng=RNG)
    np.testing.assert_allclose(reshuffle(kron(a, b)), np.outer(res(a), res(b)))
    m = ginibre(6, rng=RNG)
    r = reshuffle(m, ((2, 3), (3, 2)))
    for i, j, k, l in product(range(2), range(3), range(3), range(2)):
        assert r[i * 3 + k, j * 2 + l] == m[i * 3 + j, k * 2 + l]
