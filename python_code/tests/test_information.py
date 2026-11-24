import numpy as np
import pytest

from myqubits.core import ket, ket2dm, phi_plus, psi_minus, ghz3, kron
from myqubits import discord, entanglement as ent, entropy, measures
from myqubits.random_states import random_ket, random_state, random_unitary
from myqubits.subsystems import partial_trace

RNG = np.random.default_rng(2)
BELL = ket2dm(phi_plus)


def werner(p):
    return p * ket2dm(psi_minus) + (1 - p) * np.eye(4) / 4


def test_bell_state_values():
    assert entropy.von_neumann(BELL) == pytest.approx(0, abs=1e-12)
    assert entropy.mutual_information(BELL) == pytest.approx(2)
    assert ent.concurrence(BELL) == pytest.approx(1)
    assert ent.concurrence(phi_plus) == pytest.approx(1)
    assert ent.entanglement_of_formation(BELL) == pytest.approx(1)
    assert ent.negativity(BELL) == pytest.approx(0.5)
    assert ent.log_negativity(BELL) == pytest.approx(1)
    assert discord.quantum_discord(BELL) == pytest.approx(1, abs=1e-9)
    assert discord.geometric_discord(BELL, 0) == pytest.approx(1)
    assert discord.geometric_discord(BELL, 1) == pytest.approx(1)


def test_product_state_has_no_correlations():
    rho = kron(random_state(2, rng=RNG), random_state(2, rng=RNG))
    assert entropy.mutual_information(rho) == pytest.approx(0, abs=1e-10)
    assert ent.concurrence(rho) == pytest.approx(0, abs=1e-10)
    assert ent.negativity(rho) == 0
    assert discord.quantum_discord(rho) == pytest.approx(0, abs=1e-8)
    assert discord.geometric_discord(rho) == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.8, 1.0])
def test_werner_state(p):
    rho = werner(p)
    assert ent.concurrence(rho) == pytest.approx(max(0, (3 * p - 1) / 2), abs=1e-10)
    assert ent.negativity(rho) == pytest.approx(max(0, (3 * p - 1) / 4), abs=1e-10)
    xlogx = lambda x: x * np.log2(x) if x > 0 else 0.0
    exact = (xlogx(1 - p) - 2 * xlogx(1 + p) + xlogx(1 + 3 * p)) / 4
    for measured in (0, 1):
        assert discord.quantum_discord(rho, measured) == pytest.approx(exact, abs=1e-9)
    assert discord.geometric_discord(rho) == pytest.approx(p ** 2, abs=1e-12)


def test_concurrence_ket_matches_density_matrix_and_x_formula():
    psi = random_ket(4, rng=RNG)
    assert ent.concurrence(ket2dm(psi)) == pytest.approx(ent.concurrence(psi), abs=1e-7)
    x = np.diag([0.4, 0.1, 0.2, 0.3]).astype(complex)
    x[0, 3], x[3, 0] = 0.25j, -0.25j
    x[1, 2] = x[2, 1] = 0.05
    assert ent.concurrence(x) == pytest.approx(ent.concurrence_x(x), abs=1e-12)


def test_discord_full_output_and_local_unitary_invariance():
    rho = random_state(4, rng=RNG)
    d, theta, phi = discord.quantum_discord(rho, full_output=True)
    s_a = entropy.von_neumann(partial_trace(rho, (2, 2), 1))
    cond = discord.conditional_entropy_after_measurement(rho, theta, phi)
    assert d == pytest.approx(s_a - entropy.von_neumann(rho) + cond, abs=1e-12)
    u = kron(random_unitary(2, rng=RNG), random_unitary(2, rng=RNG))
    assert discord.quantum_discord(u @ rho @ u.conj().T) == pytest.approx(d, abs=1e-8)


def test_fidelity_and_distances():
    rho, sigma = random_state(3, rng=RNG), random_state(3, rng=RNG)
    assert measures.fidelity(rho, rho) == pytest.approx(1)
    assert measures.fidelity(rho, sigma) == pytest.approx(measures.fidelity(sigma, rho))
    psi, phi = random_ket(3, rng=RNG), random_ket(3, rng=RNG)
    assert measures.fidelity(psi, phi) == pytest.approx(abs(np.vdot(psi, phi)) ** 2)
    assert measures.fidelity(psi, sigma) == pytest.approx(measures.fidelity(ket2dm(psi), sigma))
    assert measures.trace_distance(ket("0"), ket("1")) == pytest.approx(1)
    f = measures.fidelity(rho, sigma)
    t = measures.trace_distance(rho, sigma)
    assert 1 - np.sqrt(f) <= t + 1e-12 <= np.sqrt(1 - f) + 2e-12   # Fuchs-van de Graaf
    stack = random_state(3, size=5, rng=RNG)
    np.testing.assert_allclose(measures.purity(stack), [measures.purity(s) for s in stack])


def test_entropies():
    rho = random_state(4, rng=RNG)
    w = np.linalg.eigvalsh(rho)
    assert entropy.von_neumann(rho) == pytest.approx(-(w * np.log2(w)).sum())
    assert entropy.renyi(rho, 1 + 1e-7) == pytest.approx(entropy.von_neumann(rho), abs=1e-6)
    assert entropy.renyi(rho, 0) == pytest.approx(2)            # log2(rank 4)
    assert entropy.renyi(rho, np.inf) == pytest.approx(-np.log2(w.max()))
    assert entropy.renyi(rho, 2) == pytest.approx(-np.log2(np.sum(w ** 2)))
    sigma = random_state(4, rng=RNG)
    assert entropy.relative_entropy(rho, rho) == pytest.approx(0, abs=1e-10)
    assert entropy.relative_entropy(rho, sigma) > 0
    assert entropy.relative_entropy(rho, ket2dm(phi_plus)) == np.inf
    stack = random_state(4, size=3, rng=RNG)
    np.testing.assert_allclose(entropy.von_neumann(stack), [entropy.von_neumann(s) for s in stack])
    # pure tripartite states: I3 = 0 for GHZ, mutual information of parts consistent
    assert entropy.tripartite_information(ghz3) == pytest.approx(0, abs=1e-10)
    rho8 = random_state(8, rng=RNG)
    i3 = (entropy.mutual_information(rho8, (2, 2, 2), 0, 1) + entropy.mutual_information(rho8, (2, 2, 2), 0, 2)
          - entropy.mutual_information(rho8, (2, 2, 2), 0, [1, 2]))
    assert entropy.tripartite_information(rho8) == pytest.approx(i3)
    assert entropy.quantum_conditional_entropy(BELL) == pytest.approx(-1)


def test_classical_information():
    p = np.outer([0.3, 0.7], [0.2, 0.5, 0.3])                    # independent X, Y
    assert entropy.classical_mutual_information(p) == pytest.approx(0, abs=1e-12)
    assert entropy.conditional_entropy(p, given=1) == pytest.approx(entropy.shannon_entropy([0.3, 0.7]))
    corr = np.diag([0.5, 0.5])
    assert entropy.conditional_entropy(corr) == pytest.approx(0)
    assert entropy.classical_mutual_information(corr.ravel(), (2, 2)) == pytest.approx(1)
    q = RNG.dirichlet(np.ones(12))
    np.testing.assert_allclose(entropy.marginal(q, (2, 3, 2), 1), q.reshape(2, 3, 2).sum((0, 2)))
    np.testing.assert_allclose(entropy.marginal(q, (2, 3, 2), [0, 2]), q.reshape(2, 3, 2).sum(1))
    assert entropy.binary_entropy(0.5) == pytest.approx(1)
    assert entropy.shannon_entropy([1, 0, 0]) == 0
