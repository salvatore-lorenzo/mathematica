import numpy as np
import pytest

from myqubits import channels as ch
from myqubits.bases import pauli_basis
from myqubits.core import cnot, ket0, ket1, kron
from myqubits.random_states import random_kraus, random_state

RNG = np.random.default_rng(3)


@pytest.fixture
def kraus():
    return random_kraus(3, 4, rng=RNG)


def test_all_representations_agree(kraus):
    rho = random_state(3, rng=RNG)
    ref = sum(k @ rho @ k.conj().T for k in kraus)
    np.testing.assert_allclose(ch.apply_kraus(kraus, rho), ref, atol=1e-12)
    np.testing.assert_allclose(ch.apply_superop(ch.kraus_to_superop(kraus), rho), ref, atol=1e-12)
    np.testing.assert_allclose(ch.apply_choi(ch.kraus_to_choi(kraus), rho), ref, atol=1e-12)
    np.testing.assert_allclose(ch.apply_ptm(ch.kraus_to_ptm(kraus), rho), ref, atol=1e-12)
    np.testing.assert_allclose(ch.apply_kraus(ch.choi_to_kraus(ch.kraus_to_choi(kraus)), rho), ref,
                               atol=1e-12)
    np.testing.assert_allclose(ch.apply_kraus(ch.ptm_to_kraus(ch.kraus_to_ptm(kraus)), rho), ref,
                               atol=1e-12)


def test_round_trips(kraus):
    lsup = ch.kraus_to_superop(kraus)
    choi = ch.kraus_to_choi(kraus)
    ptm = ch.kraus_to_ptm(kraus)
    np.testing.assert_allclose(ch.superop_to_choi(lsup), choi, atol=1e-12)
    np.testing.assert_allclose(ch.choi_to_superop(choi), lsup, atol=1e-12)
    np.testing.assert_allclose(ch.ptm_to_choi(ptm), choi, atol=1e-12)
    np.testing.assert_allclose(ch.choi_to_ptm(choi), ptm, atol=1e-12)
    np.testing.assert_allclose(ch.map_to_ptm(lambda r: ch.apply_kraus(kraus, r), 3), ptm, atol=1e-12)
    assert len(ch.choi_to_kraus(choi)) <= 4
    assert np.isrealobj(ptm)                     # Hermiticity-preserving map, Hermitian basis


def test_standard_channels():
    rho = random_state(2, rng=RNG)
    np.testing.assert_allclose(ch.pauli_channel(rho, (0.3, 0.3, 0.3)), ch.depolarizing(rho, 0.3))
    np.testing.assert_allclose(ch.apply_kraus(ch.pauli_kraus(0.1, 0.2, 0.3), rho),
                               ch.pauli_channel(rho, (0.1, 0.2, 0.3)), atol=1e-12)
    g = 0.4
    out = ch.amplitude_damping(rho, g)
    np.testing.assert_allclose(out[0, 0], rho[0, 0] + g * rho[1, 1])
    np.testing.assert_allclose(out[0, 1], np.sqrt(1 - g) * rho[0, 1])
    ptm = ch.map_to_ptm(lambda r: ch.depolarizing(r, 0.2), 2)
    np.testing.assert_allclose(ptm, np.diag([1, 0.8, 0.8, 0.8]), atol=1e-12)
    stack = random_state(2, size=6, rng=RNG)
    batch = ch.amplitude_damping(stack, g)
    np.testing.assert_allclose(batch[2], ch.amplitude_damping(stack[2], g))


def test_channel_object(kraus):
    c = ch.Channel.from_kraus(kraus)
    assert c.is_cptp()
    rho = random_state(3, rng=RNG)
    np.testing.assert_allclose((c @ c)(rho), c(c(rho)), atol=1e-12)
    np.testing.assert_allclose((c ** 3)(rho), c(c(c(rho))), atol=1e-12)
    np.testing.assert_allclose(ch.Channel.from_choi(c.choi())(rho), c(rho), atol=1e-12)
    np.testing.assert_allclose(ch.Channel.from_ptm(c.ptm())(rho), c(rho), atol=1e-12)
    np.testing.assert_allclose(ch.Channel.from_kraus(c.kraus())(rho), c(rho), atol=1e-12)
    ad = ch.Channel.from_kraus(ch.amplitude_damping_kraus(0.5))
    assert ad.is_tp() and not ad.is_unital()
    transpose = ch.Channel.from_map(lambda r: r.T, 2)
    assert transpose.is_tp() and not transpose.is_cp()
    with pytest.warns(UserWarning):
        transpose.kraus()


def test_rectangular_kraus_partial_trace_channel():
    # Tr_B as a channel from 4x4 to 2x2: K_i = I x <i|
    kraus = np.array([kron(np.eye(2), ket0[None]), kron(np.eye(2), ket1[None])])
    c = ch.Channel.from_kraus(kraus)
    assert c.dims == (2, 4) and c.is_cptp()
    rho = random_state(4, rng=RNG)
    np.testing.assert_allclose(c(rho), rho.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3), atol=1e-12)
    np.testing.assert_allclose(ch.Channel.from_choi(c.choi(), c.dims)(rho), c(rho), atol=1e-12)


def test_pauli_transfer_matrix_of_cnot_is_signed_permutation():
    ptm = ch.kraus_to_ptm(cnot[None], basis=pauli_basis(2))
    assert np.allclose(np.abs(ptm).sum(0), 1) and np.allclose(np.abs(ptm).sum(1), 1)
