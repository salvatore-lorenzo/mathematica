"""Random states, unitaries, POVMs and channels.

Every function takes ``rng``: ``None`` (fresh entropy), an integer seed or a
:class:`numpy.random.Generator`, so results are reproducible.  Most also take
``size`` to draw a whole stack at once (shape ``(*size, d)`` or
``(*size, d, d)``), fully vectorized.
"""

from __future__ import annotations

import numpy as np

from .core import dag, hermitian_function

__all__ = ["random_simplex", "random_ket", "ginibre", "random_unitary",
           "random_special_unitary", "random_orthogonal", "random_state",
           "random_povm", "random_kraus"]


def _rng(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _size(size) -> tuple[int, ...]:
    if size is None:
        return ()
    return (size,) if np.ndim(size) == 0 else tuple(size)


def random_simplex(d: int, size=None, rng=None) -> np.ndarray:
    """Probability vector uniformly distributed on the simplex (flat Dirichlet)."""
    return _rng(rng).dirichlet(np.ones(d), size=size)


def random_ket(d: int, size=None, rng=None) -> np.ndarray:
    """Haar-random normalized ket(s): a normalized complex Gaussian vector."""
    g = ginibre(d, 1, size=size, rng=rng)[..., 0]
    return g / np.linalg.norm(g, axis=-1, keepdims=True)


def ginibre(m: int, n: int | None = None, size=None, rng=None) -> np.ndarray:
    """``m x n`` matrix with independent standard normal real and imaginary parts."""
    rng = _rng(rng)
    shape = _size(size) + (m, m if n is None else n)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _haar_from_qr(g):
    q, r = np.linalg.qr(g)
    diag = np.diagonal(r, axis1=-2, axis2=-1)
    return q * (diag / np.abs(diag))[..., None, :]      # Mezzadri's phase fix


def random_unitary(d: int, size=None, rng=None) -> np.ndarray:
    """Haar-random ``d x d`` unitary (QR of a Ginibre matrix with phases fixed)."""
    return _haar_from_qr(ginibre(d, d, size, rng))


def random_special_unitary(d: int, size=None, rng=None) -> np.ndarray:
    """Haar-random unitary divided by a ``d``-th root of its determinant."""
    u = random_unitary(d, size, rng)
    return u / (np.linalg.det(u) ** (1 / d))[..., None, None]


def random_orthogonal(d: int, size=None, rng=None) -> np.ndarray:
    """Haar-random real orthogonal ``d x d`` matrix."""
    g = _rng(rng).standard_normal(_size(size) + (d, d))
    return _haar_from_qr(g)


def random_state(d: int, kind: str | int = "hs", size=None, rng=None) -> np.ndarray:
    """Random density matrix.

    ``kind`` is ``"hs"`` (Hilbert-Schmidt measure), ``"bures"`` (Bures
    measure) or an integer ``k >= d``: the measure induced by tracing out a
    ``k``-dimensional environment from a Haar-random pure state.
    """
    rng = _rng(rng)
    if isinstance(kind, str) and kind.lower() == "hs":
        g = ginibre(d, d, size, rng)
    elif isinstance(kind, str) and kind.lower() == "bures":
        u = random_unitary(d, size, rng)
        g = (np.eye(d) + u) @ ginibre(d, d, size, rng)
    elif isinstance(kind, (int, np.integer)) and kind >= d:
        g = ginibre(d, int(kind), size, rng)
    else:
        raise ValueError(f"kind must be 'hs', 'bures' or an integer >= {d}, got {kind!r}")
    a = g @ dag(g)
    return a / np.trace(a, axis1=-2, axis2=-1).real[..., None, None]


def random_povm(d: int, k: int, rng=None) -> np.ndarray:
    """Random POVM: ``k`` positive ``d x d`` matrices summing to the identity, shape ``(k, d, d)``."""
    g = ginibre(d, k, size=k, rng=rng)
    w = g @ dag(g)
    s = hermitian_function(w.sum(0), lambda x: x ** -0.5)
    return s @ w @ s


def random_kraus(d: int, k: int, rng=None) -> np.ndarray:
    """``k`` random Kraus operators of a trace-preserving channel on dimension ``d``, shape ``(k, d, d)``.

    ``K_j = G_j H^{-1/2}`` with ``H = sum_j G_j† G_j``.  Mathematica's
    ``RandomKrausOPS[n, k]`` on ``n`` qubits is ``random_kraus(2**n, k)``.
    """
    g = ginibre(d, d, size=k, rng=rng)
    h = (dag(g) @ g).sum(0)
    return g @ hermitian_function(h, lambda x: x ** -0.5)
