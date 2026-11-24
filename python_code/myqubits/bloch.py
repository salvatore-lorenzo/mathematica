"""Bloch-sphere plots with matplotlib (optional dependency).

Mathematica's ``Show[sfera, assi, point[rho, Red], arrow[psi, Blue]]`` becomes

>>> b = BlochSphere()
>>> b.add_points(rho, color="red").add_vectors(psi, color="blue")
>>> b.show()

States can be kets ``(2,)``, density matrices ``(2, 2)``, Bloch vectors
``(3,)``, or collections of them: ``(n, 2)`` kets, ``(n, 2, 2)`` density
matrices, ``(n, 3)`` vectors, or a Python list of NumPy arrays of any of
these kinds.  A single ``(2, 2)`` array is always a density matrix.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:                                   # pragma: no cover
    raise ImportError("myqubits.bloch needs matplotlib: pip install matplotlib") from exc

from .qubits import bloch_vector

__all__ = ["BlochSphere", "bloch_points"]


def bloch_points(states) -> np.ndarray:
    """Real Bloch coordinates, shape ``(n, 3)``, of one or many qubit states."""
    if isinstance(states, (list, tuple)) and any(isinstance(s, np.ndarray) for s in states):
        # a list of arrays is always a collection of states, e.g. [ket0, ket1]
        return np.concatenate([bloch_points(s) for s in states])
    arr = np.asarray(states)
    if arr.ndim == 1 and arr.shape[0] == 3:
        pts = arr[None]
    elif arr.ndim == 1 and arr.shape[0] == 2:
        pts = bloch_vector(arr)[None]
    elif arr.ndim == 2 and arr.shape == (2, 2):
        pts = bloch_vector(arr)[None]
    elif arr.ndim == 2 and arr.shape[1] == 3:
        pts = arr
    elif arr.ndim == 2 and arr.shape[1] == 2:               # stack of kets
        pts = bloch_vector(arr[:, :, None] * arr.conj()[:, None, :])
    elif arr.ndim == 3 and arr.shape[1:] == (2, 2):
        pts = bloch_vector(arr)
    else:
        raise ValueError(f"cannot interpret an array of shape {arr.shape} as qubit states")
    return np.real(np.asarray(pts, dtype=complex))


class BlochSphere:
    """A Bloch sphere drawn on a 3-D matplotlib axis (``sfera`` + ``assi``).

    The ``add_*`` methods return ``self`` so calls can be chained; the
    figure and axis are available as ``.fig`` and ``.ax``.
    """

    def __init__(self, ax=None, *, figsize=(5, 5), mesh: int = 7, sphere_color="white",
                 sphere_alpha: float = 0.3, wire_color="gray", show_axes: bool = True,
                 labels=("x", "y", "z")):
        if ax is None:
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(projection="3d")
        else:
            self.ax, self.fig = ax, ax.figure
        self._draw_sphere(mesh, sphere_color, sphere_alpha, wire_color)
        if show_axes:
            self._draw_axes(labels)
        ax = self.ax
        ax.set_box_aspect((1, 1, 1))
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-1.05, 1.05)
        ax.set_axis_off()

    def _draw_sphere(self, mesh, color, alpha, wire_color):
        u = np.linspace(0, 2 * np.pi, 72)
        v = np.linspace(0, np.pi, 36)
        x, y, z = np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=False)
        t = np.linspace(0, 2 * np.pi, 120)
        for theta in np.linspace(0, np.pi, mesh + 2)[1:-1]:                  # parallels
            self.ax.plot(np.sin(theta) * np.cos(t), np.sin(theta) * np.sin(t),
                         np.full_like(t, np.cos(theta)), color=wire_color, lw=0.5)
        s = np.linspace(0, np.pi, 60)
        for phi in np.linspace(0, 2 * np.pi, mesh + 2)[1:-1]:                # meridians
            self.ax.plot(np.sin(s) * np.cos(phi), np.sin(s) * np.sin(phi), np.cos(s),
                         color=wire_color, lw=0.5)

    def _draw_axes(self, labels):
        for k, label in enumerate(labels):
            e = np.zeros(3)
            e[k] = 1
            self.ax.plot(*np.stack([-e, e]).T, color="black", lw=1)
            if label:
                self.ax.text(*(1.15 * e), label, ha="center", va="center")

    def add_points(self, states, color="C0", size: float = 30, **kwargs) -> "BlochSphere":
        """Scatter the Bloch vectors of ``states`` (``point``)."""
        pts = bloch_points(states)
        self.ax.scatter(*pts.T, color=color, s=size, depthshade=False, **kwargs)
        return self

    def add_vectors(self, states, color="C0", **kwargs) -> "BlochSphere":
        """Arrows from the origin to the Bloch vectors of ``states`` (``arrow``)."""
        pts = bloch_points(states)
        zeros = np.zeros(len(pts))
        self.ax.quiver(zeros, zeros, zeros, *pts.T, color=color, arrow_length_ratio=0.08,
                       linewidth=1.5, **kwargs)
        self.ax.scatter(*pts.T, color=color, s=15, depthshade=False)
        return self

    def add_line(self, states, color="C0", markers: bool = True, **kwargs) -> "BlochSphere":
        """Polyline through the Bloch vectors of ``states``, e.g. a trajectory (``line``)."""
        pts = bloch_points(states)
        self.ax.plot(*pts.T, color=color, **kwargs)
        if markers:
            self.ax.scatter(*pts.T, color=color, s=10, depthshade=False)
        return self

    def show(self) -> None:
        plt.show()

    def save(self, path, **kwargs) -> None:
        self.fig.savefig(path, **kwargs)
