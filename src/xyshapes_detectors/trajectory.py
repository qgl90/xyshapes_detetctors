"""Z-stepped RK4 trajectories for intercepting detector XY surfaces.

The state is ``(x, y, tx, ty, q/p)`` at ``z``. Coordinates are mm, momentum
is MeV/c, and ``q/p`` is in elementary-charge/(MeV/c). The model has no
material, energy loss, scattering, or decays. A trajectory that approaches a
longitudinal turn is rejected instead of returning misleading sensor hits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .magnetic_field import MEV_PER_TESLA_MM


class TrajectoryError(RuntimeError):
    """A requested track could not be transported monotonically in z."""


@dataclass(frozen=True)
class TrackState:
    """Paraxial charged-track state at one global z position, in mm."""

    x: float
    y: float
    tx: float
    ty: float
    q_over_p: float
    z: float = 0.0

    def __post_init__(self):
        values = (self.x, self.y, self.tx, self.ty, self.q_over_p, self.z)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("track state values must be finite")

    @classmethod
    def from_charge_momentum(cls, charge, momentum_mev, x, y, tx, ty, z=0.0):
        """Create the state from charge and momentum magnitude in MeV/c."""
        momentum_mev = float(momentum_mev)
        if momentum_mev <= 0.0 or not math.isfinite(momentum_mev):
            raise ValueError("momentum must be finite and positive")
        return cls(
            float(x), float(y), float(tx), float(ty),
            float(charge) / momentum_mev, float(z),
        )

    @classmethod
    def from_values(cls, values, z=0.0):
        """Create from ``(x, y, tx, ty, q/p)`` and an explicit ``z``."""
        if len(values) != 5:
            raise ValueError("state must contain (x, y, tx, ty, q/p)")
        return cls(*(float(value) for value in values), z=float(z))

    def as_tuple(self):
        return self.x, self.y, self.tx, self.ty, self.q_over_p


@dataclass(frozen=True)
class Trajectory:
    """Integrated trajectory nodes and cubic-Hermite XY interpolation."""

    z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    q_over_p: float

    @property
    def points(self):
        """Trajectory nodes as an ``(n, 3)`` array of ``(x, y, z)`` in mm."""
        return np.column_stack((self.x, self.y, self.z))

    def xy_at(self, z):
        """Return ``(..., 2)`` XY intercepts at requested z values in mm."""
        requested = np.asarray(z, dtype=float)
        if not np.all(np.isfinite(requested)):
            raise ValueError("requested z values must be finite")
        if np.any(requested < self.z[0]) or np.any(requested > self.z[-1]):
            raise ValueError(
                f"requested z must lie within [{self.z[0]:g}, {self.z[-1]:g}] mm"
            )

        flat = requested.reshape(-1)
        upper = np.searchsorted(self.z, flat, side="right")
        upper = np.clip(upper, 1, len(self.z) - 1)
        lower = upper - 1
        z0, z1 = self.z[lower], self.z[upper]
        h = z1 - z0
        t = np.clip((flat - z0) / h, 0.0, 1.0)
        t2, t3 = t * t, t * t * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2

        x = (
            h00 * self.x[lower] + h10 * h * self.tx[lower]
            + h01 * self.x[upper] + h11 * h * self.tx[upper]
        )
        y = (
            h00 * self.y[lower] + h10 * h * self.ty[lower]
            + h01 * self.y[upper] + h11 * h * self.ty[upper]
        )
        # Preserve exact node values and shape for both scalar and array input.
        exact = np.searchsorted(self.z, flat, side="left")
        candidates = np.flatnonzero(exact < len(self.z))
        at_node = np.zeros(flat.shape, dtype=bool)
        at_node[candidates] = self.z[exact[candidates]] == flat[candidates]
        x[at_node] = self.x[exact[at_node]]
        y[at_node] = self.y[exact[at_node]]
        return np.column_stack((x, y)).reshape((*requested.shape, 2))

    def intercepts(self, sensor_z):
        """Alias emphasizing XY intercepts at detector sensor planes."""
        return self.xy_at(sensor_z)


def _derivative(field, z, state, q_over_p, min_forward_cosine):
    x, y, tx, ty = state
    inverse_cosine = math.sqrt(1.0 + tx * tx + ty * ty)
    if not math.isfinite(inverse_cosine) or 1.0 / inverse_cosine < min_forward_cosine:
        raise TrajectoryError(
            f"trajectory is curling or nearly transverse to z near z={z:.3f} mm; "
            "the monotonic-z approximation cannot reach downstream planes"
        )

    bx, by, bz = field.b_at(x, y, z)
    curvature = MEV_PER_TESLA_MM * q_over_p * inverse_cosine
    dtx = curvature * (tx * ty * bx - (1.0 + tx * tx) * by + ty * bz)
    dty = curvature * ((1.0 + ty * ty) * bx - tx * ty * by - tx * bz)
    return np.asarray((tx, ty, dtx, dty), dtype=float)


def _rk4_z_step(field, z, state, q_over_p, step, min_forward_cosine):
    k1 = _derivative(field, z, state, q_over_p, min_forward_cosine)
    k2 = _derivative(
        field, z + step / 2.0, state + step * k1 / 2.0,
        q_over_p, min_forward_cosine,
    )
    k3 = _derivative(
        field, z + step / 2.0, state + step * k2 / 2.0,
        q_over_p, min_forward_cosine,
    )
    k4 = _derivative(
        field, z + step, state + step * k3, q_over_p, min_forward_cosine
    )
    result = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if not np.all(np.isfinite(result)):
        raise TrajectoryError(f"non-finite track state reached near z={z:g} mm")
    return result


def create_trajectory(
    field,
    state,
    z_end,
    *,
    sensor_z=(),
    step_mm=10.0,
    min_forward_cosine=0.05,
    max_steps=1_000_000,
):
    """Integrate one track forward in z and provide XY sensor intercepts.

    ``state`` is a :class:`TrackState` or ``(x, y, tx, ty, q/p)`` tuple. Sensor
    plane z values are added as integration nodes, then :meth:`Trajectory.xy_at`
    uses cubic-Hermite interpolation for any other requested z in the span.
    Transport stops with :class:`TrajectoryError` rather than returning an
    incomplete path if it curls, approaches a z turn, or exceeds ``max_steps``.
    """
    if not isinstance(state, TrackState):
        state = TrackState.from_values(state)
    z_start, z_end = float(state.z), float(z_end)
    step_mm = float(step_mm)
    min_forward_cosine = float(min_forward_cosine)
    if not math.isfinite(z_end) or z_end <= z_start:
        raise ValueError("z_end must be finite and downstream of the initial z")
    if not math.isfinite(step_mm) or step_mm <= 0.0:
        raise ValueError("step_mm must be finite and positive")
    if not 0.0 < min_forward_cosine <= 1.0:
        raise ValueError("min_forward_cosine must be in (0, 1]")
    sensor_z = np.asarray(sensor_z, dtype=float).reshape(-1)
    if not np.all(np.isfinite(sensor_z)):
        raise ValueError("sensor z values must be finite")
    if np.any((sensor_z < z_start) | (sensor_z > z_end)):
        raise ValueError("all sensor z values must lie within the requested span")

    count = int(math.ceil((z_end - z_start) / step_mm))
    if count > max_steps:
        raise TrajectoryError(
            f"requested path needs {count} z steps, exceeding max_steps={max_steps}"
        )
    regular = np.linspace(z_start, z_end, count + 1)
    nodes = np.unique(np.concatenate((regular, sensor_z, (z_start, z_end))))
    if nodes.size - 1 > max_steps:
        raise TrajectoryError(
            f"requested path needs {nodes.size - 1} z steps, exceeding max_steps={max_steps}"
        )

    states = np.empty((nodes.size, 4), dtype=float)
    states[0] = (state.x, state.y, state.tx, state.ty)
    for index in range(nodes.size - 1):
        z0, z1 = nodes[index], nodes[index + 1]
        states[index + 1] = _rk4_z_step(
            field, z0, states[index], state.q_over_p, z1 - z0,
            min_forward_cosine,
        )
    # Also catch a near-turn reached exactly at the final integration node.
    _derivative(
        field, nodes[-1], states[-1], state.q_over_p, min_forward_cosine
    )
    return Trajectory(
        nodes, states[:, 0], states[:, 1], states[:, 2], states[:, 3],
        state.q_over_p,
    )
