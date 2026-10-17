r"""Propagate charged tracks through a magnetic field map, in millimetres.

A track is fully described by its charge, its momentum magnitude, its
direction, and its origin.  Everything is millimetres and MeV/c, so the
trajectory can be intersected directly with the Shapely XY acceptances and the
``z`` column of the layout CSVs.

Equation of motion
------------------
With :math:`\hat t` the unit direction and :math:`s` the path length in mm,

.. math::

    \frac{d\vec r}{ds} = \hat t, \qquad
    \frac{d\hat t}{ds} = \frac{0.299792458 \, q}{p} \, \hat t \times \vec B

for ``q`` in units of the elementary charge, ``p`` in MeV/c and ``B`` in
tesla.  ``|p|`` is conserved: this is a vacuum transport model with no energy
loss, no multiple scattering and no decays.

The integrator is a classical fourth-order Runge-Kutta stepper, vectorised
over tracks so that large samples propagate in one pass.  Tracks are marched
through a sorted list of plane ``z`` values and the crossing point of each
plane is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .magnetic_field import MEV_PER_TESLA_MM


#: Reasons a track stopped being transported.
STOP_REASONS = ("completed", "backward", "escaped", "max_steps")


@dataclass(frozen=True)
class Track:
    """One charged track leaving ``(x, y, z)`` with a unit direction.

    ``charge`` is in units of the elementary charge and ``p`` is the momentum
    magnitude in MeV/c.  ``(dx, dy, dz)`` need not be normalised on input;
    :meth:`normalised` returns the unit-direction version, and every
    constructor already normalises.
    """

    charge: float
    p: float
    dx: float
    dy: float
    dz: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # ------------------------------------------------------------ builders

    @classmethod
    def from_direction(cls, charge, p, direction, origin=(0.0, 0.0, 0.0)):
        """A track of momentum ``p`` (MeV/c) along ``direction``."""
        dx, dy, dz = (float(value) for value in direction)
        x, y, z = (float(value) for value in origin)
        return cls(float(charge), float(p), dx, dy, dz, x, y, z).normalised()

    @classmethod
    def from_momentum(cls, charge, px, py, pz, origin=(0.0, 0.0, 0.0)):
        """A track from its momentum components in MeV/c."""
        p = math.sqrt(float(px) ** 2 + float(py) ** 2 + float(pz) ** 2)
        if p == 0.0:
            raise ValueError("the momentum vector must be non-zero")
        return cls.from_direction(charge, p, (px, py, pz), origin)

    @classmethod
    def from_slopes(cls, charge, p, tx, ty, origin=(0.0, 0.0, 0.0)):
        """A track from LHCb slopes ``tx = px/pz`` and ``ty = py/pz``."""
        return cls.from_direction(charge, p, (tx, ty, 1.0), origin)

    @classmethod
    def from_eta_phi(cls, charge, p, eta, phi, origin=(0.0, 0.0, 0.0)):
        """A track from pseudorapidity and azimuth, with ``p`` in MeV/c."""
        theta = 2.0 * math.atan(math.exp(-float(eta)))
        direction = (
            math.sin(theta) * math.cos(float(phi)),
            math.sin(theta) * math.sin(float(phi)),
            math.cos(theta),
        )
        return cls.from_direction(charge, p, direction, origin)

    @classmethod
    def from_pt_eta_phi(cls, charge, pt, eta, phi, origin=(0.0, 0.0, 0.0)):
        """A track from transverse momentum (MeV/c), pseudorapidity and azimuth."""
        theta = 2.0 * math.atan(math.exp(-float(eta)))
        sin_theta = math.sin(theta)
        if sin_theta == 0.0:
            raise ValueError("eta is too large to convert pt into p")
        return cls.from_eta_phi(charge, float(pt) / sin_theta, eta, phi, origin)

    # --------------------------------------------------------- projections

    def normalised(self):
        norm = math.sqrt(self.dx**2 + self.dy**2 + self.dz**2)
        if norm == 0.0:
            raise ValueError("the direction must be non-zero")
        return replace(
            self, dx=self.dx / norm, dy=self.dy / norm, dz=self.dz / norm
        )

    @property
    def direction(self):
        return (self.dx, self.dy, self.dz)

    @property
    def origin(self):
        return (self.x, self.y, self.z)

    @property
    def px(self):
        return self.p * self.dx

    @property
    def py(self):
        return self.p * self.dy

    @property
    def pz(self):
        return self.p * self.dz

    @property
    def pt(self):
        return self.p * math.hypot(self.dx, self.dy)

    @property
    def tx(self):
        return self.dx / self.dz if self.dz else math.inf

    @property
    def ty(self):
        return self.dy / self.dz if self.dz else math.inf

    @property
    def theta(self):
        return math.atan2(math.hypot(self.dx, self.dy), self.dz)

    @property
    def phi(self):
        return math.atan2(self.dy, self.dx)

    @property
    def eta(self):
        transverse = math.hypot(self.dx, self.dy)
        if transverse == 0.0:
            return math.inf if self.dz > 0 else -math.inf
        return -math.log(math.tan(0.5 * math.atan2(transverse, self.dz)))

    def as_row(self):
        """The track as a flat dictionary, ready to be augmented with masks."""
        return {
            "charge": self.charge,
            "p": self.p,
            "px": self.px,
            "py": self.py,
            "pz": self.pz,
            "pt": self.pt,
            "eta": self.eta,
            "phi": self.phi,
            "tx": self.tx,
            "ty": self.ty,
            "x0": self.x,
            "y0": self.y,
            "z0": self.z,
        }


@dataclass(frozen=True)
class PlaneCrossings:
    """Where a sample of tracks crossed each requested ``z`` plane.

    ``x``, ``y`` and ``s`` have shape ``(n_tracks, n_planes)``.  Entries where
    ``crossed`` is ``False`` are ``nan``.  ``s`` is the path length from the
    track origin in millimetres.
    """

    plane_z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    crossed: np.ndarray
    z_exit: np.ndarray
    s_exit: np.ndarray
    stop_reason: np.ndarray
    steps: int

    @property
    def n_tracks(self):
        return self.x.shape[0]

    @property
    def n_planes(self):
        return self.x.shape[1]

    def plane(self, index):
        """``(x, y, crossed)`` arrays for one plane index."""
        return self.x[:, index], self.y[:, index], self.crossed[:, index]

    def positions(self, index):
        """``(n_tracks, 3)`` crossing positions for one plane index."""
        z = np.full(self.x.shape[0], self.plane_z[index])
        return np.stack((self.x[:, index], self.y[:, index], z), axis=-1)


def tracks_to_state(tracks):
    """Turn an iterable of :class:`Track` into propagation arrays."""
    tracks = list(tracks)
    if not tracks:
        raise ValueError("at least one track is needed")
    charge = np.array([track.charge for track in tracks], dtype=float)
    p = np.array([track.p for track in tracks], dtype=float)
    position = np.array([track.origin for track in tracks], dtype=float)
    direction = np.array([track.direction for track in tracks], dtype=float)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("track directions must be non-zero")
    return charge, p, position, direction / norm


def _cross(a, b):
    """Cross product of two ``(n, 3)`` arrays, faster than ``np.cross``."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def _rk4_step(field, position, direction, kappa, h):
    """One RK4 step of signed length ``h`` (shape ``(n, 1)``) in path length."""
    d1 = direction
    a1 = kappa * _cross(d1, field.b_at_points(position))

    half = 0.5 * h
    d2 = direction + half * a1
    a2 = kappa * _cross(d2, field.b_at_points(position + half * d1))

    d3 = direction + half * a2
    a3 = kappa * _cross(d3, field.b_at_points(position + half * d2))

    d4 = direction + h * a3
    a4 = kappa * _cross(d4, field.b_at_points(position + h * d3))

    new_position = position + (h / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
    new_direction = direction + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    norm = np.linalg.norm(new_direction, axis=1, keepdims=True)
    np.divide(new_direction, norm, out=new_direction, where=norm > 0)
    return new_position, new_direction


def propagate(
    tracks,
    field,
    plane_z,
    *,
    step=10.0,
    step_free=100.0,
    envelope=(10000.0, 10000.0),
    max_steps=200000,
):
    """Transport ``tracks`` through ``field`` and record plane crossings.

    ``tracks`` is an iterable of :class:`Track`, or the four state arrays
    ``(charge, p, position, direction)`` from :func:`tracks_to_state` passed as
    a tuple.  ``plane_z`` is any iterable of plane positions in millimetres; it is sorted
    internally and the returned arrays follow that sorted order, which
    :attr:`PlaneCrossings.plane_z` reports.  Planes upstream of a track's
    origin are never crossed, because transport only runs forward in ``z``.

    ``step`` is the RK4 step length inside the field map and ``step_free`` the
    longer step used where the map gives no field, which is exact there because
    the motion is a straight line.  A track stops as soon as it turns around in
    ``z``, leaves the ``(|x|, |y|)`` ``envelope``, or passes the last plane.
    """
    if isinstance(tracks, tuple) and len(tracks) == 4:
        # Already-built state arrays, as returned by tracks_to_state.
        charge, p, position, direction = (
            np.asarray(item, dtype=float) for item in tracks
        )
        position = position.reshape(-1, 3).copy()
        direction = direction.reshape(-1, 3).copy()
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        if np.any(norm == 0.0):
            raise ValueError("track directions must be non-zero")
        direction /= norm
    else:
        charge, p, position, direction = tracks_to_state(tracks)
    if np.any(p <= 0.0):
        raise ValueError("every track needs a positive momentum")

    plane_z = np.sort(np.asarray(plane_z, dtype=float).ravel())
    n_planes = plane_z.size
    if n_planes == 0:
        raise ValueError("at least one plane z is needed")
    # A +inf sentinel keeps the "next plane" index in range once a track has
    # passed every plane, so no bounds check is needed in the inner loop.
    padded_z = np.append(plane_z, np.inf)

    n_tracks = position.shape[0]
    hit_x = np.full((n_tracks, n_planes), np.nan)
    hit_y = np.full((n_tracks, n_planes), np.nan)
    hit_s = np.full((n_tracks, n_planes), np.nan)
    crossed = np.zeros((n_tracks, n_planes), dtype=bool)
    z_exit = position[:, 2].copy()
    s_exit = np.zeros(n_tracks)
    stop_code = np.zeros(n_tracks, dtype=np.int8)

    # kappa = 0.299792458 * q / p, the curvature per tesla, in 1/mm.
    kappa = (MEV_PER_TESLA_MM * charge / p).reshape(-1, 1)
    z_low, z_high = field.bounds[2], field.bounds[5]
    envelope_x, envelope_y = (abs(float(value)) for value in envelope)

    forward = direction[:, 2] > 0.0
    index = np.flatnonzero(forward)
    stop_code[~forward] = STOP_REASONS.index("backward")

    position = position[index]
    direction = direction[index]
    kappa = kappa[index]
    path = np.zeros(index.size)
    next_plane = np.searchsorted(plane_z, position[:, 2], side="left")

    steps = 0
    while index.size and steps < max_steps:
        steps += 1
        z0 = position[:, 2]
        in_map = (z0 >= z_low) & (z0 <= z_high)
        h = np.where(in_map, step, step_free).reshape(-1, 1)

        start_position, start_direction = position, direction
        position, direction = _rk4_step(field, position, direction, kappa, h)
        z1 = position[:, 2]
        path = path + h[:, 0]

        # Record every plane the step swept past, newest-to-furthest.
        while True:
            target = padded_z[next_plane]
            hit = z1 >= target
            if not hit.any():
                break
            rows = np.flatnonzero(hit)
            sub_h = h[rows]
            start_z = start_position[rows, 2]
            span = z1[rows] - start_z
            fraction = np.where(
                np.abs(span) > 0.0, (target[rows] - start_z) / span, 0.0
            )
            np.clip(fraction, 0.0, 1.0, out=fraction)
            cross_position, cross_direction = _rk4_step(
                field,
                start_position[rows],
                start_direction[rows],
                kappa[rows],
                fraction.reshape(-1, 1) * sub_h,
            )
            # The RK4 sub-step lands close to the plane; slide along the local
            # tangent to sit exactly on it.
            slip = np.where(
                np.abs(cross_direction[:, 2]) > 1e-12,
                (target[rows] - cross_position[:, 2]) / cross_direction[:, 2],
                0.0,
            )
            columns = next_plane[rows]
            global_rows = index[rows]
            hit_x[global_rows, columns] = (
                cross_position[:, 0] + slip * cross_direction[:, 0]
            )
            hit_y[global_rows, columns] = (
                cross_position[:, 1] + slip * cross_direction[:, 1]
            )
            hit_s[global_rows, columns] = (
                path[rows] - sub_h[:, 0] * (1.0 - fraction) + slip
            )
            crossed[global_rows, columns] = True
            next_plane[rows] += 1

        turned = z1 <= z0
        escaped = (np.abs(position[:, 0]) > envelope_x) | (
            np.abs(position[:, 1]) > envelope_y
        )
        completed = next_plane >= n_planes
        stopping = turned | escaped | completed
        if stopping.any():
            rows = np.flatnonzero(stopping)
            global_rows = index[rows]
            z_exit[global_rows] = z1[rows]
            s_exit[global_rows] = path[rows]
            codes = np.where(
                completed[rows],
                STOP_REASONS.index("completed"),
                np.where(
                    turned[rows],
                    STOP_REASONS.index("backward"),
                    STOP_REASONS.index("escaped"),
                ),
            )
            stop_code[global_rows] = codes
            keep = ~stopping
            index = index[keep]
            position = position[keep]
            direction = direction[keep]
            kappa = kappa[keep]
            path = path[keep]
            next_plane = next_plane[keep]

    if index.size:
        z_exit[index] = position[:, 2]
        s_exit[index] = path
        stop_code[index] = STOP_REASONS.index("max_steps")

    return PlaneCrossings(
        plane_z=plane_z,
        x=hit_x,
        y=hit_y,
        s=hit_s,
        crossed=crossed,
        z_exit=z_exit,
        s_exit=s_exit,
        stop_reason=np.asarray(STOP_REASONS, dtype=object)[stop_code],
        steps=steps,
    )


def propagate_to_z(tracks, field, z, **kwargs):
    """``(x, y, crossed)`` arrays at a single ``z`` plane, in millimetres."""
    crossings = propagate(tracks, field, [z], **kwargs)
    return crossings.plane(0)


def trajectory(track, field, z_to, *, z_from=None, n_samples=400, **kwargs):
    """Sample one track's path as an ``(n, 3)`` array, for plotting.

    Samples are taken at equally spaced ``z`` values; rows the track never
    reached are dropped.
    """
    z_from = track.z if z_from is None else float(z_from)
    planes = np.linspace(z_from, float(z_to), int(n_samples))
    crossings = propagate([track], field, planes, **kwargs)
    reached = crossings.crossed[0]
    return np.stack(
        (
            crossings.x[0][reached],
            crossings.y[0][reached],
            crossings.plane_z[reached],
        ),
        axis=-1,
    )
