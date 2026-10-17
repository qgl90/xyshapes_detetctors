r"""Magnetic field maps expressed in the same millimetre frame as the XY shapes.

Unit convention
---------------
Positions and plane ``z`` values are millimetres, exactly like the Shapely
shapes and the ``z`` column of the layout CSVs. Field components are tesla and
momenta are MeV/c, so a single constant relates the two:

.. math::

    p_T [\\mathrm{MeV}/c] = 0.299792458 \; q [e] \; B [\\mathrm{T}] \;
                            R [\\mathrm{mm}]

:data:`MEV_PER_TESLA_MM` is that constant, and
:func:`MagneticField.bending_radius` uses it directly.

The LHCb map shipped with this repository (``magnetic_field_v5r0.root``) holds
both a ``field`` TNtuple of ``x, y, z, bx, by, bz`` samples on a regular
100 mm grid and three TH3D projections. The TNtuple is the authoritative
source here because it covers the full grid without over/underflow bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


#: MeV/c obtained from one elementary charge crossing 1 T over 1 mm of radius.
MEV_PER_TESLA_MM = 0.299792458


#: ``numpy.trapezoid`` is NumPy 2 only; ``numpy.trapz`` covers NumPy 1.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


class FieldMapError(RuntimeError):
    """Raised when a field source cannot be turned into a regular grid."""


@dataclass(frozen=True)
class Grid:
    """A regular 3D sampling grid, in millimetres."""

    origin: tuple
    spacing: tuple
    shape: tuple

    @property
    def upper(self):
        return tuple(
            o + d * (n - 1)
            for o, d, n in zip(self.origin, self.spacing, self.shape)
        )

    @property
    def bounds(self):
        """``(xmin, ymin, zmin, xmax, ymax, zmax)`` in millimetres."""
        return (*self.origin, *self.upper)

    def axis(self, index):
        return (
            self.origin[index]
            + self.spacing[index] * np.arange(self.shape[index], dtype=float)
        )

    def describe(self):
        return "  ".join(
            f"{label}: {n:d} x {d:g} mm in [{o:g}, {u:g}]"
            for label, n, d, o, u in zip(
                "xyz", self.shape, self.spacing, self.origin, self.upper
            )
        )


class MagneticField:
    """Trilinearly interpolated field map in millimetres and tesla.

    ``values`` has shape ``(nx, ny, nz, 3)`` and holds ``(Bx, By, Bz)`` in
    tesla at every grid node. Queries outside the grid return zero field,
    which is the right behaviour downstream of the LHCb magnet where the map
    ends but the Muon stations continue.

    ``scale`` multiplies every component on read, so ``scale=-1`` flips the
    magnet polarity without touching the stored map, and ``scale=0`` gives a
    field-free geometry for straight-line cross-checks.
    """

    def __init__(self, values, grid, *, scale=1.0, source=None):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 4 or values.shape[3] != 3:
            raise FieldMapError("field values must have shape (nx, ny, nz, 3)")
        if values.shape[:3] != tuple(grid.shape):
            raise FieldMapError("field values do not match the grid shape")
        self.values = values
        self.grid = grid
        self.scale = float(scale)
        self.source = str(source) if source is not None else ""
        self._prepare()

    def _prepare(self):
        # Interpolation works on the flattened node array with explicit
        # strides: eight 1D gathers are measurably cheaper than eight 3D
        # fancy-index lookups, and the propagator makes millions of them.
        shape = self.grid.shape
        self._nodes = self.values.reshape(-1, 3)
        self._strides = np.asarray(
            (shape[1] * shape[2], shape[2], 1), dtype=np.intp
        )
        self._origin = np.asarray(self.grid.origin, dtype=float)
        self._spacing = np.asarray(self.grid.spacing, dtype=float)
        self._last_index = np.asarray(shape, dtype=float) - 1.0

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_root(cls, path, *, tree="field", scale=1.0):
        """Read a field map from a ROOT file holding an ``x, y, z, bx, by, bz``
        ntuple. Requires :mod:`uproot`."""
        try:
            import uproot
        except ImportError as error:  # pragma: no cover - depends on the env
            raise FieldMapError(
                "reading ROOT field maps needs uproot; "
                "install it with `python -m pip install uproot`, or convert "
                "the map once with MagneticField.to_npz"
            ) from error

        path = Path(path)
        with uproot.open(path) as handle:
            columns = handle[tree].arrays(library="np")
        try:
            samples = [columns[name] for name in ("x", "y", "z", "bx", "by", "bz")]
        except KeyError as error:
            raise FieldMapError(
                f"{path}:{tree} must provide x, y, z, bx, by, bz branches"
            ) from error
        return cls.from_samples(*samples, scale=scale, source=path)

    @classmethod
    def from_samples(cls, x, y, z, bx, by, bz, *, scale=1.0, source=None):
        """Build a map from scattered samples that cover a regular grid."""
        positions = [np.asarray(value, dtype=float) for value in (x, y, z)]
        components = [np.asarray(value, dtype=float) for value in (bx, by, bz)]
        sizes = {value.size for value in positions + components}
        if len(sizes) != 1:
            raise FieldMapError("position and field arrays must have equal length")

        origin, spacing, shape, indices = [], [], [], []
        for axis, values in zip("xyz", positions):
            nodes = np.unique(values)
            if nodes.size < 2:
                raise FieldMapError(f"the {axis} axis needs at least two nodes")
            steps = np.diff(nodes)
            step = float(steps[0])
            if not np.allclose(steps, step, rtol=0, atol=1e-6 * abs(step)):
                raise FieldMapError(f"the {axis} axis is not regularly spaced")
            origin.append(float(nodes[0]))
            spacing.append(step)
            shape.append(int(nodes.size))
            indices.append(np.rint((values - nodes[0]) / step).astype(np.intp))

        grid = Grid(tuple(origin), tuple(spacing), tuple(shape))
        values = np.zeros((*grid.shape, 3), dtype=np.float32)
        filled = np.zeros(grid.shape, dtype=bool)
        cells = tuple(indices)
        for component, field in enumerate(components):
            values[(*cells, component)] = field
        filled[cells] = True
        if not filled.all():
            missing = int(filled.size - filled.sum())
            raise FieldMapError(
                f"the samples leave {missing} of {filled.size} grid nodes empty"
            )
        return cls(values, grid, scale=scale, source=source)

    @classmethod
    def from_npz(cls, path, *, scale=1.0):
        """Load a map previously written by :meth:`to_npz`."""
        path = Path(path)
        with np.load(path) as payload:
            grid = Grid(
                tuple(payload["origin"].tolist()),
                tuple(payload["spacing"].tolist()),
                tuple(int(value) for value in payload["shape"]),
            )
            values = payload["values"]
            source = str(payload["source"]) if "source" in payload else path
        return cls(values, grid, scale=scale, source=source)

    def to_npz(self, path):
        """Write the map as a compressed ``.npz``, for hosts without uproot."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            values=self.values,
            origin=np.asarray(self.grid.origin, dtype=float),
            spacing=np.asarray(self.grid.spacing, dtype=float),
            shape=np.asarray(self.grid.shape, dtype=np.int64),
            source=np.asarray(self.source),
        )
        return path

    @classmethod
    def load(cls, path, **kwargs):
        """Load from ``.root`` or ``.npz``, picking the reader by suffix."""
        path = Path(path)
        if path.suffix == ".npz":
            return cls.from_npz(path, **kwargs)
        return cls.from_root(path, **kwargs)

    # -------------------------------------------------------------- queries

    @property
    def bounds(self):
        return self.grid.bounds

    def scaled(self, factor):
        """Return the same map with ``scale`` multiplied by ``factor``."""
        return MagneticField(
            self.values, self.grid, scale=self.scale * factor, source=self.source
        )

    def __call__(self, x, y, z):
        return self.b_at(x, y, z)

    def b_at(self, x, y, z):
        """Interpolated ``(Bx, By, Bz)`` in tesla at millimetre coordinates.

        Scalars give a shape ``(3,)`` result; arrays broadcast to ``(..., 3)``.
        """
        x, y, z = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        points = np.stack((x, y, z), axis=-1)
        return self.b_at_points(points)

    def b_at_points(self, points):
        """Interpolated field at ``(..., 3)`` millimetre positions."""
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("points must have a trailing axis of length three")
        flat = points.reshape(-1, 3)

        fractional = (flat - self._origin) / self._spacing
        inside = np.all(
            (fractional >= 0.0) & (fractional <= self._last_index), axis=1
        )

        lower = np.floor(fractional)
        np.clip(lower, 0.0, self._last_index - 1.0, out=lower)
        weight = fractional - lower
        np.clip(weight, 0.0, 1.0, out=weight)

        sx, sy, sz = self._strides
        base = lower.astype(np.intp) @ self._strides
        u, v, w = weight[:, 0:1], weight[:, 1:2], weight[:, 2:3]
        un, vn, wn = 1.0 - u, 1.0 - v, 1.0 - w
        nodes = self._nodes

        result = nodes[base] * (un * vn * wn)
        result += nodes[base + sz] * (un * vn * w)
        result += nodes[base + sy] * (un * v * wn)
        result += nodes[base + sy + sz] * (un * v * w)
        result += nodes[base + sx] * (u * vn * wn)
        result += nodes[base + sx + sz] * (u * vn * w)
        result += nodes[base + sx + sy] * (u * v * wn)
        result += nodes[base + sx + sy + sz] * (u * v * w)

        result *= self.scale
        result[~inside] = 0.0
        return result.reshape(points.shape)

    def integral_bdl(self, z_from, z_to, *, x=0.0, y=0.0, samples=2001):
        """``∫B dz`` in T·mm along a line of constant ``x`` and ``y``.

        The on-axis ``By`` integral is the usual summary number for the LHCb
        dipole; divide by 1000 for T·m.
        """
        z = np.linspace(float(z_from), float(z_to), int(samples))
        field = self.b_at(np.full_like(z, x), np.full_like(z, y), z)
        return _trapezoid(field, z, axis=0)

    @staticmethod
    def bending_radius(p, b, charge=1.0):
        """Radius in mm for momentum ``p`` (MeV/c) transverse to ``b`` (T)."""
        denominator = MEV_PER_TESLA_MM * abs(charge) * np.abs(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator > 0, p / denominator, np.inf)

    def __repr__(self):
        return (
            f"MagneticField(grid=[{self.grid.describe()}], "
            f"scale={self.scale:g}, source={self.source!r})"
        )


class UniformField(MagneticField):
    """A constant field, handy for analytic cross-checks of the propagator."""

    def __init__(self, bx=0.0, by=0.0, bz=0.0):
        self._b = np.asarray((bx, by, bz), dtype=float)
        grid = Grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 2))
        values = np.broadcast_to(self._b.astype(np.float32), (2, 2, 2, 3))
        super().__init__(values.copy(), grid, source="uniform")

    @property
    def bounds(self):
        return (-np.inf,) * 3 + (np.inf,) * 3

    def b_at_points(self, points):
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("points must have a trailing axis of length three")
        return np.broadcast_to(
            self._b * self.scale, points.shape
        ).copy()

    def scaled(self, factor):
        field = UniformField(*(self._b * self.scale * factor))
        return field

    def __repr__(self):
        return f"UniformField(bx={self._b[0]:g}, by={self._b[1]:g}, bz={self._b[2]:g})"


def zero_field():
    """A field-free map, so tracks travel in straight lines."""
    return UniformField()


def main():
    """Inspect a field map, or convert one to ``.npz`` for hosts without uproot."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="field map, either .root or .npz")
    parser.add_argument("--tree", default="field", help="ntuple name in a ROOT file")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="polarity factor applied on read")
    parser.add_argument("--to-npz", metavar="PATH",
                        help="write the map as a compressed .npz")
    parser.add_argument("--scan", nargs=2, type=float, metavar=("X", "Y"),
                        default=(0.0, 0.0),
                        help="x and y in mm for the printed z scan")
    parser.add_argument("--scan-step", type=float, default=1000.0,
                        help="z spacing of the printed scan in mm")
    args = parser.parse_args()

    path = Path(args.path)
    if path.suffix == ".npz":
        field = MagneticField.from_npz(path, scale=args.scale)
    else:
        field = MagneticField.from_root(path, tree=args.tree, scale=args.scale)
    print(field)

    x, y = args.scan
    z_low, z_high = field.bounds[2], field.bounds[5]
    integral = field.integral_bdl(z_low, z_high, x=x, y=y)
    print(f"integral B dl at x={x:g} mm, y={y:g} mm: "
          f"Bx={integral[0]:.1f}  By={integral[1]:.1f}  Bz={integral[2]:.1f} T mm")
    print(f"{'z [mm]':>10} {'Bx [T]':>10} {'By [T]':>10} {'Bz [T]':>10}")
    for z in np.arange(z_low, z_high + 0.5 * args.scan_step, args.scan_step):
        bx, by, bz = field.b_at(x, y, float(z))
        print(f"{z:10.0f} {bx:10.4f} {by:10.4f} {bz:10.4f}")

    if args.to_npz:
        print(field.to_npz(args.to_npz))


if __name__ == "__main__":
    main()
