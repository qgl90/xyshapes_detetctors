"""Turn propagated tracks into per-detector acceptance counts.

The layout CSVs describe one Shapely XY shape per extracted surface, so a
single logical layer can appear many times: each Target Velo module has
different sensors at two centre-Z positions 0.992 mm apart, and one Mighty Pixel station
is 120 staggered tiles spread over 5 mm in ``z``.  Counting raw surfaces would
therefore be meaningless.

This module groups surfaces into **stations**: planes of the same detector
whose ``z`` values sit within a tolerance are merged into one Shapely union
evaluated at their mean ``z``.  A track *fires* a station when its crossing
point falls inside that union, and the per-detector counts are then the
numbers the acceptance studies ask for.

Grouping can also be given explicitly with a ``station`` column in the layout
CSV, which always takes precedence over the ``z`` tolerance.

A station is fired by testing the crossing at its mean ``z`` against the union
of its surfaces.  That is a small approximation where a station is thick: on a
flat-in-slope sample of the baseline setup it agrees exactly for MP, FT, RICH1,
RICH2, PicoCal and the Muon stations, and shifts the mean TV and UP counts by
0.2% and 2% against evaluating every surface at its own ``z``.  Pass
``exact_surfaces=True`` to do the latter instead, at the cost of propagating
through every surface plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import shapely
from shapely.ops import unary_union

from .layout import load_layout
from .propagation import Track, propagate


#: Default ``z`` tolerance in mm for merging surfaces into one station.
DEFAULT_TOLERANCE = 10.0

#: Per-detector overrides.  Consecutive Target Velo sensors are only ~8 mm
#: apart, so they need a tighter window than the 7 mm-thick UP layers.
DETECTOR_TOLERANCE = {"TV": 2.0}

#: Detectors whose layer names mark the entry and exit faces of a volume.
ENTRY_EXIT_KEYS = ("entry", "exit")


def _add_canonical_flags(columns, suffix=""):
    """Add stable output aliases used by geometry-comparison workflows."""
    def value(name):
        key = f"{name}_{suffix}" if suffix else name
        return columns.get(key)

    aliases = {}
    tv_count = value("nTV")
    if tv_count is not None:
        aliases["hasTV_3Hits"] = tv_count >= 3
        aliases["hasTV_4Hits"] = tv_count >= 4
    muon_count = value("nMUON")
    if muon_count is not None:
        aliases["nMuon"] = muon_count
        aliases["hasMuon"] = muon_count > 0

    for detector, label in (("RICH1", "Rich1"), ("RICH2", "Rich2")):
        for face in ENTRY_EXIT_KEYS:
            face_hit = value(f"{detector}_{face}")
            if face_hit is not None:
                aliases[f"has{label}{face.title()}"] = face_hit

    entries = [value(f"{detector}_entry") for detector in ("RICH1", "RICH2")]
    exits = [value(f"{detector}_exit") for detector in ("RICH1", "RICH2")]
    entries = [item for item in entries if item is not None]
    exits = [item for item in exits if item is not None]
    if entries:
        aliases["hasRichEntry"] = np.logical_or.reduce(entries)
    if exits:
        aliases["hasRichExit"] = np.logical_or.reduce(exits)

    picocal_count = value("nPicoCal")
    if picocal_count is not None:
        aliases["hasPicoCal"] = picocal_count > 0

    for name, array in aliases.items():
        columns[f"{name}_{suffix}" if suffix else name] = array


@dataclass(frozen=True)
class Station:
    """One counting unit: the merged acceptance of nearby surfaces."""

    detector: str
    name: str
    z: float
    z_min: float
    z_max: float
    geometry: object
    layers: tuple
    surfaces: tuple = ()

    @property
    def n_surfaces(self):
        return len(self.layers)


def _station_key(detector, rows):
    """A readable station name from its member layer names."""
    names = [row.name for row in rows]
    if len(names) == 1:
        return names[0]
    for key in ENTRY_EXIT_KEYS:
        if all(key in name for name in names):
            return f"{detector}_{key}"
    return f"{detector}_z{np.mean([row.z for row in rows]):.1f}"


def group_stations(layers, *, tolerance=None, detector_tolerance=None,
                   combine="union", explicit=None):
    """Merge detector layers into stations.

    ``layers`` are :class:`~xyshapes_detectors.layout.DetectorLayer` objects.
    ``explicit`` optionally maps a layer name to a station label, which
    overrides the ``z`` clustering.  ``combine`` is ``"union"`` so that a track
    fires a station by crossing any of its surfaces, or ``"intersection"`` to
    require the overlap of all of them.
    """
    if combine not in {"union", "intersection"}:
        raise ValueError("combine must be 'union' or 'intersection'")
    tolerance = DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    overrides = dict(DETECTOR_TOLERANCE)
    overrides.update(detector_tolerance or {})
    explicit = explicit or {}

    by_detector = {}
    for layer in layers:
        by_detector.setdefault(layer.detector, []).append(layer)

    stations = []
    for detector, detector_layers in by_detector.items():
        window = overrides.get(detector, tolerance)
        groups = []
        if any(layer.name in explicit for layer in detector_layers):
            labelled = {}
            for layer in sorted(detector_layers, key=lambda item: item.z):
                label = explicit.get(layer.name, layer.name)
                labelled.setdefault(label, []).append(layer)
            groups = [(label, group) for label, group in labelled.items()]
        else:
            plain = []
            for layer in sorted(detector_layers, key=lambda item: item.z):
                if plain and layer.z - plain[-1][-1].z <= window:
                    plain[-1].append(layer)
                else:
                    plain.append([layer])
            groups = [(None, group) for group in plain]

        for label, group in groups:
            shapes = [layer.geometry for layer in group]
            if len(shapes) == 1:
                geometry = shapes[0]
            elif combine == "union":
                geometry = unary_union(shapes)
            else:
                geometry = shapes[0]
                for shape in shapes[1:]:
                    geometry = geometry.intersection(shape)
            z_values = [layer.z for layer in group]
            stations.append(Station(
                detector=detector,
                name=label or _station_key(detector, group),
                z=float(np.mean(z_values)),
                z_min=float(min(z_values)),
                z_max=float(max(z_values)),
                geometry=geometry,
                layers=tuple(layer.name for layer in group),
                surfaces=tuple((layer.z, layer.geometry) for layer in group),
            ))
    return sorted(stations, key=lambda station: station.z)


def _read_explicit_stations(path):
    """Read the optional ``station`` column of a layout CSV."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "station" not in reader.fieldnames:
            return {}
        return {
            row["layer"]: row["station"]
            for row in reader
            if row.get("station")
            and row.get("Use", "1").strip() not in {"0", "false", "False"}
        }


class AcceptanceModel:
    """Stations of an XY-surface setup, ready to be fired by tracks."""

    def __init__(self, stations, *, source="", exact_surfaces=False):
        self.stations = tuple(stations)
        if not self.stations:
            raise ValueError("an acceptance model needs at least one station")
        self.source = str(source)
        self.exact_surfaces = bool(exact_surfaces)
        self.detectors = tuple(dict.fromkeys(
            station.detector for station in self.stations
        ))

        if self.exact_surfaces:
            if any(not station.surfaces for station in self.stations):
                raise ValueError(
                    "exact_surfaces needs stations that kept their member "
                    "surfaces; rebuild the model with group_stations"
                )
            self.plane_z = np.array(sorted({
                z for station in self.stations for z, _ in station.surfaces
            }))
            column_of = {z: index for index, z in enumerate(self.plane_z)}
            self._members = [
                [(column_of[z], geometry) for z, geometry in station.surfaces]
                for station in self.stations
            ]
        else:
            self.plane_z = np.array([station.z for station in self.stations])
            self._members = [
                [(index, station.geometry)]
                for index, station in enumerate(self.stations)
            ]

        for members in self._members:
            for _, geometry in members:
                shapely.prepare(geometry)

    # ----------------------------------------------------------- factories

    @classmethod
    def from_layers(cls, layers, *, source="", exact_surfaces=False, **kwargs):
        kwargs.setdefault("explicit", {
            layer.name: getattr(layer, "station", None)
            for layer in layers if getattr(layer, "station", None)
        })
        return cls(
            group_stations(layers, **kwargs),
            source=source,
            exact_surfaces=exact_surfaces,
        )

    @classmethod
    def from_layout(cls, path, *, shape_dirs=None, up_dir=None,
                    shape_registry=None, **kwargs):
        """Build from a layout CSV and the pickle directories it references."""
        path = Path(path)
        if up_dir is None:
            up_dir = path.parent / "__no_up_fallback__"
        layers = load_layout(
            path,
            shape_registry=shape_registry,
            up_dir=up_dir,
            shape_dirs=shape_dirs,
        )
        kwargs.setdefault("explicit", _read_explicit_stations(path))
        return cls.from_layers(layers, source=str(path), **kwargs)

    @classmethod
    def from_setup(cls, directory, **kwargs):
        """Build from a setup written by ``examples/build_lhcb_xy_setup.py``."""
        directory = Path(directory)
        return cls.from_layout(
            directory / "layout.csv",
            shape_dirs=[directory / "shapes"],
            **kwargs,
        )

    @classmethod
    def load(cls, source, **kwargs):
        """Build from either a setup directory or a layout CSV path.

        A bare layout CSV resolves its pickles from the directory holding it,
        unless ``shape_dirs`` says otherwise.
        """
        source = Path(source)
        if source.is_dir():
            return cls.from_setup(source, **kwargs)
        kwargs.setdefault("shape_dirs", [source.parent])
        return cls.from_layout(source, **kwargs)

    # ------------------------------------------------------------- queries

    def __len__(self):
        return len(self.stations)

    def stations_of(self, detector):
        return tuple(s for s in self.stations if s.detector == detector)

    def indices_of(self, detector):
        return np.array(
            [i for i, s in enumerate(self.stations) if s.detector == detector],
            dtype=int,
        )

    def fired(self, crossings):
        """``(n_tracks, n_stations)`` boolean array of fired stations.

        ``crossings`` must come from :func:`~xyshapes_detectors.propagation.propagate`
        called with :attr:`plane_z`.
        """
        if not np.array_equal(crossings.plane_z, self.plane_z):
            raise ValueError(
                "the crossings were made for different planes; propagate with "
                "AcceptanceModel.plane_z"
            )
        hits = np.zeros((crossings.n_tracks, len(self.stations)), dtype=bool)
        for station, members in enumerate(self._members):
            for column, geometry in members:
                reached = crossings.crossed[:, column]
                if not reached.any():
                    continue
                inside = shapely.contains_xy(
                    geometry,
                    crossings.x[reached, column],
                    crossings.y[reached, column],
                )
                hits[reached, station] |= inside
        return hits

    def fired_trajectory(self, trajectory):
        """Return logical-layer hits for a Z-integrated trajectory.

        The trajectory must reach the last surface in this model. Each
        surface is tested at its own z, including surfaces grouped into one
        logical layer; this avoids replacing a thick layer with a mean-z mask.
        """
        if self.plane_z.size == 0:
            return np.zeros((1, 0), dtype=bool)
        if self.plane_z[-1] > trajectory.z[-1]:
            raise ValueError(
                "trajectory must span all acceptance surfaces: "
                f"[{self.plane_z[0]:g}, {self.plane_z[-1]:g}] mm"
            )
        downstream = self.plane_z >= trajectory.z[0]
        xy = np.full((len(self.plane_z), 2), np.nan)
        xy[downstream] = trajectory.intercepts(self.plane_z[downstream])
        hits = np.zeros((1, len(self.stations)), dtype=bool)
        for station, members in enumerate(self._members):
            for column, geometry in members:
                if not downstream[column]:
                    continue
                hits[0, station] |= shapely.contains_xy(
                    geometry, xy[column, 0], xy[column, 1]
                )
        return hits

    def counts(self, crossings=None, *, fired=None):
        """``{detector: counts}`` with one integer per track."""
        if fired is None:
            fired = self.fired(crossings)
        return {
            detector: fired[:, self.indices_of(detector)].sum(axis=1)
            for detector in self.detectors
        }

    # ------------------------------------------------------------ evaluate

    def evaluate(self, tracks, field, *, min_muon_stations=3,
                 station_flags=False, kinematics=True, label_suffix="",
                 **propagate_kwargs):
        """Propagate ``tracks`` and return the acceptance columns.

        The result is a dictionary of equal-length arrays: the track
        kinematics, the propagation outcome, ``n<DETECTOR>`` station counts,
        ``has<DETECTOR>`` flags, the entry/exit flags of the RICH and PicoCal
        volumes, ``inPicoCal``, and ``isMuon`` for at least
        ``min_muon_stations`` fired Muon stations.
        """
        tracks = [tracks] if isinstance(tracks, Track) else list(tracks)
        crossings = propagate(tracks, field, self.plane_z, **propagate_kwargs)
        fired = self.fired(crossings)
        columns = {}

        if kinematics:
            rows = [track.as_row() for track in tracks]
            for name in rows[0]:
                columns[name] = np.array([row[name] for row in rows], dtype=float)
            charge = columns["charge"]
            if np.all(charge == np.rint(charge)):
                columns["charge"] = charge.astype(int)

        columns["z_exit"] = crossings.z_exit
        columns["s_exit"] = crossings.s_exit
        columns["stop_reason"] = crossings.stop_reason

        for detector, count in self.counts(fired=fired).items():
            columns[f"n{detector}"] = count.astype(int)
            columns[f"has{detector}"] = count > 0
            if label_suffix:
                columns[f"n{detector}_{label_suffix}"] = columns[f"n{detector}"]
                columns[f"has{detector}_{label_suffix}"] = columns[f"has{detector}"]

        for detector in self.detectors:
            faces = {}
            for index in self.indices_of(detector):
                name = self.stations[index].name.lower()
                for key in ENTRY_EXIT_KEYS:
                    if name.endswith(key):
                        faces[key] = fired[:, index]
            if len(faces) != len(ENTRY_EXIT_KEYS):
                continue
            entry, exit_ = faces["entry"], faces["exit"]
            columns[f"{detector}_entry"] = entry
            columns[f"{detector}_exit"] = exit_
            columns[f"{detector}_entry_exit"] = entry & exit_
            columns[f"{detector}_entry_only"] = entry & ~exit_
            if label_suffix:
                columns[f"{detector}_entry_{label_suffix}"] = entry
                columns[f"{detector}_exit_{label_suffix}"] = exit_
                columns[f"{detector}_entry_exit_{label_suffix}"] = entry & exit_
                columns[f"{detector}_entry_only_{label_suffix}"] = entry & ~exit_

        if "PicoCal_entry_exit" in columns:
            columns["inPicoCal"] = columns["PicoCal_entry_exit"]
        elif "nPicoCal" in columns:
            columns["inPicoCal"] = columns["nPicoCal"] > 0

        muon = next(
            (name for name in self.detectors if name.upper().startswith("MUON")),
            None,
        )
        if muon is not None:
            columns["isMuon"] = columns[f"n{muon}"] >= int(min_muon_stations)

        if station_flags:
            for index, station in enumerate(self.stations):
                columns[f"fired_{station.detector}_{station.name}"] = fired[:, index]
                if label_suffix:
                    columns[f"fired_{station.detector}_{station.name}_{label_suffix}"] = fired[:, index]

        _add_canonical_flags(columns)
        if label_suffix:
            _add_canonical_flags(columns, label_suffix)
        return columns

    def augment(self, table, field, **kwargs):
        """Add the acceptance columns to an existing ``{name: array}`` table.

        ``table`` must carry ``charge``, ``p`` and either ``tx``/``ty`` slopes
        or ``px``/``py``/``pz`` components; ``x0``/``y0``/``z0`` default to the
        origin.  The input columns are kept unchanged.
        """
        tracks = tracks_from_table(table)
        columns = self.evaluate(tracks, field, kinematics=False, **kwargs)
        merged = {name: np.asarray(values) for name, values in table.items()}
        merged.update(columns)
        return merged

    def label_crossings(self, crossing_table, *, station_flags=False,
                        label_suffix="Run5"):
        """Label a table that already contains transported x/y at station z.

        The table must contain one pair of columns per station named
        ``x_<detector>_<station>`` and ``y_<detector>_<station>``.  This is a
        lightweight bridge for external transport code that does not use this
        package's propagator but wants the same acceptance masks.
        """
        columns = {name: np.asarray(values) for name, values in crossing_table.items()}
        if not self.stations:
            return {}
        n_rows = len(next(iter(columns.values())))
        fired = np.zeros((n_rows, len(self.stations)), dtype=bool)
        for index, station in enumerate(self.stations):
            base = f"{station.detector}_{station.name}"
            try:
                x = columns[f"x_{base}"]
                y = columns[f"y_{base}"]
            except KeyError as error:
                raise KeyError(
                    f"crossing table needs x_{base!s} and y_{base!s} columns"
                ) from error
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                fired[finite, index] = shapely.contains_xy(station.geometry, x[finite], y[finite])
        result = {}
        for detector, count in self.counts(fired=fired).items():
            result[f"n{detector}_{label_suffix}"] = count.astype(int)
            result[f"has{detector}_{label_suffix}"] = count > 0
        for detector in self.detectors:
            faces = {}
            for index in self.indices_of(detector):
                name = self.stations[index].name.lower()
                for key in ENTRY_EXIT_KEYS:
                    if name.endswith(key):
                        faces[key] = fired[:, index]
            if len(faces) == len(ENTRY_EXIT_KEYS):
                entry, exit_ = faces["entry"], faces["exit"]
                result[f"{detector}_entry_{label_suffix}"] = entry
                result[f"{detector}_exit_{label_suffix}"] = exit_
                result[f"{detector}_entry_exit_{label_suffix}"] = entry & exit_
                result[f"{detector}_entry_only_{label_suffix}"] = entry & ~exit_
        if station_flags:
            for index, station in enumerate(self.stations):
                result[f"fired_{station.detector}_{station.name}_{label_suffix}"] = fired[:, index]
        _add_canonical_flags(result, label_suffix)
        return result

    def label_surface_crossings(self, crossing_table, *, station_flags=False,
                                label_suffix="Run5"):
        """Label externally projected XY coordinates at every source surface.

        Input columns are named ``x_<detector>_<plane>`` and
        ``y_<detector>_<plane>``, where plane names come from
        :meth:`LoadedMaskConfiguration.plane_inventory` / the layout's
        ``layer`` column. A logical station fires if any member surface is hit.
        """
        columns = {name: np.asarray(values) for name, values in crossing_table.items()}
        if not self.stations:
            return {}
        n_rows = len(next(iter(columns.values())))
        fired = np.zeros((n_rows, len(self.stations)), dtype=bool)
        for station_index, station in enumerate(self.stations):
            for surface_index, (z, geometry) in enumerate(station.surfaces):
                plane = station.layers[surface_index]
                base = f"{station.detector}_{plane}"
                try:
                    x = columns[f"x_{base}"]
                    y = columns[f"y_{base}"]
                except KeyError as error:
                    raise KeyError(
                        f"crossing table needs x_{base} and y_{base} columns "
                        f"for z={z:g} mm"
                    ) from error
                finite = np.isfinite(x) & np.isfinite(y)
                if finite.any():
                    fired[finite, station_index] |= shapely.contains_xy(
                        geometry, x[finite], y[finite]
                    )

        result = {}
        for detector, count in self.counts(fired=fired).items():
            result[f"n{detector}_{label_suffix}"] = count.astype(int)
            result[f"has{detector}_{label_suffix}"] = count > 0
        for detector in self.detectors:
            faces = {}
            for index in self.indices_of(detector):
                name = self.stations[index].name.lower()
                for key in ENTRY_EXIT_KEYS:
                    if name.endswith(key):
                        faces[key] = fired[:, index]
            if len(faces) == len(ENTRY_EXIT_KEYS):
                entry, exit_ = faces["entry"], faces["exit"]
                result[f"{detector}_entry_{label_suffix}"] = entry
                result[f"{detector}_exit_{label_suffix}"] = exit_
                result[f"{detector}_entry_exit_{label_suffix}"] = entry & exit_
                result[f"{detector}_entry_only_{label_suffix}"] = entry & ~exit_
        if station_flags:
            for index, station in enumerate(self.stations):
                result[f"fired_{station.detector}_{station.name}_{label_suffix}"] = fired[:, index]
        _add_canonical_flags(result, label_suffix)
        return result

    # ------------------------------------------------------------- display

    def describe(self):
        mode = (
            "each surface at its own z"
            if self.exact_surfaces
            else "merged surfaces at the station mean z"
        )
        lines = [f"AcceptanceModel from {self.source or 'layers'} ({mode})"]
        for detector in self.detectors:
            stations = self.stations_of(detector)
            surfaces = sum(station.n_surfaces for station in stations)
            lines.append(
                f"  {detector:8s} {len(stations):3d} stations "
                f"({surfaces:4d} surfaces)  "
                f"z = [{stations[0].z:9.2f}, {stations[-1].z:9.2f}] mm"
            )
        return "\n".join(lines)

    def station_table(self):
        """One dictionary per station, for writing an inventory CSV."""
        rows = []
        for station in self.stations:
            xmin, ymin, xmax, ymax = station.geometry.bounds
            rows.append({
                "detector": station.detector,
                "station": station.name,
                "z": f"{station.z:.4f}",
                "z_min": f"{station.z_min:.4f}",
                "z_max": f"{station.z_max:.4f}",
                "surfaces": station.n_surfaces,
                "x_min": f"{xmin:.3f}",
                "y_min": f"{ymin:.3f}",
                "x_max": f"{xmax:.3f}",
                "y_max": f"{ymax:.3f}",
            })
        return rows

    def __repr__(self):
        return (
            f"AcceptanceModel({len(self.stations)} stations, "
            f"detectors={self.detectors}, "
            f"exact_surfaces={self.exact_surfaces}, source={self.source!r})"
        )


def tracks_from_table(table):
    """Build :class:`Track` objects from a ``{name: array}`` table."""
    columns = {name: np.asarray(values) for name, values in table.items()}
    for required in ("charge", "p"):
        if required not in columns:
            raise KeyError(f"the track table needs a {required!r} column")
    n_tracks = len(columns["p"])
    zeros = np.zeros(n_tracks)

    def column(name, default=None):
        if name in columns:
            return np.asarray(columns[name], dtype=float)
        if default is None:
            return None
        return default

    origin = (column("x0", zeros), column("y0", zeros), column("z0", zeros))
    if {"px", "py", "pz"} <= columns.keys():
        directions = np.stack(
            [column("px"), column("py"), column("pz")], axis=-1
        )
    elif {"tx", "ty"} <= columns.keys():
        directions = np.stack(
            [column("tx"), column("ty"), np.ones(n_tracks)], axis=-1
        )
    else:
        raise KeyError("the track table needs px/py/pz or tx/ty columns")

    charge = np.asarray(columns["charge"], dtype=float)
    momentum = np.asarray(columns["p"], dtype=float)
    return [
        Track.from_direction(
            charge[i], momentum[i], directions[i],
            (origin[0][i], origin[1][i], origin[2][i]),
        )
        for i in range(n_tracks)
    ]


def write_table(path, columns):
    """Write a ``{name: array}`` table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    rows = len(columns[names[0]])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for index in range(rows):
            writer.writerow([_format(columns[name][index]) for name in names])
    return path


def _format(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return value
