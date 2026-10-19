"""High-level transport from a named detector setup to fired-layer labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from .configuration import LoadedMaskConfiguration, load_mask_configuration
from .magnetic_field import MagneticField
from .propagation import PlaneCrossings, propagate
from .trajectory import TrackState, Trajectory, TrajectoryError, create_trajectory
from .selection import MTSelection


@dataclass(frozen=True)
class TrackTransportResult:
    """One transported state, its path, and detector acceptance properties."""

    configuration: str
    state: TrackState
    trajectory: Trajectory
    layer_hits: dict[str, bool]
    detector_counts: dict[str, int]
    labels: dict[str, object]

    def as_record(self):
        """Return only acceptance fields, ready to update an existing track row."""
        return dict(self.labels)


@dataclass(frozen=True)
class BatchTransportResult:
    """Reusable physical-plane crossings and mask columns for many tracks."""

    configuration: str
    crossings: PlaneCrossings
    labels: dict

    def as_columns(self):
        return dict(self.labels)


class DetectorTransport:
    """Connect one loaded detector mask configuration to one field service."""

    def __init__(self, configuration, field, *, step_mm=10.0, mt_selection=None):
        if not isinstance(configuration, LoadedMaskConfiguration):
            raise TypeError("configuration must be a LoadedMaskConfiguration")
        if not callable(getattr(field, "b_at", None)):
            raise TypeError("field must provide b_at(x, y, z)")
        self.configuration = configuration
        self.field = field
        self.mt_selection = MTSelection() if mt_selection is None else mt_selection
        if not isinstance(self.mt_selection, MTSelection):
            raise TypeError("mt_selection must be an MTSelection")
        self.step_mm = float(step_mm)
        if not np.isfinite(self.step_mm) or self.step_mm <= 0:
            raise ValueError("step_mm must be finite and positive")

    @classmethod
    def from_files(
        cls,
        configuration,
        field_path,
        *,
        polarity=1,
        scale=1.0,
        exact_surfaces=True,
        step_mm=10.0,
        catalog=None,
        mt_selection=None,
    ):
        """Load a named setup and ROOT/NPZ field map as one transport service."""
        masks = load_mask_configuration(
            configuration, catalog=catalog, exact_surfaces=exact_surfaces
        )
        field = MagneticField.load(field_path, polarity=polarity, scale=scale)
        return cls(masks, field, step_mm=step_mm, mt_selection=mt_selection)

    def propagate(self, state, *, z_end=None, label_suffix="", min_muon_stations=3):
        """Transport one state through every mask surface and return its labels.

        ``state`` is a :class:`TrackState` or ``(x, y, tx, ty, q/p)`` at a
        specified z. Masks upstream of the initial state are not crossed.
        A curling/non-forward track raises :class:`TrajectoryError`.
        """
        if not isinstance(state, TrackState):
            state = TrackState.from_values(state)
        model = self.configuration.model
        planes = model.plane_z
        if planes.size == 0:
            raise ValueError("the loaded detector setup has no surfaces")
        final_z = float(planes[-1] if z_end is None else z_end)
        path = create_trajectory(
            self.field,
            state,
            final_z,
            sensor_z=planes[planes >= state.z],
            step_mm=self.step_mm,
        )
        fired = model.fired_trajectory(path)[0]
        counts = {
            detector: int(values[0])
            for detector, values in model.counts(fired=fired[None, :]).items()
        }
        hits = {
            f"{station.detector}_{station.name}": bool(fired[index])
            for index, station in enumerate(model.stations)
        }
        labels = self._labels(
            hits, counts, min_muon_stations=min_muon_stations,
            suffix=label_suffix,
            mt_selection=self.mt_selection,
        )
        return TrackTransportResult(
            self.configuration.id, state, path, hits, counts, labels
        )

    def propagate_batch(self, states, *, label_suffix="", min_muon_stations=3,
                        max_steps=200000):
        """Vectorized RK4 transport and labels for a batch of independent tracks.

        Accept a sequence of TrackState objects or a mapping of broadcastable
        x, y, z, tx, ty, q_over_p arrays. This uses the established path-length
        RK4 integrator, with self.step_mm in the field and 100 mm outside it.
        Unlike propagate(), it returns plane crossings without storing a dense
        spline per track. Reuse those crossings to compare identical plane-z
        inventories using labels_from_crossings(). Incomplete transport raises.
        """
        crossings, initial_z = self._propagate_to_planes(
            states, self.configuration.model.plane_z, max_steps=max_steps,
        )
        labels = self.labels_from_crossings(
            crossings, label_suffix=label_suffix,
            min_muon_stations=min_muon_stations, initial_z=initial_z,
        )
        return BatchTransportResult(self.configuration.id, crossings, labels)

    def _propagate_to_planes(self, states, planes, *, max_steps=200000):
        if isinstance(states, Mapping):
            names = ("x", "y", "z", "tx", "ty", "q_over_p")
            columns = np.broadcast_arrays(*[
                np.asarray(states[name], dtype=float) for name in names
            ])
            values = np.column_stack([column.ravel() for column in columns])
        else:
            values = np.asarray([
                (state.x, state.y, state.z, state.tx, state.ty, state.q_over_p)
                for state in states
            ], dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("batch states must be nonempty and finite")
        if len(planes) == 0:
            raise ValueError("the loaded detector setup has no surfaces")
        q_over_p = values[:, 5]
        charge = np.sign(q_over_p)
        momentum = np.ones_like(q_over_p)
        np.divide(1.0, np.abs(q_over_p), out=momentum, where=q_over_p != 0)
        direction = np.column_stack((values[:, 3:5], np.ones(len(values))))
        crossings = propagate(
            (charge, momentum, values[:, :3], direction), self.field, planes,
            step=self.step_mm, max_steps=max_steps,
        )
        return crossings, values[:, 2]

    def labels_from_crossings(self, crossings, *, label_suffix="", min_muon_stations=3,
                              initial_z=None):
        """Apply this geometry to an already transported batch, without field work.

        Physical plane-z arrays must match exactly; model.fired validates them.
        A failed or incomplete trajectory is never silently labelled as a miss.
        """
        expected = np.ones_like(crossings.crossed)
        if initial_z is not None:
            origins = np.broadcast_to(np.asarray(initial_z, dtype=float), (crossings.n_tracks,))
            if not np.all(np.isfinite(origins)):
                raise ValueError("initial_z must be finite")
            expected = crossings.plane_z[None, :] >= origins[:, None]
        failed = (crossings.stop_reason != "completed") | np.any(expected & ~crossings.crossed, axis=1)
        if failed.any():
            rows = np.flatnonzero(failed)
            raise TrajectoryError(
                f"{len(rows)} batch tracks failed transport or missed requested planes; "
                f"rows {rows[:10].tolist()}, stop reasons {crossings.stop_reason[rows[:10]].tolist()}"
            )
        model = self.configuration.model
        fired = model.fired(replace(crossings, crossed=crossings.crossed & expected))
        counts = {detector: count.astype(int)
                  for detector, count in model.counts(fired=fired).items()}
        hits = {f"{station.detector}_{station.name}": fired[:, index]
                for index, station in enumerate(model.stations)}
        return self._labels(hits, counts, min_muon_stations=min_muon_stations,
                            suffix=label_suffix, mt_selection=self.mt_selection)

    @staticmethod
    def _labels(hits, counts, *, min_muon_stations, suffix, mt_selection=None):
        def put(name, value):
            if isinstance(value, np.generic):
                value = value.item()
            labels[f"{name}_{suffix}" if suffix else name] = value

        labels = {}
        for detector, count in counts.items():
            put(f"n{detector}", count)
            put(f"has{detector}", count > 0)
        if "TV" in counts:
            put("hasTV_3Hits", counts["TV"] >= 3)
            put("hasTV_4Hits", counts["TV"] >= 4)
        if "MP" in counts and "FT" in counts:
            selection = MTSelection() if mt_selection is None else mt_selection
            put("hasMT", selection.evaluate(counts["MP"], counts["FT"]))

        face_hits = {}
        for station_key, fired in hits.items():
            detector, station = station_key.split("_", 1)
            put(f"fired_{detector}_{station}", fired)
            for face in ("entry", "exit"):
                if station.lower().endswith(face):
                    face_hits.setdefault(detector, {})[face] = fired

        for detector, faces in face_hits.items():
            if set(faces) == {"entry", "exit"}:
                entry, exit_ = faces["entry"], faces["exit"]
                put(f"{detector}_entry", entry)
                put(f"{detector}_exit", exit_)
                put(f"{detector}_entry_exit", entry & exit_)
                put(f"{detector}_entry_only", entry & np.logical_not(exit_))

        picocal = next(
            (detector for detector in counts if detector.lower() == "picocal"), None
        )
        if picocal is not None:
            faces = face_hits.get(picocal, {})
            put(
                "inPicoCal",
                faces["entry"] & faces["exit"]
                if set(faces) == {"entry", "exit"}
                else counts[picocal] > 0,
            )
            put("hasPicoCal", counts[picocal] > 0)

        muon = next(
            (detector for detector in counts if detector.upper().startswith("MUON")),
            None,
        )
        if muon is not None:
            put("nMuon", counts[muon])
            put("hasMuon", counts[muon] > 0)
            put("isMuon", counts[muon] >= int(min_muon_stations))

        for detector, label in (("RICH1", "Rich1"), ("RICH2", "Rich2")):
            faces = face_hits.get(detector, {})
            for face in ("entry", "exit"):
                if face in faces:
                    put(f"has{label}{face.title()}", faces[face])

        for face in ("entry", "exit"):
            rich_hits = [
                faces[face]
                for detector, faces in face_hits.items()
                if detector in ("RICH1", "RICH2") and face in faces
            ]
            if rich_hits:
                put(f"hasRich{face.title()}", np.logical_or.reduce(rich_hits))
        return labels
