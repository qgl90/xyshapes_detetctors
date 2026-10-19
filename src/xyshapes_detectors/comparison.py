"""Propagate once and annotate tracks with multiple detector configurations."""

from dataclasses import replace
import re

import numpy as np

from .configuration import LoadedMaskConfiguration, load_mask_configuration
from .magnetic_field import MagneticField
from .transport import DetectorTransport


DEFAULT_COMPARISON = {
    f"up_{up}_ft_{ft}": f"lhcb_lower_up_{up}_ft_{ft}"
    for up in ("gdml", "y8", "y8x6")
    for ft in ("8_8_10", "10_10_12")
}


class GeometryComparison:
    """Reuse one field propagation for all requested geometric acceptances.

    configurations maps an output suffix to a registered ID, setup directory
    or LoadedMaskConfiguration. The default is the three UP x two FT setups.
    """

    def __init__(self, field, configurations=None, *, step_mm=10.0, catalog=None):
        choices = DEFAULT_COMPARISON if configurations is None else configurations
        if not choices:
            raise ValueError("at least one geometry configuration is required")
        self.configurations = {}
        self.services = {}
        for label, identifier in choices.items():
            if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label):
                raise ValueError("geometry labels must start with a letter and contain only letters, digits, underscores")
            config = identifier if isinstance(identifier, LoadedMaskConfiguration) else load_mask_configuration(
                identifier, catalog=catalog, exact_surfaces=True,
            )
            if not config.model.exact_surfaces:
                raise ValueError("geometry comparison requires exact_surfaces=True")
            self.configurations[label] = config
            self.services[label] = DetectorTransport(config, field, step_mm=step_mm)
        self.plane_z = np.unique(np.concatenate([c.model.plane_z for c in self.configurations.values()]))

    @classmethod
    def from_files(cls, field_path, configurations=None, *, polarity=1, scale=1.0,
                   step_mm=10.0, catalog=None):
        field = MagneticField.load(field_path, polarity=polarity, scale=scale)
        return cls(field, configurations, step_mm=step_mm, catalog=catalog)

    def labels_from_crossings(self, crossings, *, initial_z=None, min_muon_stations=3,
                              include_layer_hits=False):
        """Label externally projected crossings on self.plane_z without transport.

        Supply each track's initial_z to exempt upstream planes from the
        completeness check. Failed downstream transport is still an error.
        """
        if not np.array_equal(crossings.plane_z, self.plane_z):
            raise ValueError("crossings must use the comparison's complete plane_z array")
        columns = {}
        for label, service in self.services.items():
            planes = service.configuration.model.plane_z
            indices = np.searchsorted(self.plane_z, planes)
            subset = replace(crossings, plane_z=planes, x=crossings.x[:, indices],
                             y=crossings.y[:, indices], s=crossings.s[:, indices],
                             crossed=crossings.crossed[:, indices])
            columns.update(service.labels_from_crossings(
                subset, label_suffix=label, initial_z=initial_z,
                min_muon_stations=min_muon_stations,
            ))
        if not include_layer_hits:
            columns = {key: value for key, value in columns.items() if not key.startswith("fired_")}
        return columns

    def label_tracks(self, states, *, max_steps=200000, min_muon_stations=3,
                     include_layer_hits=False):
        """Return one-dimensional label arrays in input-track order.

        states accepts TrackState objects or broadcastable x,y,z,tx,ty,q_over_p
        columns. Propagation happens once over the union of all sensor Zs.
        """
        service = next(iter(self.services.values()))
        crossings, initial_z = service._propagate_to_planes(states, self.plane_z, max_steps=max_steps)
        return self.labels_from_crossings(crossings, initial_z=initial_z,
                                         min_muon_stations=min_muon_stations,
                                         include_layer_hits=include_layer_hits)


def label_tracks(states, *, field=None, field_path=None, configurations=None,
                 polarity=1, scale=1.0, step_mm=10.0, catalog=None,
                 max_steps=200000, min_muon_stations=3, include_layer_hits=False):
    """One-call full-detector transport and geometry-suffixed track labels.

    Supply exactly one field service or field_path. Reuse GeometryComparison
    for successive large batches to avoid loading field and geometry again.
    """
    if (field is None) == (field_path is None):
        raise ValueError("supply exactly one of field or field_path")
    if field_path is not None:
        service = GeometryComparison.from_files(field_path, configurations, polarity=polarity,
                                                scale=scale, step_mm=step_mm, catalog=catalog)
    else:
        if polarity != 1 or scale != 1.0:
            raise ValueError("configure polarity/scale on the supplied field service")
        service = GeometryComparison(field, configurations, step_mm=step_mm, catalog=catalog)
    return service.label_tracks(states, max_steps=max_steps, min_muon_stations=min_muon_stations,
                                include_layer_hits=include_layer_hits)
