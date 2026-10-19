"""Detector-plane XY acceptances, and track transport through them."""

from .geometry_io import (
    box_with_hole,
    dump_geometries,
    load_geometry,
    polygon_from_points,
    save_geometry_figure,
)
from .acceptance import AcceptanceModel, Station, group_stations, write_table
from .configuration import (
    LoadedMaskConfiguration,
    available_configurations,
    load_mask_configuration,
)
from .magnetic_field import (
    MEV_PER_TESLA_MM,
    MagneticField,
    UniformField,
    zero_field,
)
from .propagation import Track, propagate, propagate_to_z, trajectory
from .trajectory import TrackState, Trajectory, TrajectoryError, create_trajectory
from .transport import BatchTransportResult, DetectorTransport, TrackTransportResult
from .variants import ft_with_hole_layers, mp_hole_envelope
from .comparison import (
    DEFAULT_COMPARISON, GeometryComparison, PolarityComparison,
    label_tracks, label_tracks_both_polarities,
)
from .selection import MTSelection

__all__ = [
    "MTSelection",
    "PolarityComparison",
    "label_tracks_both_polarities",
    "DEFAULT_COMPARISON",
    "GeometryComparison",
    "label_tracks",
    "box_with_hole",
    "polygon_from_points",
    "dump_geometries",
    "load_geometry",
    "save_geometry_figure",
    "MEV_PER_TESLA_MM",
    "MagneticField",
    "UniformField",
    "zero_field",
    "Track",
    "propagate",
    "propagate_to_z",
    "trajectory",
    "TrackState",
    "Trajectory",
    "TrajectoryError",
    "create_trajectory",
    "DetectorTransport",
    "TrackTransportResult",
    "BatchTransportResult",
    "ft_with_hole_layers",
    "mp_hole_envelope",
    "AcceptanceModel",
    "Station",
    "group_stations",
    "write_table",
    "LoadedMaskConfiguration",
    "available_configurations",
    "load_mask_configuration",
]
