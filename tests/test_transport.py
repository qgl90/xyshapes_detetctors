import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box

from xyshapes_detectors.acceptance import (
    AcceptanceModel,
    group_stations,
    tracks_from_table,
    write_table,
)
from xyshapes_detectors.geometry_io import box_with_hole
from xyshapes_detectors.layout import DetectorLayer
from xyshapes_detectors.configuration import LoadedMaskConfiguration
from xyshapes_detectors.magnetic_field import (
    MEV_PER_TESLA_MM,
    FieldMapError,
    MagneticField,
    UniformField,
    zero_field,
)
from xyshapes_detectors.propagation import (
    Track,
    propagate,
    propagate_to_z,
    trajectory,
)
from xyshapes_detectors.trajectory import (
    TrackState,
    TrajectoryError,
    create_trajectory,
)
from xyshapes_detectors.transport import DetectorTransport


ROOT = Path(__file__).resolve().parents[1]
FIELD_FILE = ROOT / "MagField" / "magnetic_field_v5r0.root"


def linear_field_samples():
    """A small analytic map: By grows linearly with z inside the grid."""
    axis = np.linspace(-500.0, 500.0, 11)
    z_axis = np.linspace(0.0, 1000.0, 11)
    x, y, z = np.meshgrid(axis, axis, z_axis, indexing="ij")
    bx = np.zeros_like(x)
    by = -z / 1000.0
    bz = np.zeros_like(x)
    return [value.ravel() for value in (x, y, z, bx, by, bz)]


class MagneticFieldTest(unittest.TestCase):
    def test_from_samples_reproduces_nodes(self):
        field = MagneticField.from_samples(*linear_field_samples())
        self.assertEqual(field.grid.shape, (11, 11, 11))
        self.assertEqual(field.grid.spacing, (100.0, 100.0, 100.0))
        np.testing.assert_allclose(field.b_at(0.0, 0.0, 500.0), (0.0, -0.5, 0.0))
        np.testing.assert_allclose(field.b_at(100.0, -200.0, 300.0), (0.0, -0.3, 0.0))

    def test_interpolates_between_nodes(self):
        field = MagneticField.from_samples(*linear_field_samples())
        np.testing.assert_allclose(field.b_at(0.0, 0.0, 450.0), (0.0, -0.45, 0.0))

    def test_outside_the_grid_is_field_free(self):
        field = MagneticField.from_samples(*linear_field_samples())
        np.testing.assert_allclose(field.b_at(0.0, 0.0, 1500.0), (0.0, 0.0, 0.0))
        np.testing.assert_allclose(field.b_at(5000.0, 0.0, 500.0), (0.0, 0.0, 0.0))

    def test_scale_flips_the_polarity(self):
        field = MagneticField.from_samples(*linear_field_samples())
        np.testing.assert_allclose(
            field.scaled(-1).b_at(0.0, 0.0, 500.0), (0.0, 0.5, 0.0)
        )
        np.testing.assert_allclose(
            field.scaled(0).b_at(0.0, 0.0, 500.0), (0.0, 0.0, 0.0)
        )

    def test_broadcasting_and_shapes(self):
        field = MagneticField.from_samples(*linear_field_samples())
        self.assertEqual(field.b_at(0.0, 0.0, 500.0).shape, (3,))
        self.assertEqual(field.b_at([0.0, 1.0], 0.0, 500.0).shape, (2, 3))
        points = np.zeros((4, 5, 3))
        self.assertEqual(field.b_at_points(points).shape, (4, 5, 3))

    def test_irregular_samples_are_rejected(self):
        x, y, z, bx, by, bz = linear_field_samples()
        z = z.copy()
        z[0] = 7.0
        with self.assertRaises(FieldMapError):
            MagneticField.from_samples(x, y, z, bx, by, bz)

    def test_npz_round_trip(self):
        field = MagneticField.from_samples(*linear_field_samples())
        with tempfile.TemporaryDirectory() as directory:
            path = field.to_npz(Path(directory) / "field.npz")
            loaded = MagneticField.load(path)
        self.assertEqual(loaded.grid, field.grid)
        np.testing.assert_array_equal(loaded.values, field.values)

    def test_bending_radius_matches_the_unit_convention(self):
        # 1 GeV/c in 1 T bends with a 3.336 m radius.
        radius = MagneticField.bending_radius(1000.0, 1.0)
        self.assertAlmostEqual(radius, 1000.0 / MEV_PER_TESLA_MM, places=6)
        self.assertEqual(MagneticField.bending_radius(1000.0, 0.0), math.inf)

    def test_uniform_field_is_constant_everywhere(self):
        field = UniformField(0.0, -1.0, 0.0)
        np.testing.assert_allclose(field.b_at(1e6, -1e6, 1e6), (0.0, -1.0, 0.0))
        np.testing.assert_allclose(zero_field().b_at(0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class TrackTest(unittest.TestCase):
    def test_slopes_round_trip(self):
        track = Track.from_slopes(+1, 10000.0, 0.2, -0.1)
        self.assertAlmostEqual(track.tx, 0.2)
        self.assertAlmostEqual(track.ty, -0.1)
        self.assertAlmostEqual(
            math.sqrt(track.dx**2 + track.dy**2 + track.dz**2), 1.0
        )
        self.assertAlmostEqual(track.p, 10000.0)

    def test_momentum_components_round_trip(self):
        track = Track.from_momentum(-1, 300.0, -400.0, 5000.0)
        self.assertAlmostEqual(track.px, 300.0, places=6)
        self.assertAlmostEqual(track.py, -400.0, places=6)
        self.assertAlmostEqual(track.pz, 5000.0, places=6)
        self.assertAlmostEqual(track.pt, 500.0, places=6)

    def test_eta_phi_round_trip(self):
        track = Track.from_eta_phi(+1, 20000.0, 3.0, 0.75)
        self.assertAlmostEqual(track.eta, 3.0, places=9)
        self.assertAlmostEqual(track.phi, 0.75, places=9)

    def test_pt_eta_phi_round_trip(self):
        track = Track.from_pt_eta_phi(-1, 1500.0, 2.5, -1.2)
        self.assertAlmostEqual(track.pt, 1500.0, places=6)
        self.assertAlmostEqual(track.eta, 2.5, places=9)

    def test_zero_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            Track.from_direction(+1, 1000.0, (0.0, 0.0, 0.0))


class PropagationTest(unittest.TestCase):
    def test_straight_line_without_field(self):
        track = Track.from_slopes(+1, 10000.0, 0.2, -0.1)
        x, y, crossed = propagate_to_z([track], zero_field(), 5000.0)
        self.assertTrue(crossed[0])
        self.assertAlmostEqual(x[0], 1000.0, places=6)
        self.assertAlmostEqual(y[0], -500.0, places=6)

    def test_uniform_field_matches_the_analytic_circle(self):
        p, charge, by = 10000.0, 1.0, -1.0
        field = UniformField(0.0, by, 0.0)
        omega = MEV_PER_TESLA_MM * charge * by / p
        track = Track.from_direction(charge, p, (0.0, 0.0, 1.0))
        planes = np.array([1000.0, 3000.0, 5000.0, 10000.0])
        crossings = propagate([track], field, planes, step=10.0)
        for index, z in enumerate(planes):
            self.assertTrue(crossings.crossed[0, index])
            s = math.asin(z * omega) / omega
            expected_x = (math.cos(omega * s) - 1.0) / omega
            self.assertAlmostEqual(crossings.x[0, index], expected_x, places=6)
            self.assertAlmostEqual(crossings.s[0, index], s, places=6)
            self.assertAlmostEqual(crossings.y[0, index], 0.0, places=9)

    def test_opposite_charges_bend_apart(self):
        field = UniformField(0.0, -1.0, 0.0)
        plus = Track.from_direction(+1, 5000.0, (0.0, 0.0, 1.0))
        minus = Track.from_direction(-1, 5000.0, (0.0, 0.0, 1.0))
        crossings = propagate([plus, minus], field, [4000.0])
        self.assertAlmostEqual(crossings.x[0, 0], -crossings.x[1, 0], places=6)
        self.assertGreater(abs(crossings.x[0, 0]), 1.0)

    def test_momentum_magnitude_is_conserved(self):
        # A conserved |p| keeps the trajectory on a circle of fixed radius,
        # so every crossing must sit at that radius from the circle centre.
        p, charge, by = 4000.0, 1.0, -1.0
        field = UniformField(0.0, by, 0.0)
        radius = p / (MEV_PER_TESLA_MM * abs(charge * by))
        centre_x = -math.copysign(radius, charge * by)
        track = Track.from_direction(charge, p, (0.0, 0.0, 1.0))
        planes = [500.0, 1500.0, 3000.0, 4000.0]
        crossings = propagate([track], field, planes, step=10.0)
        for index, z in enumerate(planes):
            self.assertTrue(crossings.crossed[0, index])
            distance = math.hypot(crossings.x[0, index] - centre_x, z)
            self.assertAlmostEqual(distance, radius, delta=1e-5)

    def test_step_size_is_converged_at_the_default(self):
        field = UniformField(0.0, -1.0, 0.0)
        track = Track.from_direction(+1, 3000.0, (0.0, 0.0, 1.0))
        coarse = propagate([track], field, [5000.0], step=10.0)
        fine = propagate([track], field, [5000.0], step=0.5)
        self.assertAlmostEqual(coarse.x[0, 0], fine.x[0, 0], places=5)

    def test_planes_upstream_of_the_origin_are_never_crossed(self):
        track = Track.from_direction(+1, 10000.0, (0.0, 0.0, 1.0), origin=(0, 0, 0))
        crossings = propagate([track], zero_field(), [-500.0, 500.0])
        self.assertFalse(crossings.crossed[0, 0])
        self.assertTrue(crossings.crossed[0, 1])

    def test_backward_tracks_stop_immediately(self):
        track = Track.from_direction(+1, 10000.0, (0.0, 0.0, -1.0))
        crossings = propagate([track], zero_field(), [500.0])
        self.assertEqual(crossings.stop_reason[0], "backward")
        self.assertFalse(crossings.crossed[0, 0])

    def test_tracks_leaving_the_envelope_stop(self):
        track = Track.from_slopes(+1, 10000.0, 1.0, 0.0)
        crossings = propagate(
            [track], zero_field(), [10000.0], envelope=(1000.0, 1000.0)
        )
        self.assertEqual(crossings.stop_reason[0], "escaped")
        self.assertFalse(crossings.crossed[0, 0])

    def test_origin_offset_is_respected(self):
        track = Track.from_slopes(+1, 10000.0, 0.1, 0.0, origin=(5.0, 0.0, 100.0))
        x, y, crossed = propagate_to_z([track], zero_field(), 1100.0)
        self.assertTrue(crossed[0])
        self.assertAlmostEqual(x[0], 5.0 + 0.1 * 1000.0, places=6)

    def test_trajectory_samples_the_path(self):
        field = UniformField(0.0, -1.0, 0.0)
        track = Track.from_direction(+1, 5000.0, (0.0, 0.0, 1.0))
        path = trajectory(track, field, 4000.0, n_samples=41)
        self.assertEqual(path.shape, (41, 3))
        self.assertAlmostEqual(path[0, 2], 0.0)
        self.assertAlmostEqual(path[-1, 2], 4000.0)
        self.assertTrue(np.all(np.diff(path[:, 0]) > 0.0))

    def test_batch_matches_single_tracks(self):
        field = UniformField(0.0, -1.0, 0.0)
        tracks = [
            Track.from_slopes(charge, p, 0.05, -0.02)
            for charge in (+1, -1)
            for p in (3000.0, 20000.0)
        ]
        planes = [1000.0, 5000.0, 9000.0]
        batch = propagate(tracks, field, planes)
        for index, track in enumerate(tracks):
            single = propagate([track], field, planes)
            np.testing.assert_allclose(batch.x[index], single.x[0], atol=1e-9)
            np.testing.assert_allclose(batch.y[index], single.y[0], atol=1e-9)


class ZTrajectoryTest(unittest.TestCase):
    def test_straight_track_intercepts_exact_sensor_planes(self):
        state = TrackState(10.0, -5.0, 0.1, -0.2, 0.0, z=100.0)
        path = create_trajectory(
            zero_field(), state, 1100.0, sensor_z=(100.0, 500.0, 1100.0),
            step_mm=250.0,
        )
        np.testing.assert_allclose(
            path.intercepts((100.0, 500.0, 1100.0)),
            ((10.0, -5.0), (50.0, -85.0), (110.0, -205.0)),
            atol=1e-12,
        )

    def test_z_stepping_agrees_with_path_length_propagator(self):
        field = UniformField(0.0, -1.0, 0.0)
        initial = TrackState.from_charge_momentum(
            +1, 10000.0, 0.0, 0.0, 0.05, -0.02
        )
        planes = np.asarray((500.0, 1000.0, 2000.0))
        z_path = create_trajectory(
            field, initial, 2000.0, sensor_z=planes, step_mm=2.0
        )
        path_track = Track.from_slopes(
            +1, 10000.0, 0.05, -0.02, origin=(0.0, 0.0, 0.0)
        )
        path_length = propagate([path_track], field, planes, step=2.0)
        np.testing.assert_allclose(
            z_path.intercepts(planes),
            np.column_stack((path_length.x[0], path_length.y[0])),
            atol=0.02,
        )

    def test_hermite_interpolation_agrees_between_integration_nodes(self):
        field = UniformField(0.0, -1.0, 0.0)
        initial = TrackState.from_charge_momentum(
            +1, 10000.0, 0.0, 0.0, 0.05, -0.02
        )
        path = create_trajectory(field, initial, 2000.0, step_mm=10.0)
        z_query = 1234.5
        interpolated = path.xy_at(z_query)
        reference = propagate(
            [Track.from_slopes(+1, 10000.0, 0.05, -0.02)],
            field,
            [z_query],
            step=0.25,
        )
        np.testing.assert_allclose(
            interpolated, (reference.x[0, 0], reference.y[0, 0]), atol=0.02
        )

    def test_trajectory_can_fire_exact_acceptance_surfaces(self):
        model = AcceptanceModel.from_layers(toy_layers(), exact_surfaces=True)
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        state = TrackState.from_charge_momentum(
            track.charge, track.p, track.x, track.y, track.tx, track.ty, z=track.z
        )
        path = create_trajectory(
            zero_field(), state, max(model.plane_z) + 10,
            sensor_z=model.plane_z,
        )
        z_stepped = model.counts(fired=model.fired_trajectory(path))
        path_length = model.evaluate([track], zero_field())
        for detector in model.detectors:
            self.assertEqual(
                z_stepped[detector][0], path_length[f"n{detector}"][0]
            )

    def test_trajectory_must_span_all_acceptance_surfaces(self):
        model = AcceptanceModel.from_layers(toy_layers(), exact_surfaces=True)
        path = create_trajectory(zero_field(), TrackState(0, 0, 0, 0, 0), 1000)
        with self.assertRaisesRegex(ValueError, "span all acceptance surfaces"):
            model.fired_trajectory(path)

    def test_state_tuple_uses_q_over_p_in_inverse_mev(self):
        state = TrackState.from_values((1.0, 2.0, 0.03, -0.04, -2e-4), z=15.0)
        self.assertEqual(state.as_tuple(), (1.0, 2.0, 0.03, -0.04, -2e-4))
        self.assertEqual(state.z, 15.0)

    def test_curling_track_raises_instead_of_returning_partial_hits(self):
        state = TrackState(0.0, 0.0, 0.0, 0.0, 1.0 / 50.0)
        with self.assertRaisesRegex(TrajectoryError, "curling|transverse"):
            create_trajectory(
                UniformField(0.0, -1.0, 0.0), state, 500.0, step_mm=5.0
            )

    def test_z_trajectory_rejects_extrapolation_and_upstream_requests(self):
        state = TrackState(0.0, 0.0, 0.0, 0.0, 0.0, z=10.0)
        path = create_trajectory(zero_field(), state, 20.0)
        with self.assertRaises(ValueError):
            path.xy_at(21.0)
        with self.assertRaises(ValueError):
            create_trajectory(zero_field(), state, 5.0)


class DetectorTransportTest(unittest.TestCase):
    def setUp(self):
        model = AcceptanceModel.from_layers(toy_layers(), exact_surfaces=True)
        self.configuration = LoadedMaskConfiguration(
            "toy", "Toy", "test setup", Path("."), {}, model
        )

    def test_facade_returns_detector_and_layer_flags(self):
        service = DetectorTransport(self.configuration, zero_field())
        state = TrackState.from_charge_momentum(
            +1, 50000.0, 0.0, 0.0, 0.02, 0.01
        )
        result = service.propagate(state, label_suffix="toy")
        self.assertEqual(result.detector_counts["TV"], 2)
        self.assertEqual(result.detector_counts["MUON"], 4)
        self.assertTrue(result.labels["nTV_toy"] == 2)
        self.assertTrue(result.labels["hasRich1Entry_toy"])
        self.assertTrue(result.labels["inPicoCal_toy"])
        self.assertTrue(result.labels["isMuon_toy"])
        self.assertTrue(result.layer_hits["MUON_M2"])
        row = {"q_over_p": state.q_over_p, **result.as_record()}
        self.assertEqual(row["q_over_p"], state.q_over_p)

    def test_batch_matches_single_track_positions_and_labels(self):
        service = DetectorTransport(self.configuration, UniformField(0, -1, 0))
        states = [TrackState.from_charge_momentum(q, 50000, 0, 0, .02, .01)
                  for q in (-1, 0, 1)]
        batch = service.propagate_batch(states, label_suffix="toy")
        for row, state in enumerate(states):
            single = service.propagate(state, label_suffix="toy")
            for name, value in single.labels.items():
                self.assertEqual(batch.labels[name][row], value, name)
            np.testing.assert_allclose(
                single.trajectory.intercepts(batch.crossings.plane_z),
                np.column_stack((batch.crossings.x[row], batch.crossings.y[row])),
                atol=2e-5,
            )

    def test_batch_accepts_broadcast_columns_and_reuses_crossings(self):
        service = DetectorTransport(self.configuration, zero_field())
        columns = dict(x=0, y=0, z=0, tx=[.02, .04], ty=.01, q_over_p=1 / 50000)
        batch = service.propagate_batch(columns)
        labels = service.labels_from_crossings(batch.crossings, label_suffix="other")
        np.testing.assert_array_equal(labels["nTV_other"], batch.labels["nTV"])
        np.testing.assert_array_equal(batch.as_columns()["nMuon"], [4, 4])

    def test_failed_batch_transport_raises(self):
        service = DetectorTransport(self.configuration, UniformField(0, -1, 0))
        with self.assertRaisesRegex(TrajectoryError, "batch tracks failed"):
            service.propagate_batch([TrackState(0, 0, 0, 0, 1 / 50)])


def toy_layers():
    """A small stand-in setup: paired faces, four UP layers, four Muon stations."""
    layers = []
    for index, z in enumerate((100.0, 100.8, 200.0, 200.8)):
        layers.append(DetectorLayer("TV", f"TV{index}", z, "tv", box(-40, -40, 40, 40)))
    for index, z in enumerate((1000.0, 1000.2, 1006.5, 1006.7)):
        layers.append(DetectorLayer("UP", f"UP{index}", z, "up", box(-500, -500, 500, 500)))
    for name, z in (("RICH1_entry", 500.0), ("RICH1_exit", 900.0)):
        layers.append(DetectorLayer("RICH1", name, z, name, box(-300, -300, 300, 300)))
    for name, z in (("PicoCal_entry", 5000.0), ("PicoCal_exit", 5035.0)):
        layers.append(DetectorLayer("PicoCal", name, z, name, box(-2000, -2000, 2000, 2000)))
    for index, z in enumerate((6000.0, 7000.0, 8000.0, 9000.0)):
        layers.append(DetectorLayer(
            "MUON", f"M{index + 2}", z, f"M{index + 2}",
            box_with_hole(6000, 6000, hole_width=200, hole_height=200),
        ))
    return layers


class StationGroupingTest(unittest.TestCase):
    def test_nearby_surfaces_merge_into_one_station(self):
        stations = group_stations(toy_layers())
        counts = {}
        for station in stations:
            counts[station.detector] = counts.get(station.detector, 0) + 1
        self.assertEqual(counts["TV"], 2)
        self.assertEqual(counts["UP"], 1)
        self.assertEqual(counts["RICH1"], 2)
        self.assertEqual(counts["PicoCal"], 2)
        self.assertEqual(counts["MUON"], 4)

    def test_tv_uses_a_tighter_default_window(self):
        stations = group_stations(toy_layers())
        tv = [station for station in stations if station.detector == "TV"]
        self.assertEqual([station.n_surfaces for station in tv], [2, 2])
        self.assertAlmostEqual(tv[0].z, 100.4)

    def test_entry_and_exit_faces_keep_their_names(self):
        names = {
            station.name
            for station in group_stations(toy_layers())
            if station.detector == "RICH1"
        }
        self.assertEqual(names, {"RICH1_entry", "RICH1_exit"})

    def test_explicit_station_labels_win(self):
        explicit = {"UP0": "UPa", "UP1": "UPa", "UP2": "UPb", "UP3": "UPb"}
        stations = group_stations(toy_layers(), explicit=explicit)
        up = [station for station in stations if station.detector == "UP"]
        self.assertEqual(sorted(station.name for station in up), ["UPa", "UPb"])

    def test_intersection_combines_conservatively(self):
        layers = [
            DetectorLayer("X", "a", 10.0, "a", box(-100, -100, 100, 100)),
            DetectorLayer("X", "b", 10.5, "b", box(-50, -50, 150, 150)),
        ]
        union = group_stations(layers)[0]
        overlap = group_stations(layers, combine="intersection")[0]
        self.assertAlmostEqual(union.geometry.bounds[0], -100.0)
        self.assertAlmostEqual(overlap.geometry.bounds[0], -50.0)


class AcceptanceModelTest(unittest.TestCase):
    def setUp(self):
        self.model = AcceptanceModel.from_layers(toy_layers())
        self.field = zero_field()

    def test_station_inventory(self):
        self.assertEqual(len(self.model), 11)
        self.assertEqual(
            self.model.detectors, ("TV", "RICH1", "UP", "PicoCal", "MUON")
        )
        self.assertEqual(len(self.model.station_table()), 11)
        self.assertIn("MUON", self.model.describe())

    def test_a_central_track_fires_everything(self):
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        columns = self.model.evaluate([track], self.field)
        self.assertEqual(columns["nTV"][0], 2)
        self.assertEqual(columns["nUP"][0], 1)
        self.assertEqual(columns["nMUON"][0], 4)
        self.assertTrue(columns["isMuon"][0])
        self.assertTrue(columns["inPicoCal"][0])
        self.assertTrue(columns["RICH1_entry_exit"][0])
        self.assertFalse(columns["RICH1_entry_only"][0])

    def test_canonical_detector_mask_flags(self):
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        columns = self.model.evaluate([track], self.field)
        self.assertEqual(columns["nTV"][0], 2)
        self.assertEqual(columns["nUP"][0], 1)
        self.assertEqual(columns["nMuon"][0], 4)
        self.assertTrue(columns["hasRichEntry"][0])
        self.assertTrue(columns["hasRichExit"][0])
        self.assertTrue(columns["hasRich1Entry"][0])
        self.assertTrue(columns["hasPicoCal"][0])

    def test_canonical_flags_get_configuration_suffixes(self):
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        columns = self.model.evaluate([track], self.field, label_suffix="lower")
        for name in (
            "nMuon", "hasRichEntry", "hasRichExit", "hasRich1Entry",
            "hasPicoCal",
        ):
            np.testing.assert_array_equal(columns[f"{name}_lower"], columns[name])

    def test_a_track_down_the_beam_hole_fires_no_muon_station(self):
        track = Track.from_direction(+1, 50000.0, (0.0, 0.0, 1.0))
        columns = self.model.evaluate([track], self.field)
        self.assertEqual(columns["nMUON"][0], 0)
        self.assertFalse(columns["isMuon"][0])
        self.assertTrue(columns["inPicoCal"][0])

    def test_entry_only_when_the_exit_face_is_missed(self):
        # 0.4 rad leaves the 300 mm RICH1 square between its two faces.
        track = Track.from_slopes(+1, 50000.0, 0.4, 0.0)
        columns = self.model.evaluate([track], self.field)
        self.assertTrue(columns["RICH1_entry"][0])
        self.assertFalse(columns["RICH1_exit"][0])
        self.assertTrue(columns["RICH1_entry_only"][0])
        self.assertFalse(columns["RICH1_entry_exit"][0])

    def test_muon_threshold_is_configurable(self):
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        columns = self.model.evaluate([track], self.field, min_muon_stations=5)
        self.assertFalse(columns["isMuon"][0])

    def test_station_flags_are_optional(self):
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        columns = self.model.evaluate([track], self.field, station_flags=True)
        self.assertTrue(columns["fired_MUON_M2"][0])

    def test_charge_changes_the_counts_in_a_field(self):
        field = UniformField(0.0, -1.0, 0.0)
        tracks = [
            Track.from_direction(charge, 50000.0, (0.0, 0.0, 1.0))
            for charge in (+1, -1)
        ]
        columns = self.model.evaluate(tracks, field)
        self.assertTrue(columns["isMuon"][0])
        self.assertTrue(columns["isMuon"][1])
        # Equal and opposite deflection, so the counts must agree.
        self.assertEqual(columns["nMUON"][0], columns["nMUON"][1])
        self.assertEqual(columns["nTV"][0], columns["nTV"][1])

    def test_exact_surfaces_uses_one_plane_per_surface(self):
        exact = AcceptanceModel.from_layers(toy_layers(), exact_surfaces=True)
        self.assertEqual(len(exact), len(self.model))
        self.assertEqual(exact.plane_z.size, len(toy_layers()))
        self.assertEqual(self.model.plane_z.size, len(self.model))
        self.assertIn("each surface at its own z", exact.describe())

    def test_exact_surfaces_agrees_where_stations_are_thin(self):
        exact = AcceptanceModel.from_layers(toy_layers(), exact_surfaces=True)
        tracks = [
            Track.from_slopes(+1, 50000.0, tx, 0.01)
            for tx in (0.0, 0.02, 0.1, 0.3)
        ]
        merged_columns = self.model.evaluate(tracks, self.field)
        exact_columns = exact.evaluate(tracks, self.field)
        for detector in self.model.detectors:
            np.testing.assert_array_equal(
                merged_columns[f"n{detector}"],
                exact_columns[f"n{detector}"],
            )

    def test_exact_surfaces_needs_member_surfaces(self):
        from xyshapes_detectors.acceptance import Station

        bare = Station("X", "x", 1.0, 1.0, 1.0, box(-1, -1, 1, 1), ("x",))
        with self.assertRaises(ValueError):
            AcceptanceModel([bare], exact_surfaces=True)

    def test_label_suffix_adds_aliases(self):
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        columns = self.model.evaluate(
            [track], self.field, label_suffix="Run5"
        )
        self.assertEqual(columns["nMUON_Run5"][0], columns["nMUON"][0])
        self.assertEqual(columns["hasUP_Run5"][0], columns["hasUP"][0])

    def test_load_accepts_a_directory_or_a_layout_csv(self):
        from xyshapes_detectors.acceptance import AcceptanceModel as Model
        from xyshapes_detectors.geometry_io import dump_geometries

        row = "Use,detector,layer,z,xyshape\n1,X,x0,500,plate\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A bare layout CSV finds its pickles beside itself.
            dump_geometries({"plate": box(-100, -100, 100, 100)}, root / "bare")
            layout = root / "bare" / "layout.csv"
            layout.write_text(row, encoding="utf-8")
            from_csv = Model.load(layout)

            # A setup directory keeps them under shapes/.
            setup = root / "setup"
            dump_geometries({"plate": box(-100, -100, 100, 100)}, setup / "shapes")
            (setup / "layout.csv").write_text(row, encoding="utf-8")
            from_dir = Model.load(setup)
        self.assertEqual(from_csv.detectors, ("X",))
        self.assertEqual(from_dir.detectors, ("X",))

    def test_crossings_from_other_planes_are_refused(self):
        track = Track.from_slopes(+1, 50000.0, 0.0, 0.0)
        crossings = propagate([track], self.field, [1.0, 2.0])
        with self.assertRaises(ValueError):
            self.model.fired(crossings)

    def test_augment_keeps_the_input_columns(self):
        table = {
            "event": np.array([1, 2]),
            "charge": np.array([+1, -1]),
            "p": np.array([50000.0, 50000.0]),
            "tx": np.array([0.02, -0.02]),
            "ty": np.array([0.01, -0.01]),
        }
        augmented = self.model.augment(table, self.field)
        np.testing.assert_array_equal(augmented["event"], table["event"])
        self.assertIn("isMuon", augmented)
        self.assertNotIn("eta", augmented)
        self.assertEqual(len(augmented["nMUON"]), 2)

    def test_run5_label_suffix_aliases(self):
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        columns = self.model.evaluate([track], self.field, label_suffix="Run5")
        self.assertEqual(columns["nUP_Run5"][0], columns["nUP"][0])
        self.assertEqual(columns["hasMUON_Run5"][0], columns["hasMUON"][0])
        self.assertTrue(columns["RICH1_entry_exit_Run5"][0])

    def test_label_external_crossing_table(self):
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        crossings = propagate([track], self.field, self.model.plane_z)
        table = {}
        for index, station in enumerate(self.model.stations):
            table[f"x_{station.detector}_{station.name}"] = crossings.x[:, index]
            table[f"y_{station.detector}_{station.name}"] = crossings.y[:, index]
        labels = self.model.label_crossings(table, station_flags=True)
        self.assertTrue(labels["hasMUON_Run5"][0])
        self.assertTrue(labels["fired_MUON_M2_Run5"][0])
        self.assertEqual(labels["nMuon_Run5"][0], 4)
        self.assertTrue(labels["hasRich1Entry_Run5"][0])
        self.assertTrue(labels["hasPicoCal_Run5"][0])

    def test_label_external_surface_crossings(self):
        model = AcceptanceModel.from_layers(toy_layers(), exact_surfaces=True)
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        crossings = propagate([track], self.field, model.plane_z)
        column_of = {float(z): index for index, z in enumerate(model.plane_z)}
        table = {}
        for station in model.stations:
            for index, (z, _) in enumerate(station.surfaces):
                plane = station.layers[index]
                base = f"{station.detector}_{plane}"
                column = column_of[float(z)]
                table[f"x_{base}"] = crossings.x[:, column]
                table[f"y_{base}"] = crossings.y[:, column]
        labels = model.label_surface_crossings(table)
        self.assertEqual(labels["nTV_Run5"][0], 2)
        self.assertEqual(labels["nUP_Run5"][0], 1)
        self.assertEqual(labels["nMuon_Run5"][0], 4)
        self.assertTrue(labels["hasRich1Entry_Run5"][0])
        self.assertTrue(labels["hasPicoCal_Run5"][0])

    def test_tracks_from_table_accepts_momentum_components(self):
        table = {
            "charge": [1],
            "p": [1000.0],
            "px": [0.0],
            "py": [0.0],
            "pz": [1000.0],
        }
        tracks = tracks_from_table(table)
        self.assertAlmostEqual(tracks[0].dz, 1.0)

    def test_tracks_from_table_needs_a_direction(self):
        with self.assertRaises(KeyError):
            tracks_from_table({"charge": [1], "p": [1000.0]})

    def test_write_table(self):
        track = Track.from_slopes(+1, 50000.0, 0.02, 0.01)
        columns = self.model.evaluate([track], self.field)
        with tempfile.TemporaryDirectory() as directory:
            path = write_table(Path(directory) / "tracks.csv", columns)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        header = lines[0].split(",")
        self.assertIn("isMuon", header)
        self.assertEqual(lines[1].split(",")[header.index("isMuon")], "1")


@unittest.skipUnless(FIELD_FILE.exists(), "the LHCb field map is not available")
class LHCbFieldMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.field = MagneticField.from_root(FIELD_FILE)
        except FieldMapError as error:  # uproot missing
            raise unittest.SkipTest(str(error))

    def test_grid_covers_the_spectrometer(self):
        xmin, ymin, zmin, xmax, ymax, zmax = self.field.bounds
        self.assertLessEqual(zmin, -1200.0)
        self.assertGreaterEqual(zmax, 14000.0)
        self.assertLessEqual(xmin, -4000.0)
        self.assertGreaterEqual(xmax, 4000.0)

    def test_peak_on_axis_field_is_the_lhcb_dipole(self):
        by = self.field.b_at(0.0, 0.0, 4800.0)[1]
        self.assertAlmostEqual(by, -1.044, places=2)

    def test_integrated_field_is_about_four_tesla_metre(self):
        integral = self.field.integral_bdl(-1200.0, 14100.0)[1]
        self.assertAlmostEqual(integral / 1000.0, -4.23, places=1)

    def test_origin_to_10000_mm_field_integral_is_about_four_tesla_metre(self):
        integral = self.field.integral_bdl(0.0, 10000.0)[1]
        self.assertAlmostEqual(integral / 1000.0, -4.19, delta=0.08)

    def test_reversed_polarity_flips_all_field_components(self):
        reversed_field = MagneticField.load(FIELD_FILE, polarity=-1)
        points = np.asarray(((0.0, 0.0, 4800.0), (500.0, -250.0, 6000.0)))
        np.testing.assert_allclose(
            reversed_field.b_at_points(points), -self.field.b_at_points(points)
        )
        np.testing.assert_allclose(
            reversed_field.integral_bdl(0.0, 10000.0),
            -self.field.integral_bdl(0.0, 10000.0),
            atol=1e-6,
        )
        with self.assertRaises(ValueError):
            MagneticField.load(FIELD_FILE, polarity=0)

    def test_reversed_polarity_matches_reversing_track_charge(self):
        from xyshapes_detectors.trajectory import create_trajectory

        positive_state = TrackState.from_charge_momentum(
            +1, 50000.0, 0.0, 0.0, 0.01, -0.005
        )
        negative_state = TrackState.from_charge_momentum(
            -1, 50000.0, 0.0, 0.0, 0.01, -0.005
        )
        planes = (1000.0, 5000.0, 10000.0)
        positive_reversed = create_trajectory(
            MagneticField.load(FIELD_FILE, polarity=-1),
            positive_state,
            10000.0,
            sensor_z=planes,
        )
        negative_nominal = create_trajectory(
            self.field, negative_state, 10000.0, sensor_z=planes
        )
        np.testing.assert_allclose(
            positive_reversed.intercepts(planes),
            negative_nominal.intercepts(planes),
            atol=1e-9,
        )

    def test_bending_is_inversely_proportional_to_momentum(self):
        planes = [12000.0]
        deflections = {}
        for p in (10000.0, 100000.0):
            track = Track.from_direction(+1, p, (0.0, 0.0, 1.0))
            deflections[p] = propagate([track], self.field, planes).x[0, 0]
        ratio = deflections[10000.0] / deflections[100000.0]
        self.assertAlmostEqual(ratio, 10.0, delta=0.5)

    def test_muon_region_is_field_free(self):
        np.testing.assert_allclose(
            self.field.b_at(1000.0, 500.0, 17670.0), (0.0, 0.0, 0.0)
        )


if __name__ == "__main__":
    unittest.main()
