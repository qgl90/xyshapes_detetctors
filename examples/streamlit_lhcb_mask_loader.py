"""Browse named LHCb mask configurations and evaluate a reference trajectory.

Run with::

    PYTHONPATH=src streamlit run examples/streamlit_lhcb_mask_loader.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xyshapes_detectors import (  # noqa: E402
    Track,
    available_configurations,
    load_mask_configuration,
)
from xyshapes_detectors.magnetic_field import MagneticField, zero_field  # noqa: E402


st.set_page_config(page_title="LHCb Detector Mask Loader", layout="wide")
st.title("LHCb Detector Mask Loader")

catalog_path = ROOT / "detector_setups" / "catalog.json"
catalog = available_configurations(catalog_path)
ready = [item for item in catalog if item["available"]]
if not ready:
    st.error(f"No registered setup directories were found from {catalog_path}.")
    st.stop()

ids = [item["id"] for item in ready]
labels = {item["id"]: item["label"] for item in ready}
default_id = "lhcb_lower_up_y8x6_ft_8_8_10"
default_index = ids.index(default_id) if default_id in ids else 0

with st.sidebar:
    st.header("Geometry")
    selected_id = st.selectbox("Configuration", ids, index=default_index, format_func=lambda key: labels[key])
    comparison_choices = ["None", *ids]
    comparison_id = st.selectbox("Compare with", comparison_choices)
    st.caption(f"Catalog: {catalog_path.relative_to(ROOT)}")

configuration = load_mask_configuration(selected_id, catalog=catalog_path)
st.subheader(configuration.label)
st.caption(configuration.description)

config = configuration.config
up_variant = config.get("up", {}).get("variant", "n/a")
ft_stations = config.get("ft", {}).get("stations", {})
ft_modules = "/".join(str(ft_stations.get(station, "-")) for station in ("T1", "T2", "T3"))
summary = pd.DataFrame(configuration.layer_inventory())
surface_count = len(configuration.plane_inventory())
layer_count = len(configuration.model.stations)

metric_cols = st.columns(4)
metric_cols[0].metric("Subdetectors", len(configuration.model.detectors))
metric_cols[1].metric("Stored surfaces", surface_count)
metric_cols[2].metric("Logical layers", layer_count)
metric_cols[3].metric("UP / SciFi", f"{up_variant} / {ft_modules}")

geometry_tab, inventory_tab, trajectory_tab = st.tabs(["3D Geometry", "Plane Inventory", "Reference Trajectory"])

with geometry_tab:
    compressed_geometry = configuration.path / "experiment_3d.html.gz"
    geometry_path = configuration.path / "experiment_3d.html"
    if compressed_geometry.exists():
        html = gzip.decompress(compressed_geometry.read_bytes()).decode("utf-8")
        components.html(html, height=720, scrolling=True)
    elif geometry_path.exists():
        components.html(geometry_path.read_text(encoding="utf-8"), height=720, scrolling=True)
    else:
        st.warning(f"This setup has no generated 3D view: {geometry_path}")
    st.dataframe(summary, use_container_width=True, hide_index=True)

with inventory_tab:
    inventory = pd.DataFrame(configuration.plane_inventory())
    detectors = ["All", *configuration.model.detectors]
    detector_filter = st.selectbox("Subdetector", detectors)
    if detector_filter != "All":
        inventory = inventory[inventory["detector"] == detector_filter]
    st.dataframe(inventory, use_container_width=True, hide_index=True)
    st.json(config, expanded=False)

with trajectory_tab:
    st.write("Reference-track intersections are evaluated at every stored surface z. Layer counts group the surfaces using the configuration's station labels.")
    with st.form("reference_track"):
        fields = st.columns(4)
        charge = fields[0].selectbox("Charge [e]", (-1, 1), index=1)
        momentum = fields[1].number_input("Momentum [MeV/c]", min_value=1.0, value=50000.0, step=1000.0)
        tx = fields[2].number_input("tx", value=0.0, step=0.001, format="%.5f")
        ty = fields[3].number_input("ty", value=0.0, step=0.001, format="%.5f")
        origin_cols = st.columns(4)
        x0 = origin_cols[0].number_input("x0 [mm]", value=0.0, step=1.0)
        y0 = origin_cols[1].number_input("y0 [mm]", value=0.0, step=1.0)
        z0 = origin_cols[2].number_input("z0 [mm]", value=0.0, step=1.0)
        use_field = origin_cols[3].checkbox("Use magnetic field", value=False)
        field_path = st.text_input("Field map (.root or .npz)", value=str(ROOT / "magnetic_field_v5r0.root"), disabled=not use_field)
        field_scale = st.number_input("Field scale", value=1.0, step=0.1, disabled=not use_field)
        run = st.form_submit_button("Propagate and count fired layers", type="primary")

    if run:
        try:
            field = MagneticField.load(field_path, scale=field_scale) if use_field else zero_field()
            track = Track.from_slopes(charge, momentum, tx, ty, (x0, y0, z0))
            report = configuration.evaluate_reference_track(track, field)
            comparison_report = None
            if comparison_id != "None" and comparison_id != selected_id:
                comparison = load_mask_configuration(comparison_id, catalog=catalog_path)
                comparison_report = comparison.evaluate_reference_track(track, field)
            st.session_state["mask_report"] = report
            st.session_state["mask_comparison_report"] = comparison_report
            st.session_state["mask_comparison_id"] = comparison_id
        except Exception as error:
            st.error(str(error))

    report = st.session_state.get("mask_report")
    if report and report["configuration"] == selected_id:
        st.caption(f"Transport stopped: {report['stop_reason']} at z={report['z_exit_mm']:.2f} mm")
        counts = pd.DataFrame(report["counts"])
        comparison_report = st.session_state.get("mask_comparison_report")
        if comparison_report and st.session_state.get("mask_comparison_id") == comparison_id:
            comparison_counts = pd.DataFrame(comparison_report["counts"]).rename(columns={
                "fired_layers": "comparison_fired_layers",
                "fired_surfaces": "comparison_fired_surfaces",
            })
            counts = counts.merge(
                comparison_counts[["detector", "comparison_fired_layers", "comparison_fired_surfaces"]],
                on="detector",
                how="left",
            )
            counts["layer_delta"] = counts["fired_layers"] - counts["comparison_fired_layers"]
        st.subheader("Fired counts by subdetector")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.subheader("Surface intersections")
        hits = pd.DataFrame(report["planes"])
        hits["status"] = hits["fired"].map({True: "Fired", False: "Missed"})
        st.dataframe(
            hits[["detector", "plane", "fired_layer", "z_mm", "x_mm", "y_mm", "crossed", "status", "layer_fired"]],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download trajectory report JSON",
            data=json.dumps(report, indent=2) + "\n",
            file_name=f"{selected_id}_reference_track_masks.json",
            mime="application/json",
        )
