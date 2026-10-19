"""Streamlit UI for building configurable LHCb XY-surface setups.

Run with:

    PYTHONPATH=src streamlit run examples/streamlit_lhcb_builder.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from build_lhcb_xy_setup import (  # noqa: E402
    DEFAULT_CONFIGS,
    DETECTOR_DEFAULT,
    DETECTOR_DESCRIPTIONS,
    UP_PLANE_DESCRIPTIONS,
    UP_VARIANT_DESCRIPTIONS,
    build_setup,
)
from xyshapes_detectors.tv_geometry import TV_CONFIG


st.set_page_config(
    page_title="LHCb XY Surface Builder",
    layout="wide",
)


def copy_default(preset):
    return json.loads(json.dumps(DEFAULT_CONFIGS[preset]))


def summary_dataframe(rows):
    data = []
    for row in rows:
        data.append({
            "detector": row["detector"],
            "source surfaces": row["planes"],
            "fired layers": row["fired_layers"],
            "z_min [mm]": row["z_min"],
            "z_max [mm]": row["z_max"],
            "x size [mm]": row["x_max"] - row["x_min"],
            "y size [mm]": row["y_max"] - row["y_min"],
        })
    return pd.DataFrame(data)


st.title("LHCb XY Surface Setup Builder")
st.caption("Configure TV, UP, FT, RICH, PicoCal, MP, and Muon surfaces for transport and acceptance-mask checks.")

with st.sidebar:
    st.header("Setup")
    preset = st.selectbox("Start from preset", sorted(DEFAULT_CONFIGS), index=sorted(DEFAULT_CONFIGS).index("lower"))
    base = copy_default(preset)
    setup_name = st.text_input("Setup name", value=base["name"])
    output_root = st.text_input("Output root", value=str(ROOT / "detector_setups"))
    generate_3d = st.checkbox("Generate 3D HTML", value=True)

    st.header("Detector Groups")
    include = []
    for detector in DETECTOR_DEFAULT:
        enabled = st.checkbox(
            detector,
            value=detector in base.get("include", DETECTOR_DEFAULT),
            help=DETECTOR_DESCRIPTIONS[detector],
        )
        if enabled:
            include.append(detector)

left, middle, right = st.columns([1, 1, 1])

with left:
    st.subheader("UP")
    if "UP" in include:
        up_base = base.get("up", {})
        up_variant = st.radio(
            "UP variant",
            list(UP_VARIANT_DESCRIPTIONS),
            index=list(UP_VARIANT_DESCRIPTIONS).index(up_base.get("variant", "gdml")),
            format_func=lambda key: f"{key}: {UP_VARIANT_DESCRIPTIONS[key]}",
        )
        up_planes = st.radio(
            "UP plane mode",
            list(UP_PLANE_DESCRIPTIONS),
            index=list(UP_PLANE_DESCRIPTIONS).index(up_base.get("plane_mode", "all16")),
            format_func=lambda key: f"{key}: {UP_PLANE_DESCRIPTIONS[key]}",
        )
    else:
        st.info("UP disabled.")
        up_variant = "gdml"
        up_planes = "all16"

with middle:
    st.subheader("FT")
    if "FT" in include:
        ft_base = base.get("ft", {}).get("stations", {})
        ft_t1 = st.selectbox("T1 modules", (8, 10, 12), index=(8, 10, 12).index(ft_base.get("T1", 10)))
        ft_t2 = st.selectbox("T2 modules", (8, 10, 12), index=(8, 10, 12).index(ft_base.get("T2", 10)))
        ft_t3 = st.selectbox("T3 modules", (8, 10, 12), index=(8, 10, 12).index(ft_base.get("T3", 12)))
        ft_acceptance = st.selectbox("FT acceptance", ("FT_withHole", "nominal"))
        st.caption("Every FT station is generated as X, U, V, X. U=-5 deg and V=+5 deg.")
    else:
        st.info("FT disabled.")
        ft_t1 = ft_t2 = ft_t3 = 10
        ft_acceptance = "FT_withHole"

with right:
    st.subheader("Other Detectors")
    for detector in DETECTOR_DEFAULT:
        if detector not in {"UP", "FT"}:
            state = "included" if detector in include else "disabled"
            st.write(f"**{detector}**: {state}")
            st.caption(DETECTOR_DESCRIPTIONS[detector])
    if "TV" in include:
        st.caption(
            "TV retains 100 sensitive-silicon surfaces at their actual Z positions. "
            "Separate left/right modules count as 50 logical layers."
        )

config = {
    "name": setup_name,
    "source": base.get("source", "ViewLHCb lower ROOT/GDML extracted surfaces"),
    "include": include,
}
if "UP" in include:
    config["up"] = {"variant": up_variant, "plane_mode": up_planes}
if "FT" in include:
    config["ft"] = {"stations": {"T1": ft_t1, "T2": ft_t2, "T3": ft_t3}}
    config["ft_acceptance"] = ft_acceptance
if "TV" in include:
    config["tv"] = dict(TV_CONFIG)

st.subheader("Configuration Preview")
st.code(json.dumps(config, indent=2, sort_keys=True), language="json")

build = st.button("Build Setup", type="primary", disabled=not setup_name or not include)

if build:
    with st.spinner("Building geometry setup, presentation, and optional 3D view..."):
        artifacts = build_setup(
            config,
            output_root=Path(output_root),
            skip_3d=not generate_3d,
        )
    st.success(f"Built {artifacts['output_dir']}")
    st.session_state["last_artifacts"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in artifacts.items()
        if key not in {"layers", "summary_rows"}
    }
    st.session_state["last_summary"] = artifacts["summary_rows"]

if "last_artifacts" in st.session_state:
    artifacts = st.session_state["last_artifacts"]
    st.subheader("Generated Artifacts")
    cols = st.columns(3)
    for index, key in enumerate(("output_dir", "layout", "setup_pickle", "summary", "presentation", "html")):
        value = artifacts.get(key)
        if value:
            cols[index % 3].write(f"**{key}**")
            cols[index % 3].code(value)

    st.subheader("Detector Summary")
    st.dataframe(summary_dataframe(st.session_state["last_summary"]), use_container_width=True)

    presentation = artifacts.get("presentation")
    if presentation and Path(presentation).exists():
        st.download_button(
            "Download geometry summary PDF",
            data=Path(presentation).read_bytes(),
            file_name=Path(presentation).name,
            mime="application/pdf",
        )

    html = artifacts.get("html")
    if html and Path(html).exists():
        with st.expander("Preview 3D HTML path"):
            st.write("Open this file in a browser:")
            st.code(html)

st.divider()
st.subheader("How to Use This Setup for Labels")
st.code(
    """PYTHONPATH=src python examples/augment_tracks_with_setup.py \\
  detector_setups/<setup-name> \\
  tracks.csv \\
  tracks_with_acceptance.csv \\
  --field magnetic_field_v5r0.root \\
  --label-suffix Run5""",
    language="bash",
)
