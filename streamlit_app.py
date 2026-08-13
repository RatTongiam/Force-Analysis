import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from parsers import (
    parse_tsv, 
    parse_vald_forcedecks_exact, 
    parse_qtm_json, 
    parse_single_csv_cforce,
    parse_musclelab_csv
)
from biomechanics import moving_average, detect_phases_sequential, calculate_metrics
from pdf_generator import generate_pdf_report

st.set_page_config(layout="wide", page_title="Free JumpAnz Team - Prima Motion Tech")

st.title("Free JumpAnz Team - Biomechanics Analysis")
st.caption("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight")

st.sidebar.header("Data Import & Settings")

data_mode = st.sidebar.radio("Select Input Mode", [
    "Dual TSV (Plate A + B)", 
    "VALD ForceDecks (CSV/TSV)", 
    "Single JSON (QTM)", 
    "Single CSV (C-Force)",
    "MuscleLab CSV (.csv)"
])

filter_size = st.sidebar.selectbox("Smoothing Filter", [1, 7, 15, 31], index=2)
threshold_alert = st.sidebar.number_input("Asymmetry Alert %", value=15.0, step=1.0)

dt, t, f_left, f_right, f_total = None, None, None, None, None

if data_mode == "Dual TSV (Plate A + B)":
    file_a = st.sidebar.file_uploader("Upload Plate File A (.tsv)", type=["tsv"])
    side_a = st.sidebar.selectbox("Assign Side File A", ["left", "right"], index=0)
    file_b = st.sidebar.file_uploader("Upload Plate File B (.tsv)", type=["tsv"])
    side_b = st.sidebar.selectbox("Assign Side File B", ["left", "right"], index=1)
    
    if file_a and file_b:
        dt, data_a = parse_tsv(file_a)
        _, data_b = parse_tsv(file_b)
        num_frames = min(len(data_a), len(data_b))
        t = np.arange(num_frames) * dt
        fz_a = np.abs(data_a[:num_frames, 2])
        fz_b = np.abs(data_b[:num_frames, 2])
        f_left = fz_a if side_a == "left" else fz_b
        f_right = fz_b if side_a == "left" else fz_a
        f_total = f_left + f_right

elif data_mode == "VALD ForceDecks (CSV/TSV)":
    file_vald = st.sidebar.file_uploader("Upload VALD ForceDecks File (.csv / .tsv)", type=["csv", "tsv"])
    if file_vald:
        try:
            dt, t, f_left, f_right, f_total = parse_vald_forcedecks_exact(file_vald)
        except Exception as e:
            st.error(f"Error parsing VALD ForceDecks file: {e}")

elif data_mode == "Single JSON (QTM)":
    file_json = st.sidebar.file_uploader("Upload QTM JSON File (.json)", type=["json"])
    if file_json:
        try:
            plates = parse_qtm_json(file_json)
            if len(plates) >= 2:
                st.sidebar.markdown("---")
                st.sidebar.subheader("Force Plate Mapping")
                plate_names = [p["name"] for p in plates]
                
                if len(plates) == 2:
                    left_name = st.sidebar.selectbox("Left Limb Plate", plate_names, index=0)
                    right_names = [n for n in plate_names if n != left_name]
                    right_name = st.sidebar.selectbox("Right Limb Plate", plate_names, index=1 if len(plate_names) > 1 else 0)
                    
                    swap_sides = st.sidebar.toggle("🔀 Swap Left / Right Sides", value=False)
                    if swap_sides:
                        left_name, right_name = right_name, left_name
                else:
                    left_name = st.sidebar.selectbox("Left Limb Plate", plate_names, index=0)
                    right_name = st.sidebar.selectbox("Right Limb Plate", plate_names, index=min(1, len(plate_names)-1))
                
                p_left = next(p for p in plates if p["name"] == left_name)
                p_right = next(p for p in plates if p["name"] == right_name)
                
                num_frames = min(len(p_left["fz"]), len(p_right["fz"]))
                dt = p_left["dt"]
                t = np.arange(num_frames) * dt
                f_left = p_left["fz"][:num_frames]
                f_right = p_right["fz"][:num_frames]
                f_total = f_left + f_right
            else:
                st.error("QTM JSON must contain at least 2 Force Plates.")
        except Exception as e:
            st.error(f"Error parsing QTM JSON file: {e}")

elif data_mode == "Single CSV (C-Force)":
    file_csv = st.sidebar.file_uploader("Upload Single CSV File (.csv)", type=["csv"])
    if file_csv:
        try:
            dt, t, f_left, f_right, f_total = parse_single_csv_cforce(file_csv)
        except Exception as e:
            st.error(f"Error parsing Single CSV file: {e}")

elif data_mode == "MuscleLab CSV (.csv)":
    file_ml = st.sidebar.file_uploader("Upload MuscleLab CSV File (.csv)", type=["csv"])
    if file_ml:
        try:
            dt, t, f_left, f_right, f_total = parse_musclelab_csv(file_ml)
        except Exception as e:
            st.error(f"Error parsing MuscleLab CSV file: {e}")

if t is not None and f_total is not None and len(f_total) > 0:
    sf = moving_average(f_total, filter_size)
    sl = moving_average(f_left, filter_size)
    sr = moving_average(f_right, filter_size)
    
    n_samples = len(sf)
    quiet_samples = max(1, min(int(0.5 / dt), n_samples))

    sIdx_auto, bIdx_auto, zIdx_auto, tIdx_auto, lIdx_auto = detect_phases_sequential(t, sf, dt, quiet_samples)

    if "t_start" not in st.session_state or st.sidebar.button("🔄 Reset Phases"):
        st.session_state.t_start = float(t[sIdx_auto])
        st.session_state.t_braking = float(t[bIdx_auto])
        st.session_state.t_split = float(t[zIdx_auto])
        st.session_state.t_takeoff = float(t[tIdx_auto])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Phase Control")
    st.sidebar.info("💡 คลิกลากเส้นแนวตั้งบนกราฟเพื่อปรับแต่งเฟส (ระบบล็อกลำดับเฟสให้อัตโนมัติ)")

    fig_force = go.Figure()
    fig_force.add_trace(go.Scatter(x=t, y=sl, name="Left Limb", line=dict(color='#818cf8', width=0.8)))
    fig_force.add_trace(go.Scatter(x=t, y=sr, name="Right Limb", line=dict(color='#f87171', width=0.8)))
    fig_force.add_trace(go.Scatter(x=t, y=sf, name="Total Force", line=dict(color='#4d2994', width=1.2)))

    fig_force.update_layout(
        title="FORCE-TIME ANALYSIS & SUB-PHASES",
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        height=450,
        shapes=[
            dict(type="line", x0=st.session_state.t_start, x1=st.session_state.t_start, y0=0, y1=1, yref="paper", line=dict(color="#ca8a04", width=2, dash="dash")),
            dict(type="line", x0=st.session_state.t_braking, x1=st.session_state.t_braking, y0=0, y1=1, yref="paper", line=dict(color="#ef4444", width=2, dash="dash")),
            dict(type="line", x0=st.session_state.t_split, x1=st.session_state.t_split, y0=0, y1=1, yref="paper", line=dict(color="#22c55e", width=2, dash="dash")),
            dict(type="line", x0=st.session_state.t_takeoff, x1=st.session_state.t_takeoff, y0=0, y1=1, yref="paper", line=dict(color="#dc2626", width=2, dash="dash")),
        ]
    )
   # ดักจับ Event จาก Plotly โดยตรวจสอบว่าเป็น dict ก่อนเสมอ
    if isinstance(event, dict) and "edits" in event and isinstance(event["edits"], dict) and "shape" in event["edits"]:
        shape_info = event["edits"]["shape"]
        if isinstance(shape_info, dict):
            shape_idx = shape_info.get("index")
            new_x = shape_info.get("x0")
            if new_x is not None:
                min_t, max_t = float(t[0]), float(t[-1])
                new_x = max(min_t, min(max_t, new_x))
                
                if shape_idx == 0:
                    st.session_state.t_start = min(new_x, st.session_state.t_braking)
                elif shape_idx == 1:
                    st.session_state.t_braking = max(st.session_state.t_start, min(new_x, st.session_state.t_split))
                elif shape_idx == 2:
                    st.session_state.t_split = max(st.session_state.t_braking, min(new_x, st.session_state.t_takeoff))
                elif shape_idx == 3:
                    st.session_state.t_takeoff = max(st.session_state.t_split, new_x)
                st.rerun()
    t_split = st.session_state.t_split
    t_takeoff = st.session_state.t_takeoff

    sIdx = min(max(0, int(round((t_start - t[0]) / dt))), n_samples - 1)
    bIdx = min(max(0, int(round((t_braking - t[0]) / dt))), n_samples - 1)
    zIdx = min(max(0, int(round((t_split - t[0]) / dt))), n_samples - 1)
    tIdx = min(max(0, int(round((t_takeoff - t[0]) / dt))), n_samples - 1)
    
    airborne_frames = np.where((np.arange(n_samples) >= tIdx) & (sf < 25.0))[0]
    if len(airborne_frames) > 0:
        first_air = airborne_frames[0]
        non_air = np.where((np.arange(n_samples) > first_air) & (sf >= 25.0))[0]
        lIdx = non_air[0] if len(non_air) > 0 else n_samples - 1
    else:
        lIdx = n_samples - 1

    report = calculate_metrics(t, sf, sl, sr, dt, sIdx, bIdx, zIdx, tIdx, lIdx)

    pdf_bytes = generate_pdf_report(report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff)
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Download A4 PDF Report",
        data=pdf_bytes,
        file_name="Prima_Motion_CMJ_Report.pdf",
        mime="application/pdf"
    )

    deficits = np.where(sf >= 50, ((sl - sr) / np.maximum(sl, sr)) * 100, 0)
    fig_deficit = go.Figure()
    fig_deficit.add_trace(go.Scatter(x=t, y=deficits, name="Asymmetry", fill='tozeroy', fillcolor='rgba(77, 41, 148, 0.15)', line=dict(color='#4d2994', width=1.5)))
    
    fig_deficit.add_hrect(y0=-threshold_alert, y1=threshold_alert, fillcolor="rgba(34, 197, 94, 0.15)", line_width=0)
    fig_deficit.update_layout(title="L/R ASYMMETRY % (Threshold Alert)", xaxis_title="Time (s)", yaxis_title="Deficit %", yaxis_range=[-55, 55], height=260)
    fig_deficit.update_xaxes(range=[t[0], t[-1]])
    st.plotly_chart(fig_deficit, width="stretch")

    st.markdown("### Standard Biomechanical Analysis Report (Anicic et al., 2023)")
    table_rows = []
    for phase_name, metrics in report.items():
        table_rows.append({"Biomechanical Metric": f"=== {phase_name.upper()} ===", "Left": "", "Right": "", "TOTAL": "", "Deficit %": ""})
        for metric_name, vals in metrics.items():
            table_rows.append({
                "Biomechanical Metric": metric_name,
                "Left": vals["Left"],
                "Right": vals["Right"],
                "TOTAL": vals["Total"],
                "Deficit %": vals["Deficit"]
            })
    
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    st.download_button(
        label="📥 Download A4 PDF Report",
        data=pdf_bytes,
        file_name="Prima_Motion_CMJ_Report.pdf",
        mime="application/pdf",
        key="main_download_btn"
    )
else:
    st.info("Please upload data file(s) to begin analysis.")
