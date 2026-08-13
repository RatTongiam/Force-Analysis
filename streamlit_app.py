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
from biomechanics import apply_signal_filter, detect_phases_sequential, calculate_metrics
from pdf_generator import generate_pdf_report

st.set_page_config(layout="wide", page_title="Free JumpAnz Team - Prima Motion Tech")

st.title("Free JumpAnz Team - Biomechanics Analysis")
st.caption("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight")

st.sidebar.header("Data Import & Settings")

data_mode = st.sidebar.radio("Select Input Mode", [
    "MuscleLab CSV",
    "Single JSON (QTM)",
    "Dual TSV (QTM)", 
    "VALD ForceDecks CSV", 
    "C-Force Performance CSV"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Signal Filtering Options")

filter_type = st.sidebar.selectbox(
    "Select Filter Algorithm",
    ["Butterworth LPF", "Moving Average", "Raw Data (None)"],
    index=0
)

if filter_type == "Butterworth LPF":
    cutoff_freq = st.sidebar.slider("Cutoff Frequency (Hz)", min_value=5.0, max_value=100.0, value=10.0, step=5.0)
    filter_size = 15
elif filter_type == "Moving Average":
    filter_size = st.sidebar.selectbox("Window Size (Frames)", [1, 7, 15, 31], index=2)
    cutoff_freq = 10.0
else:
    cutoff_freq = 10.0
    filter_size = 1

threshold_alert = st.sidebar.number_input("Asymmetry Alert %", value=15.0, step=1.0)

dt, t, f_left, f_right, f_total = None, None, None, None, None
file_signature = None

if data_mode == "MuscleLab CSV":
    file_ml = st.sidebar.file_uploader("Upload MuscleLab CSV File (.csv)", type=["csv"])
    if file_ml:
        file_signature = f"ML_{file_ml.name}_{file_ml.size}"
        try:
            dt, t, f_left, f_right, f_total = parse_musclelab_csv(file_ml)
        except Exception as e:
            st.error(f"Error parsing MuscleLab CSV file: {e}")

elif data_mode == "Single JSON (QTM)":
    file_json = st.sidebar.file_uploader("Upload QTM JSON File (.json)", type=["json"])
    if file_json:
        file_signature = f"QTM_JSON_{file_json.name}_{file_json.size}"
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
                dt = float(p_left["dt"])
                t = np.arange(num_frames) * dt
                f_left = p_left["fz"][:num_frames]
                f_right = p_right["fz"][:num_frames]
                f_total = f_left + f_right
            else:
                st.error("QTM JSON must contain at least 2 Force Plates.")
        except Exception as e:
            st.error(f"Error parsing QTM JSON file: {e}")

elif data_mode == "Dual TSV (QTM)":
    file_a = st.sidebar.file_uploader("Upload Plate File A (.tsv)", type=["tsv"])
    side_a = st.sidebar.selectbox("Assign Side File A", ["left", "right"], index=0)
    file_b = st.sidebar.file_uploader("Upload Plate File B (.tsv)", type=["tsv"])
    side_b = st.sidebar.selectbox("Assign Side File B", ["left", "right"], index=1)
    
    if file_a and file_b:
        file_signature = f"TSV_{file_a.name}_{file_b.name}_{file_a.size}_{file_b.size}"
        try:
            dt, data_a, fz_idx_a = parse_tsv(file_a)
            _, data_b, fz_idx_b = parse_tsv(file_b)
            dt = float(dt)
            num_frames = min(len(data_a), len(data_b))
            t = np.arange(num_frames) * dt
            fz_a = np.abs(data_a[:num_frames, fz_idx_a])
            fz_b = np.abs(data_b[:num_frames, fz_idx_b])
            f_left = fz_a if side_a == "left" else fz_b
            f_right = fz_b if side_a == "left" else fz_a
            f_total = f_left + f_right
        except Exception as e:
            st.error(f"Error parsing Dual TSV files: {e}")

elif data_mode == "VALD ForceDecks CSV":
    file_vald = st.sidebar.file_uploader("Upload VALD ForceDecks File (.csv / .tsv)", type=["csv", "tsv"])
    if file_vald:
        file_signature = f"VALD_{file_vald.name}_{file_vald.size}"
        try:
            dt, t, f_left, f_right, f_total = parse_vald_forcedecks_exact(file_vald)
        except Exception as e:
            st.error(f"Error parsing VALD ForceDecks file: {e}")

elif data_mode == "C-Force Performance CSV":
    file_csv = st.sidebar.file_uploader("Upload Single CSV File (.csv)", type=["csv"])
    if file_csv:
        file_signature = f"CFORCE_{file_csv.name}_{file_csv.size}"
        try:
            dt, t, f_left, f_right, f_total = parse_single_csv_cforce(file_csv)
        except Exception as e:
            st.error(f"Error parsing Single CSV file: {e}")

if t is not None and f_total is not None and len(f_total) > 0:
    dt_val = float(dt) if dt is not None and dt > 0 else 0.001
    fs = 1.0 / dt_val
    sf = apply_signal_filter(f_total, filter_type=filter_type, cutoff=cutoff_freq, fs=fs, window_size=filter_size)
    sl = apply_signal_filter(f_left, filter_type=filter_type, cutoff=cutoff_freq, fs=fs, window_size=filter_size)
    sr = apply_signal_filter(f_right, filter_type=filter_type, cutoff=cutoff_freq, fs=fs, window_size=filter_size)
    
    n_samples = len(sf)
    quiet_samples = max(1, min(int(0.5 / dt_val), n_samples))

    sIdx_auto, bIdx_auto, zIdx_auto, tIdx_auto, lIdx_auto = detect_phases_sequential(
        t, f_total, dt_val, quiet_samples, filter_type=filter_type, cutoff=cutoff_freq
    )

    t_min = float(t[0])
    t_max = float(t[-1])

    is_file_changed = (st.session_state.get("current_file_sig") != file_signature)
    
    if is_file_changed or "t_start" not in st.session_state or st.sidebar.button("🔄 Reset Phases"):
        st.session_state.t_start = float(t[sIdx_auto])
        st.session_state.t_braking = float(t[bIdx_auto])
        st.session_state.t_split = float(t[zIdx_auto])
        st.session_state.t_takeoff = float(t[tIdx_auto])
        st.session_state.is_confirmed = False
        st.session_state.current_file_sig = file_signature

    # Strict Order Safeguard
    st.session_state.t_start = max(t_min, min(st.session_state.t_start, t_max - 3 * dt_val))
    st.session_state.t_braking = max(st.session_state.t_start + dt_val, min(st.session_state.t_braking, t_max - 2 * dt_val))
    st.session_state.t_split = max(st.session_state.t_braking + dt_val, min(st.session_state.t_split, t_max - dt_val))
    st.session_state.t_takeoff = max(st.session_state.t_split + dt_val, min(st.session_state.t_takeoff, t_max))

    t_start = st.session_state.t_start
    t_braking = st.session_state.t_braking
    t_split = st.session_state.t_split
    t_takeoff = st.session_state.t_takeoff

    t_curr_landing = min(t_max, t_takeoff + 0.5) 
    takeoff_idx = min(n_samples - 1, max(0, int(round((t_takeoff - t_min) / dt_val))))
    airborne_frames = np.where((np.arange(n_samples) >= takeoff_idx) & (sf < 25.0))[0]
    if len(airborne_frames) > 0:
        non_air = np.where((np.arange(n_samples) > airborne_frames[0]) & (sf >= 25.0))[0]
        if len(non_air) > 0:
            t_curr_landing = float(t[non_air[0]])

    crop_x_min = max(t_min, t_start - 1.0)
    crop_x_max = min(t_max, t_takeoff + 1.5)

    if not st.session_state.get("is_confirmed", False):
        display_x_min = t_min
        display_x_max = t_max
    else:
        display_x_min = crop_x_min
        display_x_max = crop_x_max

    # 1. GRAPH WITH HIGHLIGHT PHASES & PICTOGRAMS
    fig_force = go.Figure()
    fig_force.add_trace(go.Scatter(x=t, y=sl, name="Left Limb", line=dict(color='#818cf8', width=0.8)))
    fig_force.add_trace(go.Scatter(x=t, y=sr, name="Right Limb", line=dict(color='#f87171', width=0.8)))
    fig_force.add_trace(go.Scatter(x=t, y=sf, name="Total Force", line=dict(color='#4d2994', width=1.2)))

    # Phase Rectangles
    fig_force.add_vrect(x0=t_start, x1=t_braking, fillcolor="rgba(234, 179, 8, 0.12)", line_width=0)
    fig_force.add_vrect(x0=t_braking, x1=t_split, fillcolor="rgba(239, 68, 68, 0.12)", line_width=0)
    fig_force.add_vrect(x0=t_split, x1=t_takeoff, fillcolor="rgba(34, 197, 94, 0.12)", line_width=0)

    # Vertical Phase Boundary Lines
    fig_force.add_vline(x=t_start, line_width=1.5, line_dash="dash", line_color="#ca8a04")
    fig_force.add_vline(x=t_braking, line_width=1.5, line_dash="dash", line_color="#ef4444")
    fig_force.add_vline(x=t_split, line_width=1.5, line_dash="dash", line_color="#22c55e")
    fig_force.add_vline(x=t_takeoff, line_width=1.5, line_dash="dash", line_color="#dc2626")

    mid_unweight = (t_start + t_braking) / 2.0
    mid_brake = (t_braking + t_split) / 2.0
    mid_prop = (t_split + t_takeoff) / 2.0
    mid_flight = (t_takeoff + t_curr_landing) / 2.0
    mid_landing = min(t_max, t_curr_landing + 0.2)

    max_y = float(np.max(sf)) * 1.15 if len(sf) > 0 else 3000.0

    fig_force.add_annotation(x=mid_unweight, y=max_y * 0.98, text="Unweighting", showarrow=False, font=dict(size=11, color="#ca8a04", family="Arial Bold"))
    fig_force.add_annotation(x=mid_brake, y=max_y * 0.90, text="Braking", showarrow=False, font=dict(size=11, color="#ef4444", family="Arial Bold"))
    fig_force.add_annotation(x=mid_prop, y=max_y * 0.98, text="Propulsive", showarrow=False, font=dict(size=11, color="#22c55e", family="Arial Bold"))

    github_base = "https://raw.githubusercontent.com/RatTongiam/Force-Analysis/main"

    pictograms = [
        {"url": f"{github_base}/Standing.png", "x": max(display_x_min + 0.1, t_start - 0.2)},
        {"url": f"{github_base}/UP.png", "x": mid_unweight},
        {"url": f"{github_base}/BP.png", "x": mid_brake},
        {"url": f"{github_base}/PP.png", "x": mid_prop},
        {"url": f"{github_base}/FP.png", "x": mid_flight},
        {"url": f"{github_base}/LP.png", "x": mid_landing},
    ]

    for pic in pictograms:
        fig_force.add_layout_image(
            dict(
                source=pic["url"],
                xref="x",
                yref="y",
                x=pic["x"],
                y=max_y * 0.75,
                sizex=0.15,
                sizey=max_y * 0.22,
                xanchor="center",
                yanchor="bottom",
                layer="above"
            )
        )

    fig_force.update_layout(
        title="FORCE-TIME ANALYSIS & SUB-PHASES",
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        height=480,
        margin=dict(l=40, r=40, t=50, b=20)
    )
    fig_force.update_xaxes(range=[display_x_min, display_x_max])
    
    st.plotly_chart(fig_force, width="stretch")

    # 2. PHASE BOUNDARY TIMELINE CONTROLS (CLEAN SLIDER INTERFACE)
    st.markdown("##### 🎚️ Phase Boundary Timeline Controls")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        v1_min = t_min
        v1_max = max(v1_min + dt_val, t_braking - dt_val)
        val_1 = max(v1_min, min(st.session_state.t_start, v1_max))
        new_start = st.slider("1. Start Onset (s)", min_value=v1_min, max_value=v1_max, value=val_1, step=dt_val, format="%.3f", key="slide_1")

    with c2:
        v2_min = max(t_min + dt_val, new_start + dt_val)
        v2_max = max(v2_min + dt_val, t_split - dt_val)
        val_2 = max(v2_min, min(st.session_state.t_braking, v2_max))
        new_braking = st.slider("2. Braking (s)", min_value=v2_min, max_value=v2_max, value=val_2, step=dt_val, format="%.3f", key="slide_2")

    with c3:
        v3_min = max(t_min + 2 * dt_val, new_braking + dt_val)
        v3_max = max(v3_min + dt_val, t_takeoff - dt_val)
        val_3 = max(v3_min, min(st.session_state.t_split, v3_max))
        new_split = st.slider("3. Propulsive V=0 (s)", min_value=v3_min, max_value=v3_max, value=val_3, step=dt_val, format="%.3f", key="slide_3")

    with c4:
        v4_min = max(t_min + 3 * dt_val, new_split + dt_val)
        v4_max = max(v4_min + dt_val, t_max)
        val_4 = max(v4_min, min(st.session_state.t_takeoff, v4_max))
        new_takeoff = st.slider("4. Take-off (s)", min_value=v4_min, max_value=v4_max, value=val_4, step=dt_val, format="%.3f", key="slide_4")

    if (new_start, new_braking, new_split, new_takeoff) != (t_start, t_braking, t_split, t_takeoff):
        st.session_state.t_start = new_start
        st.session_state.t_braking = new_braking
        st.session_state.t_split = new_split
        st.session_state.t_takeoff = new_takeoff
        st.rerun()

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if not st.session_state.get("is_confirmed", False):
            if st.button("✅ Confirm Phases & Crop Graph", type="primary"):
                st.session_state.is_confirmed = True
                st.rerun()
        else:
            if st.button("✏️ Edit / Show Full View"):
                st.session_state.is_confirmed = False
                st.rerun()

    # Index Calculations
    sIdx = min(max(0, int(round((t_start - t[0]) / dt_val))), n_samples - 1)
    bIdx = min(max(0, int(round((t_braking - t[0]) / dt_val))), n_samples - 1)
    zIdx = min(max(0, int(round((t_split - t[0]) / dt_val))), n_samples - 1)
    tIdx = min(max(0, int(round((t_takeoff - t[0]) / dt_val))), n_samples - 1)
    lIdx = min(max(0, int(round((t_curr_landing - t[0]) / dt_val))), n_samples - 1)

    report = calculate_metrics(
        t, f_total, f_left, f_right, dt_val, sIdx, bIdx, zIdx, tIdx, lIdx, 
        filter_type=filter_type, cutoff=cutoff_freq
    )

    pdf_bytes = generate_pdf_report(
        report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, 
        threshold_alert=threshold_alert, crop_x_min=crop_x_min, crop_x_max=crop_x_max
    )
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Download A4 PDF Report",
        data=pdf_bytes,
        file_name="Prima_Motion_CMJ_Report.pdf",
        mime="application/pdf"
    )

    # 3. L/R ASYMMETRY % GRAPH
    max_sl_sr = np.maximum(sl, sr)
    deficits = np.where((sf >= 50) & (max_sl_sr > 0), ((sl - sr) / np.maximum(max_sl_sr, 1e-6)) * 100, 0)
    fig_deficit = go.Figure()

    fig_deficit.add_vrect(x0=t_start, x1=t_braking, fillcolor="rgba(234, 179, 8, 0.12)", line_width=0)
    fig_deficit.add_vrect(x0=t_braking, x1=t_split, fillcolor="rgba(239, 68, 68, 0.12)", line_width=0)
    fig_deficit.add_vrect(x0=t_split, x1=t_takeoff, fillcolor="rgba(34, 197, 94, 0.12)", line_width=0)

    fig_deficit.add_vline(x=t_start, line_width=1.5, line_dash="dash", line_color="#ca8a04")
    fig_deficit.add_vline(x=t_braking, line_width=1.5, line_dash="dash", line_color="#ef4444")
    fig_deficit.add_vline(x=t_split, line_width=1.5, line_dash="dash", line_color="#22c55e")
    fig_deficit.add_vline(x=t_takeoff, line_width=1.5, line_dash="dash", line_color="#dc2626")

    fig_deficit.add_hline(y=0, line_width=1.2, line_color="#6b7280")
    fig_deficit.add_trace(go.Scatter(x=t, y=deficits, name="Asymmetry", fill='tozeroy', fillcolor='rgba(77, 41, 148, 0.15)', line=dict(color='#4d2994', width=1.5)))
    fig_deficit.add_hrect(y0=-threshold_alert, y1=threshold_alert, fillcolor="rgba(34, 197, 94, 0.15)", line_width=0)

    fig_deficit.add_annotation(
        xref="paper", yref="y",
        x=0.01, y=38,
        text="<b>← Left Dominant (L > R)</b>",
        showarrow=False,
        font=dict(size=11, color="#818cf8"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#818cf8",
        borderwidth=1
    )

    fig_deficit.add_annotation(
        xref="paper", yref="y",
        x=0.01, y=-38,
        text="<b>← Right Dominant (R > L)</b>",
        showarrow=False,
        font=dict(size=11, color="#f87171"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#f87171",
        borderwidth=1
    )

    fig_deficit.update_layout(
        title="L/R ASYMMETRY % (Threshold Alert & Limb Dominance)", 
        xaxis_title="Time (s)", 
        yaxis_title="Deficit %", 
        yaxis_range=[-55, 55], 
        height=360,
        margin=dict(l=40, r=40, t=50, b=20)
    )
    fig_deficit.update_xaxes(range=[display_x_min, display_x_max])
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
