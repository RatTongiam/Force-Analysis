import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Free JumpAnz - Prima Motion Tech")

st.title("Free JumpAnz Team - Biomechanics Analysis")
st.subheader("PRIMA MOTION TECHNOLOGY")

# --- SIDEBAR: FILE IMPORT & CONTROL ---
st.sidebar.header("Data Import & Settings")

# 1. File Uploaders for Dual TSV
file_a = st.sidebar.file_uploader("Upload Plate File A (.tsv)", type=["tsv"])
side_a = st.sidebar.selectbox("Assign Side File A", ["left", "right"], index=0)

file_b = st.sidebar.file_uploader("Upload Plate File B (.tsv)", type=["tsv"])
side_b = st.sidebar.selectbox("Assign Side File B", ["left", "right"], index=1)

filter_size = st.sidebar.selectbox("Smoothing Filter", [1, 7, 15, 31], index=2)

# Function to parse TSV
def parse_tsv(uploaded_file):
    lines = uploaded_file.getvalue().decode('utf-8').splitlines()
    freq = 2000.0
    data = []
    for line in lines:
        line_str = line.strip()
        if line_str.startswith("FREQUENCY"):
            parts = line_str.split('\t')
            if len(parts) > 1:
                freq = float(parts[1])
        parts = line_str.split('\t')
        if len(parts) >= 3:
            try:
                vals = [float(p) for p in parts]
                data.append(vals)
            except ValueError:
                pass
    return freq, np.array(data)

if file_a and file_b:
    freq_a, data_a = parse_tsv(file_a)
    freq_b, data_b = parse_tsv(file_b)
    
    dt = 1.0 / freq_a
    num_frames = min(len(data_a), len(data_b))
    t = np.arange(num_frames) * dt
    
    # Extract Col 2 (Fz - Vertical Force)
    fz_a = np.abs(data_a[:num_frames, 2])
    fz_b = np.abs(data_b[:num_frames, 2])
    
    f_left = fz_a if side_a == "left" else fz_b
    f_right = fz_b if side_a == "left" else fz_a
    f_total = f_left + f_right
    
    # Moving Average Smoothing
    if filter_size > 1:
        f_total = pd.Series(f_total).rolling(window=filter_size, center=True, min_periods=1).mean().values
        f_left = pd.Series(f_left).rolling(window=filter_size, center=True, min_periods=1).mean().values
        f_right = pd.Series(f_right).rolling(window=filter_size, center=True, min_periods=1).mean().values

    # Sequential Sub-phase Boundaries Detection
    quiet_samples = int(0.5 / dt)
    bw = np.mean(f_total[:quiet_samples])
    mass = bw / 9.80665
    
    # Flight Phase Detection
    flight_indices = np.where(f_total < 30.0)[0]
    tIdx = flight_indices[0] if len(flight_indices) > 0 else len(f_total) - 1
    
    # Unweighting Onset (sIdx)
    sIdx = np.where(f_total[:tIdx] < (bw - 5 * np.std(f_total[:quiet_samples])))[0]
    sIdx = sIdx[0] if len(sIdx) > 0 else 0
    
    # Min Force & Braking Onset (bIdx = Min Force Position)
    peakForceIdx = sIdx + np.argmax(f_total[sIdx:tIdx])
    bIdx = sIdx + np.argmin(f_total[sIdx:peakForceIdx])
    
    # Propulsive Onset (zIdx - Velocity = 0)
    vel = np.cumsum((f_total[sIdx:tIdx] - bw) / mass) * dt
    zero_vel_idx = np.where(vel[bIdx - sIdx:] >= 0)[0]
    zIdx = (bIdx + zero_vel_idx[0]) if len(zero_vel_idx) > 0 else bIdx

    # Sidebar Manual Adjustment Sliders
    st.sidebar.markdown("---")
    st.sidebar.subheader("Phase Adjustment (Interactive)")
    
    t_start = st.sidebar.slider("Start (Unweighting Onset)", 0.0, float(t[-1]), float(t[sIdx]), step=0.005)
    t_braking = st.sidebar.slider("Braking Onset (Min Force)", 0.0, float(t[-1]), float(t[bIdx]), step=0.005)
    t_split = st.sidebar.slider("Propulsive Onset (V=0)", 0.0, float(t[-1]), float(t[zIdx]), step=0.005)
    t_takeoff = st.sidebar.slider("Take-off", 0.0, float(t[-1]), float(t[tIdx]), step=0.005)

    # Convert times back to indices
    idx_start = int(t_start / dt)
    idx_braking = int(t_braking / dt)
    idx_split = int(t_split / dt)
    idx_takeoff = int(t_takeoff / dt)

    # Plot Force-Time Curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=f_left, name="Left Limb", line=dict(color='#818cf8', width=2)))
    fig.add_trace(go.Scatter(x=t, y=f_right, name="Right Limb", line=dict(color='#f87171', width=2)))
    fig.add_trace(go.Scatter(x=t, y=f_total, name="Total Force", line=dict(color='#4d2994', width=3.5)))

    # Phase Shading
    max_f = np.max(f_total)
    fig.add_vrect(x0=t_start, x1=t_braking, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_vrect(x0=t_braking, x1=t_split, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_vrect(x0=t_split, x1=t_takeoff, fillcolor="green", opacity=0.1, line_width=0)

    # Vertical Boundary Lines
    fig.add_vline(x=t_start, line_dash="dash", line_color="orange")
    fig.add_vline(x=t_braking, line_dash="dash", line_color="red")
    fig.add_vline(x=t_split, line_dash="dash", line_color="green")
    fig.add_vline(x=t_takeoff, line_dash="dash", line_color="darkred")

    fig.update_layout(title="FORCE-TIME ANALYSIS & SUB-PHASES", xaxis_title="Time (s)", yaxis_title="Force (N)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Calculations & Metrics Table
    unweight_dur = t_braking - t_start
    braking_dur = t_split - t_braking
    propulsive_dur = t_takeoff - t_split

    st.subheader("Biomechanical Summary Metrics")
    metrics_df = pd.DataFrame({
        "Phase Metric": ["Unweighting Duration (s)", "Braking Duration (s)", "Propulsive Duration (s)", "Peak Propulsive Force (N)"],
        "Value": [f"{unweight_dur:.3f}", f"{braking_dur:.3f}", f"{propulsive_dur:.3f}", f"{np.max(f_total[idx_split:idx_takeoff+1]):.1f}"]
    })
    st.table(metrics_df)
else:
    st.info("Please upload both Plate File A and Plate File B (.tsv) to start analysis.")
