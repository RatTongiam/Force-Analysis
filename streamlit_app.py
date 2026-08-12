import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Free JumpAnz Team - Prima Motion Tech")

st.title("Free JumpAnz Team - Biomechanics Analysis")
st.caption("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight")

# --- SIDEBAR: FILE IMPORT & CONTROL ---
st.sidebar.header("Data Import & Settings")

# 1. File Uploaders for Dual TSV
file_a = st.sidebar.file_uploader("Upload Plate File A (.tsv)", type=["tsv"])
side_a = st.sidebar.selectbox("Assign Side File A", ["left", "right"], index=0)

file_b = st.sidebar.file_uploader("Upload Plate File B (.tsv)", type=["tsv"])
side_b = st.sidebar.selectbox("Assign Side File B", ["left", "right"], index=1)

filter_size = st.sidebar.selectbox("Smoothing Filter", [1, 7, 15, 31], index=2)
threshold_alert = st.sidebar.number_input("Asymmetry Alert %", value=15.0, step=1.0)

# Helper Functions
def parse_tsv(uploaded_file):
    lines = uploaded_file.getvalue().decode('utf-8').splitlines()
    freq = 2000.0
    data = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if line_str.startswith("FREQUENCY"):
            parts = line_str.split('\t')
            if len(parts) > 1:
                try:
                    freq = float(parts[1])
                except ValueError:
                    pass
            continue
        parts = line_str.split('\t')
        if len(parts) >= 3:
            try:
                vals = [float(p) for p in parts]
                data.append(vals)
            except ValueError:
                pass
    return freq, np.array(data)

def moving_average(arr, window):
    if window <= 1:
        return arr
    return pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean().values

def calc_avg(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0

def calc_impulse(arr, dt):
    return np.sum(arr) * dt

def calc_net_impulse(arr, base, dt):
    return np.sum(arr - base) * dt

# Main Processing Block
if file_a and file_b:
    freq_a, data_a = parse_tsv(file_a)
    freq_b, data_b = parse_tsv(file_b)
    
    dt = 1.0 / freq_a
    num_frames = min(len(data_a), len(data_b))
    t = np.arange(num_frames) * dt
    
    # Col Index 2 = Fz (Vertical Force)
    fz_a = np.abs(data_a[:num_frames, 2])
    fz_b = np.abs(data_b[:num_frames, 2])
    
    f_left = fz_a if side_a == "left" else fz_b
    f_right = fz_b if side_a == "left" else fz_a
    f_total = f_left + f_right
    
    # Apply Smoothing Filter
    sf = moving_average(f_total, filter_size)
    sl = moving_average(f_left, filter_size)
    sr = moving_average(f_right, filter_size)
    
    g = 9.80665
    quiet_samples = max(1, int(0.5 / dt))
    
    # 1. Quiet Standing Calibration
    bw = np.mean(sf[:quiet_samples])
    bw_left = np.mean(sl[:quiet_samples])
    bw_right = np.mean(sr[:quiet_samples])
    
    force_sd = np.std(sf[:quiet_samples])
    mass = bw / g
    mass_left = bw_left / g
    mass_right = bw_right / g
    
    # --- AUTO DETECT SUB-PHASE BOUNDARIES (SEQUENTIAL LOGIC) ---
    flight_threshold = 30.0
    flight_start = -1
    flight_end = -1

    for i in range(quiet_samples, len(sf) - 3):
        if sf[i] < flight_threshold and sf[i+1] < flight_threshold and sf[i+2] < flight_threshold:
            flight_start = i
            break

    if flight_start != -1:
        tIdx_auto = flight_start
        for i in range(flight_start, len(sf)):
            if sf[i] >= flight_threshold:
                flight_end = i
                break
        lIdx_auto = flight_end if flight_end != -1 else len(sf) - 1
    else:
        tIdx_auto = len(sf) - 1
        lIdx_auto = len(sf) - 1

    window_size = max(1, int(0.03 / dt))
    threshold_dev = max(force_sd * 5, bw * 0.025)

    sIdx_auto = 0
    for i in range(quiet_samples, tIdx_auto - window_size):
        is_exceeded = True
        for j in range(window_size):
            if abs(sf[i + j] - bw) <= threshold_dev:
                is_exceeded = False
                break
        if is_exceeded:
            sIdx_auto = i
            break

    peak_force_idx = sIdx_auto + np.argmax(sf[sIdx_auto:tIdx_auto + 1])
    min_force_idx = sIdx_auto + np.argmin(sf[sIdx_auto:peak_force_idx])
    bIdx_auto = min_force_idx

    vel_temp = np.cumsum((sf[sIdx_auto:tIdx_auto + 1] - bw) / mass) * dt
    zero_vel_matches = np.where(vel_temp[bIdx_auto - sIdx_auto:] >= 0)[0]
    zIdx_auto = (bIdx_auto + zero_vel_matches[0]) if len(zero_vel_matches) > 0 else bIdx_auto

    # Sidebar Sliders for Adjusting Phase Times
    st.sidebar.markdown("---")
    st.sidebar.subheader("Phase Adjustment")
    
    t_start = st.sidebar.slider("Start (Unweighting Onset)", 0.0, float(t[-1]), float(t[sIdx_auto]), step=0.005)
    t_braking = st.sidebar.slider("Braking Onset (Min Force)", 0.0, float(t[-1]), float(t[bIdx_auto]), step=0.005)
    t_split = st.sidebar.slider("Propulsive Onset (V=0)", 0.0, float(t[-1]), float(t[zIdx_auto]), step=0.005)
    t_takeoff = st.sidebar.slider("Take-off", 0.0, float(t[-1]), float(t[tIdx_auto]), step=0.005)

    sIdx = max(0, int(t_start / dt))
    bIdx = max(0, int(t_braking / dt))
    zIdx = max(0, int(t_split / dt))
    tIdx = max(0, int(t_takeoff / dt))
    
    lIdx_matches = np.where((t > t_takeoff + 0.05) & (sf > 30.0))[0]
    lIdx = lIdx_matches[0] if len(lIdx_matches) > 0 else len(sf) - 1

    # Kinematic Arrays Integration
    vel_total = np.zeros(len(sf))
    vel_l = np.zeros(len(sf))
    vel_r = np.zeros(len(sf))
    disp_total = np.zeros(len(sf))

    cV = 0.0
    cVL = 0.0
    cVR = 0.0
    cD = 0.0

    for i in range(sIdx, tIdx + 1):
        cV += ((sf[i] - bw) / mass) * dt
        cVL += ((sl[i] - bw_left) / mass_left) * dt
        cVR += ((sr[i] - bw_right) / mass_right) * dt
        vel_total[i] = cV
        vel_l[i] = cVL
        vel_r[i] = cVR
        cD += cV * dt
        disp_total[i] = cD

    contraction_time = max(dt, t[tIdx] - t[sIdx])
    unweight_dur = t[bIdx] - t[sIdx]
    braking_dur = t[zIdx] - t[bIdx]
    propulsive_dur = t[tIdx] - t[zIdx]

    # Sub-phase Arrays
    brak_f = sf[bIdx:zIdx + 1]
    brak_fl = sl[bIdx:zIdx + 1]
    brak_fr = sr[bIdx:zIdx + 1]
    brak_v = vel_total[bIdx:zIdx + 1]
    brak_p = brak_f * brak_v

    avg_brak_f = calc_avg(brak_f)
    avg_brak_fl = calc_avg(brak_fl)
    avg_brak_fr = calc_avg(brak_fr)
    avg_brak_p = calc_avg(brak_p)
    avg_brak_v = calc_avg(brak_v)

    peak_brak_f = np.max(brak_f) if len(brak_f) > 0 else 0.0
    peak_brak_fl = np.max(brak_fl) if len(brak_fl) > 0 else 0.0
    peak_brak_fr = np.max(brak_fr) if len(brak_fr) > 0 else 0.0
    peak_brak_p = np.min(brak_p) if len(brak_p) > 0 else 0.0

    brak_impulse = calc_impulse(brak_f, dt)
    brak_impulse_l = calc_impulse(brak_fl, dt)
    brak_impulse_r = calc_impulse(brak_fr, dt)
    brak_net_impulse = calc_net_impulse(brak_f, bw, dt)
    brak_rfd = (peak_brak_f - sf[bIdx]) / braking_dur if braking_dur > 0 else 0.0
    brak_rfd_l = (peak_brak_fl - sl[bIdx]) / braking_dur if braking_dur > 0 else 0.0
    brak_rfd_r = (peak_brak_fr - sr[bIdx]) / braking_dur if braking_dur > 0 else 0.0

    prop_f = sf[zIdx:tIdx + 1]
    prop_fl = sl[zIdx:tIdx + 1]
    prop_fr = sr[zIdx:tIdx + 1]
    prop_v = vel_total[zIdx:tIdx + 1]
    prop_p = prop_f * prop_v

    avg_prop_f = calc_avg(prop_f)
    avg_prop_fl = calc_avg(prop_fl)
    avg_prop_fr = calc_avg(prop_fr)
    avg_prop_p = calc_avg(prop_p)
    avg_prop_v = calc_avg(prop_v)

    peak_prop_f = np.max(prop_f) if len(prop_f) > 0 else 0.0
    peak_prop_fl = np.max(prop_fl) if len(prop_fl) > 0 else 0.0
    peak_prop_fr = np.max(prop_fr) if len(prop_fr) > 0 else 0.0
    peak_prop_p = np.max(prop_p) if len(prop_p) > 0 else 0.0

    prop_impulse = calc_impulse(prop_f, dt)
    prop_impulse_l = calc_impulse(prop_fl, dt)
    prop_impulse_r = calc_impulse(prop_fr, dt)
    prop_net_impulse = calc_net_impulse(prop_f, bw, dt)

    # Landing Metrics
    avg_land_f = avg_land_fl = avg_land_fr = 0.0
    peak_land_f = peak_land_fl = peak_land_fr = 0.0
    land_stiffness = 0.0
    time_to_stab = 0.0

    if lIdx > tIdx and lIdx < len(sf):
        land_end_idx = min(len(sf), lIdx + int(0.5 / dt))
        land_f = sf[lIdx:land_end_idx]
        land_fl = sl[lIdx:land_end_idx]
        land_fr = sr[lIdx:land_end_idx]

        avg_land_f = calc_avg(land_f)
        avg_land_fl = calc_avg(land_fl)
        avg_land_fr = calc_avg(land_fr)
        peak_land_f = np.max(land_f) if len(land_f) > 0 else 0.0
        peak_land_fl = np.max(land_fl) if len(land_fl) > 0 else 0.0
        peak_land_fr = np.max(land_fr) if len(land_fr) > 0 else 0.0

        cm_depth = abs(disp_total[zIdx] - disp_total[sIdx]) * 100.0
        land_stiffness = (peak_land_f / (cm_depth / 100.0)) if cm_depth > 0 else 0.0

        stab_window = int(0.5 / dt)
        for i in range(lIdx, len(sf) - stab_window):
            if np.all(np.abs(sf[i:i + stab_window] - bw) <= bw * 0.05):
                time_to_stab = t[i] - t[lIdx]
                break

    # Build Complete Report Dictionary
    report = {
        "Unweighting Phase": {
            "Unweighting Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{unweight_dur:.2f}"},
            "Unweighting Phase % (%)": {"Left": "-", "Right": "-", "Total": f"{(unweight_dur / contraction_time) * 100:.1f}"}
        },
        "Braking Phase": {
            "Avg. Braking Force (N)": {"Left": f"{avg_brak_fl:.0f}", "Right": f"{avg_brak_fr:.0f}", "Total": f"{avg_brak_f:.0f}"},
            "Avg. Braking Power (W)": {"Left": "-", "Right": "-", "Total": f"{abs(avg_brak_p):.0f}"},
            "Avg. Braking Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{abs(avg_brak_v):.2f}"},
            "Avg. Relative Braking Force (N/kg)": {"Left": "-", "Right": "-", "Total": f"{avg_brak_f / mass:.1f}"},
            "Avg. Relative Braking Power (W/kg)": {"Left": "-", "Right": "-", "Total": f"{abs(avg_brak_p) / mass:.1f}"},
            "Braking Impulse (N·s)": {"Left": f"{brak_impulse_l:.0f}", "Right": f"{brak_impulse_r:.0f}", "Total": f"{brak_impulse:.0f}"},
            "Braking Net Impulse (N·s)": {"Left": "-", "Right": "-", "Total": f"{brak_net_impulse:.0f}"},
            "Braking Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{braking_dur:.2f}"},
            "Braking Phase % (%)": {"Left": "-", "Right": "-", "Total": f"{(braking_dur / contraction_time) * 100:.1f}"},
            "Braking RFD (N/s)": {"Left": f"{brak_rfd_l:.0f}", "Right": f"{brak_rfd_r:.0f}", "Total": f"{brak_rfd:.0f}"},
            "Peak Braking Force (N)": {"Left": f"{peak_brak_fl:.0f}", "Right": f"{peak_brak_fr:.0f}", "Total": f"{peak_brak_f:.0f}"},
            "Peak Braking Power (W)": {"Left": "-", "Right": "-", "Total": f"{abs(peak_brak_p):.0f}"},
            "Peak Relative Braking Force (N/kg)": {"Left": "-", "Right": "-", "Total": f"{peak_brak_f / mass:.1f}"},
            "Peak Relative Braking Power (W/kg)": {"Left": "-", "Right": "-", "Total": f"{abs(peak_brak_p) / mass:.1f}"},
            "Relative Braking Impulse (N·s/kg)": {"Left": "-", "Right": "-", "Total": f"{brak_impulse / mass:.2f}"},
            "Relative Braking Net Impulse (N·s/kg)": {"Left": "-", "Right": "-", "Total": f"{brak_net_impulse / mass:.2f}"},
            "L/R Braking Impulse Ratio": {"Left": "-", "Right": "-", "Total": f"{brak_impulse_l / brak_impulse_r:.2f}" if brak_impulse_r > 0 else "-"}
        },
        "Propulsive Phase": {
            "Avg. Propulsive Force (N)": {"Left": f"{avg_prop_fl:.0f}", "Right": f"{avg_prop_fr:.0f}", "Total": f"{avg_prop_f:.0f}"},
            "Avg. Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{avg_prop_p:.0f}"},
            "Avg. Propulsive Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{avg_prop_v:.2f}"},
            "Avg. Relative Propulsive Power (W/kg)": {"Left": "-", "Right": "-", "Total": f"{avg_prop_p / mass:.1f}"},
            "Peak Propulsive Force (N)": {"Left": f"{peak_prop_fl:.0f}", "Right": f"{peak_prop_fr:.0f}", "Total": f"{peak_prop_f:.0f}"},
            "Peak Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{peak_prop_p:.0f}"},
            "Peak Relative Propulsive Force (N/kg)": {"Left": "-", "Right": "-", "Total": f"{peak_prop_f / mass:.1f}"},
            "Peak Relative Propulsive Power (W/kg)": {"Left": "-", "Right": "-", "Total": f"{peak_prop_p / mass:.1f}"},
            "Propulsive Impulse (N·s)": {"Left": f"{prop_impulse_l:.0f}", "Right": f"{prop_impulse_r:.0f}", "Total": f"{prop_impulse:.0f}"},
            "Propulsive Net Impulse (N·s)": {"Left": "-", "Right": "-", "Total": f"{prop_net_impulse:.0f}"},
            "Propulsive Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{propulsive_dur:.2f}"},
            "Propulsive Phase % (%)": {"Left": "-", "Right": "-", "Total": f"{(propulsive_dur / contraction_time) * 100:.1f}"},
            "Relative Propulsive Impulse (N·s/kg)": {"Left": "-", "Right": "-", "Total": f"{prop_impulse / mass:.2f}"},
            "Relative Propulsive Net Impulse (N·s/kg)": {"Left": "-", "Right": "-", "Total": f"{prop_net_impulse / mass:.2f}"},
            "L/R Propulsive Impulse Ratio": {"Left": "-", "Right": "-", "Total": f"{prop_impulse_l / prop_impulse_r:.2f}" if prop_impulse_r > 0 else "-"}
        },
        "Landing Phase": {
            "Avg. Landing Force (N)": {"Left": f"{avg_land_fl:.0f}" if avg_land_f > 0 else "N/A", "Right": f"{avg_land_fr:.0f}" if avg_land_f > 0 else "N/A", "Total": f"{avg_land_f:.0f}" if avg_land_f > 0 else "N/A"},
            "Landing Stiffness (N/m)": {"Left": "-", "Right": "-", "Total": f"{land_stiffness:.0f}" if land_stiffness > 0 else "N/A"},
            "Peak Landing Force (N)": {"Left": f"{peak_land_fl:.0f}" if peak_land_f > 0 else "N/A", "Right": f"{peak_land_fr:.0f}" if peak_land_f > 0 else "N/A", "Total": f"{peak_land_f:.0f}" if peak_land_f > 0 else "N/A"},
            "Relative Peak Landing Force (N/kg)": {"Left": "-", "Right": "-", "Total": f"{peak_land_f / mass:.1f}" if peak_land_f > 0 else "N/A"},
            "Time to Stabilization (s)": {"Left": "-", "Right": "-", "Total": f"{time_to_stab:.2f}" if time_to_stab > 0 else "N/A"},
            "L/R Landing Force Ratio": {"Left": "-", "Right": "-", "Total": f"{peak_land_fl / peak_land_fr:.2f}" if peak_land_fr > 0 else "N/A"}
        }
    }

    # Display Force-Time Chart
    max_f = np.max(sf)
    fig_force = go.Figure()
    fig_force.add_trace(go.Scatter(x=t, y=sl, name="Left Limb", line=dict(color='#818cf8', width=2)))
    fig_force.add_trace(go.Scatter(x=t, y=sr, name="Right Limb", line=dict(color='#f87171', width=2)))
    fig_force.add_trace(go.Scatter(x=t, y=sf, name="Total Force", line=dict(color='#4d2994', width=3.5)))

    fig_force.add_vrect(x0=t_start, x1=t_braking, fillcolor="yellow", opacity=0.08, line_width=0)
    fig_force.add_vrect(x0=t_braking, x1=t_split, fillcolor="red", opacity=0.08, line_width=0)
    fig_force.add_vrect(x0=t_split, x1=t_takeoff, fillcolor="green", opacity=0.08, line_width=0)

    fig_force.add_vline(x=t_start, line_dash="dot", line_color="#ca8a04")
    fig_force.add_vline(x=t_braking, line_dash="dot", line_color="#ef4444")
    fig_force.add_vline(x=t_split, line_dash="dot", line_color="#22c55e")
    fig_force.add_vline(x=t_takeoff, line_dash="dot", line_color="#dc2626")

    fig_force.add_annotation(x=t_start + (t_braking - t_start)/2, y=max_f * 1.05, text="UNWEIGHTING", showarrow=False, font=dict(color='#ca8a04', size=10))
    fig_force.add_annotation(x=t_braking + (t_split - t_braking)/2, y=max_f * 1.18, text="BRAKING", showarrow=False, font=dict(color='#ef4444', size=10))
    fig_force.add_annotation(x=t_split + (t_takeoff - t_split)/2, y=max_f * 1.05, text="PROPULSIVE", showarrow=False, font=dict(color='#22c55e', size=10))

    fig_force.update_layout(title="FORCE-TIME ANALYSIS & SUB-PHASES", xaxis_title="Time (s)", yaxis_title="Force (N)", height=420)
    st.plotly_chart(fig_force, use_container_width=True)

    # Display Asymmetry Deficit Chart
    deficits = np.where(sf >= 50, ((sl - sr) / np.maximum(sl, sr)) * 100, 0)
    fig_deficit = go.Figure()
    fig_deficit.add_trace(go.Scatter(x=t, y=deficits, name="Asymmetry", fill='tozeroy', fillcolor='rgba(77, 41, 148, 0.15)', line=dict(color='#4d2994', width=2)))
    fig_deficit.add_hrect(y0=-threshold_alert, y1=threshold_alert, fillcolor="rgba(34, 197, 94, 0.15)", line_width=0)
    fig_deficit.update_layout(title="L/R ASYMMETRY % (Threshold Alert)", xaxis_title="Time (s)", yaxis_title="Deficit %", yaxis_range=[-55, 55], height=260)
    st.plotly_chart(fig_deficit, use_container_width=True)

    # Render Report Table
    st.markdown("### Standard Biomechanical Analysis Report")
    table_rows = []
    for phase_name, metrics in report.items():
        table_rows.append({"Biomechanical Metric": f"=== {phase_name.upper()} ===", "Left": "", "Right": "", "TOTAL": ""})
        for metric_name, vals in metrics.items():
            table_rows.append({
                "Biomechanical Metric": metric_name,
                "Left": vals["Left"],
                "Right": vals["Right"],
                "TOTAL": vals["Total"]
            })
    
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

else:
    st.info("Please upload Plate File A and Plate File B (.tsv) to begin analysis.")
