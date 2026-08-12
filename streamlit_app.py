import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

st.set_page_config(layout="wide", page_title="Free JumpAnz Team - Prima Motion Tech")

st.title("Free JumpAnz Team - Biomechanics Analysis")
st.caption("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight")

# --- SIDEBAR: FILE IMPORT & CONTROL ---
st.sidebar.header("Data Import & Settings")

data_mode = st.sidebar.radio("Select Input Mode", ["Dual TSV (Plate A + B)", "Single JSON (QTM)", "Single CSV"])

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

def parse_qtm_json(uploaded_file):
    content = json.load(uploaded_file)
    root = content[0] if isinstance(content, list) else content
    
    freq = root.get("Timebase", {}).get("Frequency", 120.0)
    dt = 1.0 / freq
    plates = root.get("ForcePlates", [])
    
    if len(plates) == 0:
        return None, None, None, None, None
        
    if len(plates) >= 2:
        vals_l = np.array(plates[0]["Parts"][0]["Values"])
        vals_r = np.array(plates[-1]["Parts"][0]["Values"])
        num_frames = min(len(vals_l), len(vals_r))
        f_left = np.abs(vals_l[:num_frames, 2])
        f_right = np.abs(vals_r[:num_frames, 2])
    else:
        vals = np.array(plates[0]["Parts"][0]["Values"])
        num_frames = len(vals)
        f_left = np.abs(vals[:num_frames, 2]) * 0.5
        f_right = np.abs(vals[:num_frames, 2]) * 0.5
        
    t = np.arange(num_frames) * dt
    f_total = f_left + f_right
    return dt, t, f_left, f_right, f_total

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

# Initialize Data Variables
dt, t, f_left, f_right, f_total = None, None, None, None, None

# --- PARSING SECTION ---
if data_mode == "Dual TSV (Plate A + B)":
    file_a = st.sidebar.file_uploader("Upload Plate File A (.tsv)", type=["tsv"])
    side_a = st.sidebar.selectbox("Assign Side File A", ["left", "right"], index=0)
    file_b = st.sidebar.file_uploader("Upload Plate File B (.tsv)", type=["tsv"])
    side_b = st.sidebar.selectbox("Assign Side File B", ["left", "right"], index=1)
    
    if file_a and file_b:
        freq_a, data_a = parse_tsv(file_a)
        freq_b, data_b = parse_tsv(file_b)
        dt = 1.0 / freq_a
        num_frames = min(len(data_a), len(data_b))
        t = np.arange(num_frames) * dt
        fz_a = np.abs(data_a[:num_frames, 2])
        fz_b = np.abs(data_b[:num_frames, 2])
        f_left = fz_a if side_a == "left" else fz_b
        f_right = fz_b if side_a == "left" else fz_a
        f_total = f_left + f_right

elif data_mode == "Single JSON (QTM)":
    file_json = st.sidebar.file_uploader("Upload QTM JSON File (.json)", type=["json"])
    if file_json:
        try:
            dt, t, f_left, f_right, f_total = parse_qtm_json(file_json)
        except Exception as e:
            st.error(f"Error parsing QTM JSON file: {e}")

elif data_mode == "Single CSV":
    file_csv = st.sidebar.file_uploader("Upload CSV File (.csv)", type=["csv"])
    if file_csv:
        df_csv = pd.read_csv(file_csv)
        if 'time' in df_csv.columns:
            t = df_csv['time'].values
            dt = t[1] - t[0] if len(t) > 1 else 1.0/1000.0
            f_left = df_csv['left force'].values if 'left force' in df_csv.columns else df_csv['force '].values * 0.5
            f_right = df_csv['right force'].values if 'right force' in df_csv.columns else df_csv['force '].values * 0.5
            f_total = f_left + f_right

# --- CORE BIOMECHANICAL ENGINE ---
if t is not None and f_total is not None:
    sf = moving_average(f_total, filter_size)
    sl = moving_average(f_left, filter_size)
    sr = moving_average(f_right, filter_size)
    
    g = 9.80665
    quiet_samples = max(1, int(0.5 / dt))
    
    # Calibration
    bw = np.mean(sf[:quiet_samples])
    bw_left = np.mean(sl[:quiet_samples])
    bw_right = np.mean(sr[:quiet_samples])
    
    force_sd = np.std(sf[:quiet_samples])
    mass = bw / g
    mass_left = bw_left / g if bw_left > 0 else mass * 0.5
    mass_right = bw_right / g if bw_right > 0 else mass * 0.5
    
    # Sub-phase Detection
    flight_threshold = 30.0
    flight_indices = np.where(sf < flight_threshold)[0]
    tIdx_auto = flight_indices[0] if len(flight_indices) > 0 else len(sf) - 1
    
    lIdx_matches = np.where((t > t[tIdx_auto] + 0.05) & (sf >= flight_threshold))[0]
    lIdx_auto = lIdx_matches[0] if len(lIdx_matches) > 0 else len(sf) - 1

    window_size = max(1, int(0.03 / dt))
    threshold_dev = max(force_sd * 5, bw * 0.025)

    sIdx_auto = 0
    for i in range(quiet_samples, tIdx_auto - window_size):
        if np.all(np.abs(sf[i:i + window_size] - bw) > threshold_dev):
            sIdx_auto = i
            break

    peak_force_idx = sIdx_auto + np.argmax(sf[sIdx_auto:tIdx_auto + 1])
    min_force_idx = sIdx_auto + np.argmin(sf[sIdx_auto:peak_force_idx])
    bIdx_auto = min_force_idx

    vel_temp = np.cumsum((sf[sIdx_auto:tIdx_auto + 1] - bw) / mass) * dt
    zero_vel_matches = np.where(vel_temp[bIdx_auto - sIdx_auto:] >= 0)[0]
    zIdx_auto = (bIdx_auto + zero_vel_matches[0]) if len(zero_vel_matches) > 0 else bIdx_auto

    # Sliders for Adjusting Phase Times
    st.sidebar.markdown("---")
    st.sidebar.subheader("Phase Adjustment")
    
    t_start = st.sidebar.slider("Start (Unweighting Onset)", 0.0, float(t[-1]), float(t[sIdx_auto]), step=0.005)
    t_braking = st.sidebar.slider("Braking Onset (Min Force)", 0.0, float(t[-1]), float(t[bIdx_auto]), step=0.005)
    t_split = st.sidebar.slider("Propulsive Onset (V=0)", 0.0, float(t[-1]), float(t[zIdx_auto]), step=0.005)
    t_takeoff = st.sidebar.slider("Take-off", 0.0, float(t[-1]), float(t[tIdx_auto]), step=0.005)

    sIdx = max(0, int(round(t_start / dt)))
    bIdx = max(0, int(round(t_braking / dt)))
    zIdx = max(0, int(round(t_split / dt)))
    tIdx = max(0, int(round(t_takeoff / dt)))
    
    lIdx_matches = np.where((t > t_takeoff + 0.05) & (sf >= flight_threshold))[0]
    lIdx = lIdx_matches[0] if len(lIdx_matches) > 0 else len(sf) - 1

    # Kinematics Integration
    vel_total = np.zeros(len(sf))
    vel_l = np.zeros(len(sf))
    vel_r = np.zeros(len(sf))
    disp_total = np.zeros(len(sf))

    cV = cVL = cVR = cD = 0.0

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
    flight_dur = max(0.0, t[lIdx] - t[tIdx])

    # 24 PCA Metrics
    jh_flight = (g * (flight_dur ** 2)) / 8.0 * 100.0
    v_takeoff = vel_total[tIdx]
    jh_impulse = ((v_takeoff ** 2) / (2.0 * g)) * 100.0
    rsi_modified = (jh_flight / 100.0) / contraction_time if contraction_time > 0 else 0.0

    unweight_f = sf[sIdx:bIdx + 1]
    unweight_impulse = calc_impulse(unweight_f, dt)

    brak_f = sf[bIdx:zIdx + 1]
    brak_fl = sl[bIdx:zIdx + 1]
    brak_fr = sr[bIdx:zIdx + 1]
    brak_v = vel_total[bIdx:zIdx + 1]
    brak_p = brak_f * brak_v

    avg_brak_f = calc_avg(brak_f)
    avg_brak_fl = calc_avg(brak_fl)
    avg_brak_fr = calc_avg(brak_fr)
    avg_brak_p = calc_avg(brak_p)
    peak_brak_f = np.max(brak_f) if len(brak_f) > 0 else 0.0
    peak_brak_fl = np.max(brak_fl) if len(brak_fl) > 0 else 0.0
    peak_brak_fr = np.max(brak_fr) if len(brak_fr) > 0 else 0.0
    peak_v_neg = np.min(vel_total[sIdx:zIdx + 1]) if len(vel_total[sIdx:zIdx + 1]) > 0 else 0.0

    brak_impulse = calc_impulse(brak_f, dt)
    brak_impulse_l = calc_impulse(brak_fl, dt)
    brak_impulse_r = calc_impulse(brak_fr, dt)
    brak_net_impulse = calc_net_impulse(brak_f, bw, dt)

    prop_f = sf[zIdx:tIdx + 1]
    prop_fl = sl[zIdx:tIdx + 1]
    prop_fr = sr[zIdx:tIdx + 1]
    prop_v = vel_total[zIdx:tIdx + 1]
    prop_p = prop_f * prop_v

    avg_prop_f = calc_avg(prop_f)
    avg_prop_fl = calc_avg(prop_fl)
    avg_prop_fr = calc_avg(prop_fr)
    avg_prop_p = calc_avg(prop_p)

    peak_prop_f = np.max(prop_f) if len(prop_f) > 0 else 0.0
    peak_prop_fl = np.max(prop_fl) if len(prop_fl) > 0 else 0.0
    peak_prop_fr = np.max(prop_fr) if len(prop_fr) > 0 else 0.0
    peak_prop_p = np.max(prop_p) if len(prop_p) > 0 else 0.0
    peak_v_prop = np.max(prop_v) if len(prop_v) > 0 else 0.0

    prop_impulse = calc_impulse(prop_f, dt)
    prop_impulse_l = calc_impulse(prop_fl, dt)
    prop_impulse_r = calc_impulse(prop_fr, dt)
    prop_net_impulse = calc_net_impulse(prop_f, bw, dt)
    positive_impulse = brak_impulse + prop_impulse

    land_impulse = 0.0
    if lIdx > tIdx and lIdx < len(sf):
        land_end_idx = min(len(sf), lIdx + int(0.5 / dt))
        land_impulse = calc_impulse(sf[lIdx:land_end_idx], dt)

    com_depth = abs(disp_total[zIdx] - disp_total[sIdx]) * 100.0
    com_takeoff = disp_total[tIdx] * 100.0
    leg_stiffness = (peak_brak_f / (com_depth / 100.0)) if com_depth > 0 else 0.0
    flight_jump_ratio = flight_dur / contraction_time if contraction_time > 0 else 0.0

    report = {
        "1. Performance Component (59% Variance)": {
            "Jump Height - Flight Time (cm)": {"Left": "-", "Right": "-", "Total": f"{jh_flight:.1f}"},
            "Jump Height - Impulse-Momentum (cm)": {"Left": "-", "Right": "-", "Total": f"{jh_impulse:.1f}"},
            "Flight Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{flight_dur:.2f}"},
            "Take-off Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{v_takeoff:.2f}"},
            "Peak Propulsive Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{peak_v_prop:.2f}"},
            "RSI Modified (AU)": {"Left": "-", "Right": "-", "Total": f"{rsi_modified:.2f}"},
            "Peak Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{peak_prop_p:.0f}"},
            "Landing Impulse (N·s)": {"Left": "-", "Right": "-", "Total": f"{land_impulse:.0f}"},
            "COM Height at Take-off (cm)": {"Left": "-", "Right": "-", "Total": f"{com_takeoff:.1f}"}
        },
        "2. Eccentric Component (16% Variance)": {
            "Mean Braking Power (W)": {"Left": "-", "Right": "-", "Total": f"{abs(avg_brak_p):.0f}"},
            "Mean Braking Force (N)": {"Left": f"{avg_brak_fl:.0f}", "Right": f"{avg_brak_fr:.0f}", "Total": f"{avg_brak_f:.0f}"},
            "Braking Impulse (N·s)": {"Left": f"{brak_impulse_l:.0f}", "Right": f"{brak_impulse_r:.0f}", "Total": f"{brak_impulse:.0f}"},
            "Unloading Impulse (N·s)": {"Left": "-", "Right": "-", "Total": f"{unweight_impulse:.0f}"},
            "Peak Negative Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{peak_v_neg:.2f}"}
        },
        "3. Concentric Component (11% Variance)": {
            "Mean Propulsive Force (N)": {"Left": f"{avg_prop_fl:.0f}", "Right": f"{avg_prop_fr:.0f}", "Total": f"{avg_prop_f:.0f}"},
            "Peak Propulsive Force (N)": {"Left": f"{peak_prop_fl:.0f}", "Right": f"{peak_prop_fr:.0f}", "Total": f"{peak_prop_f:.0f}"},
            "Peak Braking Force (N)": {"Left": f"{peak_brak_fl:.0f}", "Right": f"{peak_brak_fr:.0f}", "Total": f"{peak_brak_f:.0f}"},
            "Mean Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{avg_prop_p:.0f}"},
            "Propulsive Impulse (N·s)": {"Left": f"{prop_impulse_l:.0f}", "Right": f"{prop_impulse_r:.0f}", "Total": f"{prop_impulse:.0f}"},
            "Positive Impulse (N·s)": {"Left": "-", "Right": "-", "Total": f"{positive_impulse:.0f}"}
        },
        "4. Jump Strategy Component (6% Variance)": {
            "Propulsive Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{propulsive_dur:.2f}"},
            "Countermovement Depth (cm)": {"Left": "-", "Right": "-", "Total": f"{com_depth:.1f}"},
            "Leg Stiffness (N/m)": {"Left": "-", "Right": "-", "Total": f"{leg_stiffness:.0f}" if leg_stiffness > 0 else "N/A"},
            "Flight Time : Jump Time Ratio (AU)": {"Left": "-", "Right": "-", "Total": f"{flight_jump_ratio:.2f}"}
        }
    }

    # Plot Force-Time Chart
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

    # Plot Asymmetry Deficit Chart
    deficits = np.where(sf >= 50, ((sl - sr) / np.maximum(sl, sr)) * 100, 0)
    fig_deficit = go.Figure()
    fig_deficit.add_trace(go.Scatter(x=t, y=deficits, name="Asymmetry", fill='tozeroy', fillcolor='rgba(77, 41, 148, 0.15)', line=dict(color='#4d2994', width=2)))
    fig_deficit.add_hrect(y0=-threshold_alert, y1=threshold_alert, fillcolor="rgba(34, 197, 94, 0.15)", line_width=0)
    fig_deficit.update_layout(title="L/R ASYMMETRY % (Threshold Alert)", xaxis_title="Time (s)", yaxis_title="Deficit %", yaxis_range=[-55, 55], height=260)
    st.plotly_chart(fig_deficit, use_container_width=True)

    # Render Report Table
    st.markdown("### Standard Biomechanical Analysis Report (Anicic et al., 2023)")
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
    st.info("Please upload data file(s) to begin analysis.")
