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

# -------------------------------------------------------------
# 1. Sidebar: Presentation, Themes & Clinical Inputs
# -------------------------------------------------------------
st.sidebar.header("🎨 Theme & Presentation")

app_theme = st.sidebar.selectbox(
    "Select Dashboard Theme",
    ["Modern Coach (Dashboard)", "Classic Purple (Original)"],
    index=0
)

report_grouping = st.sidebar.radio(
    "📊 Metric Grouping Mode",
    ["Movement Sub-phases (Phase-by-Phase)", "Variance Explained (Anicic et al., 2023)"],
    index=0
)

if "Coach" in app_theme:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏥 Clinical & Research Inputs")
    involved_limb = st.sidebar.selectbox("Involved Limb", ["None / Athlete", "Left", "Right"], index=0)
    baseline_jh = st.sidebar.number_input("Baseline JH (cm, optional)", value=0.0, step=0.5, format="%.1f")
    mdc_pct = st.sidebar.number_input("MDC / Meaningful Change (%)", value=5.0, step=0.5, format="%.1f")
else:
    involved_limb = "None / Athlete"
    baseline_jh = 0.0
    mdc_pct = 5.0

# Theme Color Palettes & CSS Injection
if "Coach" in app_theme:
    col_l_hex = "#2f6fed"
    col_r_hex = "#ed7d31"
    col_tot_hex = "#11395f"
    st.markdown("""
        <style>
        .coach-header {
            background: linear-gradient(120deg, #0d3154, #1c507f);
            color: #ffffff; padding: 18px 22px; border-radius: 12px; margin-bottom: 15px;
        }
        .kpi-card {
            background: #ffffff; border: 1px solid #e4e9f1; border-radius: 12px;
            padding: 12px 14px; box-shadow: 0 2px 6px rgba(16,24,40,.04); text-align: center;
        }
        .kpi-metric { font-size: 24px; font-weight: 800; color: #11395f; margin: 4px 0; }
        .kpi-sub { font-size: 11px; color: #667085; font-weight: 600; }
        .coach-box {
            background: #f7fbff; border-left: 5px solid #11395f; padding: 14px;
            border-radius: 8px; font-size: 13.5px; line-height: 1.5; margin-bottom: 8px;
        }
        .coach-action {
            background: #fff8f1; border: 1px solid #f5d2b7; padding: 9px 12px;
            border-radius: 8px; font-size: 12.5px; margin-top: 6px; color: #7c2d12;
        }
        .qc-banner-pass { background: #e9f7f0; color: #167a55; border: 1px solid #c9ead9; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; }
        .qc-banner-review { background: #fff4dc; color: #8a5a00; border: 1px solid #f2dcaa; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; }
        .qc-banner-reject { background: #fdecec; color: #a52b2b; border: 1px solid #f2caca; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; }
        .status-pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 10.5px; font-weight: 800; }
        .status-low { background: #e9f7f0; color: #167a55; }
        .status-watch { background: #fff2d6; color: #9b6700; }
        .status-flag { background: #fdecec; color: #b83232; }
        </style>
    """, unsafe_allow_html=True)
else:
    col_l_hex = "#818cf8"
    col_r_hex = "#f87171"
    col_tot_hex = "#4d2994"

st.sidebar.markdown("---")
st.sidebar.header("📁 Data Import & Settings")

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
    cutoff_freq = st.sidebar.slider("Cutoff Frequency (Hz)", min_value=5.0, max_value=100.0, value=50.0, step=5.0)
    filter_size = 15
elif filter_type == "Moving Average":
    filter_size = st.sidebar.selectbox("Window Size (Frames)", [1, 7, 15, 31], index=2)
    cutoff_freq = 50.0
else:
    cutoff_freq = 50.0
    filter_size = 1

threshold_alert = st.sidebar.number_input("Asymmetry Alert %", value=15.0, step=1.0)

# Dashboard Header
if "Coach" in app_theme:
    st.markdown("""
        <div class="coach-header">
            <h2 style="margin:0; font-size:24px; color:#fff;">CMJ Coach Analyzer v4 — Research & Clinical</h2>
            <p style="margin:4px 0 0; font-size:12px; color:#dce8f5;">PRIMA MOTION TECHNOLOGY • Comprehensive Biomechanics & Dual-Plate Asymmetry Engine</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.title("Free JumpAnz Team - Biomechanics Analysis")
    st.caption("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight")

dt, t, f_left, f_right, f_total = None, None, None, None, None
file_signature = None

# Parsers
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

# Helper Functions
def asym_badge(val_l, val_r):
    try:
        vl, vr = float(val_l), float(val_r)
        mx = max(abs(vl), abs(vr))
        if mx == 0:
            return "-", '<span class="status-pill status-low">Equal</span>'
        diff = ((vl - vr) / mx) * 100.0
        txt = f"{abs(diff):.1f}% {'Left' if diff > 0 else 'Right'}"
        cls = "status-low" if abs(diff) < 10 else ("status-watch" if abs(diff) < 15 else "status-flag")
        badge = f'<span class="status-pill {cls}">{"Low" if abs(diff)<10 else ("Monitor" if abs(diff)<15 else "Flag")}</span>'
        return txt, badge
    except Exception:
        return "-", "-"

if t is not None and f_total is not None and len(f_total) > 0:
    dt_val = float(dt) if dt is not None and dt > 0 else 0.001
    fs = 1.0 / dt_val
    g = 9.80665
    
    n_samples = len(f_total)
    quiet_samples = max(1, min(int(0.5 / dt_val), n_samples))

    # Phase Detection
    sIdx_auto, bIdx_auto, zIdx_auto, tIdx_auto, lIdx_auto, lIdx_l, lIdx_r, offsets = detect_phases_sequential(
        t, f_total, dt_val, quiet_samples=quiet_samples, 
        filter_type=filter_type, cutoff=cutoff_freq, 
        sl_raw=f_left, sr_raw=f_right
    )

    offset_l, offset_r = offsets
    f_left_zeroed = f_left - offset_l
    f_right_zeroed = f_right - offset_r
    f_total_zeroed = f_left_zeroed + f_right_zeroed

    sf = apply_signal_filter(f_total_zeroed, filter_type=filter_type, cutoff=cutoff_freq, fs=fs, window_size=filter_size)
    sl = apply_signal_filter(f_left_zeroed, filter_type=filter_type, cutoff=cutoff_freq, fs=fs, window_size=filter_size)
    sr = apply_signal_filter(f_right_zeroed, filter_type=filter_type, cutoff=cutoff_freq, fs=fs, window_size=filter_size)

    t_min = float(t[0])
    t_max = float(t[-1])

    is_file_changed = (st.session_state.get("current_file_sig") != file_signature)
    
    if is_file_changed or "t_start" not in st.session_state or st.sidebar.button("🔄 Reset Phases"):
        st.session_state.t_start = float(t[sIdx_auto])
        st.session_state.t_braking = float(t[bIdx_auto])
        st.session_state.t_split = float(t[zIdx_auto])
        st.session_state.t_takeoff = float(t[tIdx_auto])
        st.session_state.t_landing = float(t[lIdx_auto])
        st.session_state.is_confirmed = False
        st.session_state.current_file_sig = file_signature

    st.session_state.t_start = max(t_min, min(st.session_state.t_start, t_max - 4 * dt_val))
    st.session_state.t_braking = max(st.session_state.t_start + dt_val, min(st.session_state.t_braking, t_max - 3 * dt_val))
    st.session_state.t_split = max(st.session_state.t_braking + dt_val, min(st.session_state.t_split, t_max - 2 * dt_val))
    st.session_state.t_takeoff = max(st.session_state.t_split + dt_val, min(st.session_state.t_takeoff, t_max - dt_val))
    st.session_state.t_landing = max(st.session_state.t_takeoff + dt_val, min(st.session_state.t_landing, t_max))

    t_start = st.session_state.t_start
    t_braking = st.session_state.t_braking
    t_split = st.session_state.t_split
    t_takeoff = st.session_state.t_takeoff
    t_landing = st.session_state.t_landing

    total_duration = max(dt_val, t_landing - t_start)
    pad_time = 0.20 * total_duration

    crop_x_min = max(t_min, t_start - pad_time)
    crop_x_max = min(t_max, t_landing + pad_time)

    display_x_min = crop_x_min if st.session_state.get("is_confirmed", False) else t_min
    display_x_max = crop_x_max if st.session_state.get("is_confirmed", False) else t_max

    sIdx = min(max(0, int(round((t_start - t[0]) / dt_val))), n_samples - 1)
    bIdx = min(max(0, int(round((t_braking - t[0]) / dt_val))), n_samples - 1)
    zIdx = min(max(0, int(round((t_split - t[0]) / dt_val))), n_samples - 1)
    tIdx = min(max(0, int(round((t_takeoff - t[0]) / dt_val))), n_samples - 1)
    lIdx = min(max(0, int(round((t_landing - t[0]) / dt_val))), n_samples - 1)

    group_mode_param = "phase" if "Phase-by-Phase" in report_grouping else "variance"

    report = calculate_metrics(
        t, f_total, f_left, f_right, dt_val, sIdx, bIdx, zIdx, tIdx, lIdx, 
        filter_type=filter_type, cutoff=cutoff_freq, offsets=offsets,
        lIdx_l=lIdx_l, lIdx_r=lIdx_r, group_by=group_mode_param
    )

    # Calculate Kinematics (COM Velocity & Power)
    bw = np.mean(sf[:quiet_samples])
    bw_sd = np.std(sf[:quiet_samples])
    mass = bw / g if bw > 0 else 70.0

    vel_total = np.zeros(n_samples)
    disp_total = np.zeros(n_samples)
    net_acc = (sf - bw) / mass
    for i in range(sIdx + 1, min(tIdx + 1, n_samples)):
        vel_total[i] = vel_total[i - 1] + 0.5 * (net_acc[i - 1] + net_acc[i]) * dt_val
        disp_total[i] = disp_total[i - 1] + 0.5 * (vel_total[i - 1] + vel_total[i]) * dt_val

    power_total = sf * vel_total
    power_wkg = power_total / mass

    # Specific Variables for Coach Dashboard
    v_to = vel_total[tIdx]
    flight_dur = max(0.0, t_landing - t_takeoff)
    jh_imp_val = ((v_to ** 2) / (2.0 * g)) * 100.0
    jh_flt_val = (g * (flight_dur ** 2)) / 8.0 * 100.0
    ttt_val = max(dt_val, t_takeoff - t_start)
    rsi_val = (jh_imp_val / 100.0) / ttt_val if ttt_val > 0 else 0.0
    ppk_wkg = np.max(power_wkg[zIdx:tIdx+1]) if tIdx > zIdx else 0.0
    com_depth = abs(disp_total[zIdx] - disp_total[sIdx]) * 100.0

    # Net / Gross Impulse
    l_bw = np.mean(sl[:quiet_samples])
    r_bw = np.mean(sr[:quiet_samples])
    
    br_net_l = float(np.trapezoid(sl[bIdx:zIdx+1] - l_bw, dx=dt_val)) if zIdx > bIdx else 0.0
    br_net_r = float(np.trapezoid(sr[bIdx:zIdx+1] - r_bw, dx=dt_val)) if zIdx > bIdx else 0.0
    br_gross_l = float(np.trapezoid(sl[bIdx:zIdx+1], dx=dt_val)) if zIdx > bIdx else 0.0
    br_gross_r = float(np.trapezoid(sr[bIdx:zIdx+1], dx=dt_val)) if zIdx > bIdx else 0.0
    
    pr_net_l = float(np.trapezoid(sl[zIdx:tIdx+1] - l_bw, dx=dt_val)) if tIdx > zIdx else 0.0
    pr_net_r = float(np.trapezoid(sr[zIdx:tIdx+1] - r_bw, dx=dt_val)) if tIdx > zIdx else 0.0
    pr_gross_l = float(np.trapezoid(sl[zIdx:tIdx+1], dx=dt_val)) if tIdx > zIdx else 0.0
    pr_gross_r = float(np.trapezoid(sr[zIdx:tIdx+1], dx=dt_val)) if tIdx > zIdx else 0.0

    pk_br_l = np.max(sl[bIdx:zIdx+1]) if zIdx > bIdx else 0.0
    pk_br_r = np.max(sr[bIdx:zIdx+1]) if zIdx > bIdx else 0.0
    pk_pr_l = np.max(sl[zIdx:tIdx+1]) if tIdx > zIdx else 0.0
    pk_pr_r = np.max(sr[zIdx:tIdx+1]) if tIdx > zIdx else 0.0

    # Landing Load Variables
    land_search_end = min(n_samples, lIdx + int(0.5 / dt_val))
    pk_land_tot = np.max(sf[lIdx:land_search_end]) if land_search_end > lIdx else 0.0
    pk_land_l = np.max(sl[lIdx:land_search_end]) if land_search_end > lIdx else 0.0
    pk_land_r = np.max(sr[lIdx:land_search_end]) if land_search_end > lIdx else 0.0
    ttp_land_ms = (np.argmax(sf[lIdx:land_search_end]) * dt_val) * 1000.0 if land_search_end > lIdx else 0.0
    
    imp_250_end = min(n_samples, lIdx + int(0.25 / dt_val))
    land_imp_250_l = float(np.trapezoid(sl[lIdx:imp_250_end], dx=dt_val)) if imp_250_end > lIdx else 0.0
    land_imp_250_r = float(np.trapezoid(sr[lIdx:imp_250_end], dx=dt_val)) if imp_250_end > lIdx else 0.0
    land_imp_250_tot = land_imp_250_l + land_imp_250_r

    # Loading Rate 20-80%
    pk_idx = lIdx + np.argmax(sf[lIdx:land_search_end]) if land_search_end > lIdx else lIdx
    f_20 = 0.20 * pk_land_tot
    f_80 = 0.80 * pk_land_tot
    sub_land = sf[lIdx:pk_idx+1]
    idx_20 = np.where(sub_land >= f_20)[0]
    idx_80 = np.where(sub_land >= f_80)[0]
    if len(idx_20) > 0 and len(idx_80) > 0 and idx_80[0] > idx_20[0]:
        load_rate = (f_80 - f_20) / ((idx_80[0] - idx_20[0]) * dt_val)
    else:
        load_rate = 0.0

    # -------------------------------------------------------------
    # 2. Modern Dashboard: Top KPI Cards
    # -------------------------------------------------------------
    if "Coach" in app_theme:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-sub">JUMP HEIGHT (IMPULSE)</div><div class="kpi-metric">{jh_imp_val:.1f} cm</div><div class="kpi-sub">Preferred Kinetics</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-sub">JUMP HEIGHT (FLIGHT)</div><div class="kpi-metric">{jh_flt_val:.1f} cm</div><div class="kpi-sub">Quality-check Estimate</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-sub">RSI MODIFIED</div><div class="kpi-metric">{rsi_val:.2f}</div><div class="kpi-sub">JH (m) / TTT (s)</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-sub">PEAK CONCENTRIC POWER</div><div class="kpi-metric">{ppk_wkg:.1f} W/kg</div><div class="kpi-sub">{ppk_wkg*mass:.0f} W Total</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. Main Interactive Curve & Coach Snapshot
    # -------------------------------------------------------------
    if "Coach" in app_theme:
        col_g, col_s = st.columns([8, 4])
    else:
        col_g = st.container()

    with col_g:
        fig_force = go.Figure()
        fig_force.add_trace(go.Scatter(x=t, y=sl, name="Left Limb", line=dict(color=col_l_hex, width=1.1 if "Coach" in app_theme else 0.8)))
        fig_force.add_trace(go.Scatter(x=t, y=sr, name="Right Limb", line=dict(color=col_r_hex, width=1.1 if "Coach" in app_theme else 0.8)))
        fig_force.add_trace(go.Scatter(x=t, y=sf, name="Total Force", line=dict(color=col_tot_hex, width=1.8 if "Coach" in app_theme else 1.2)))

        fig_force.add_vrect(x0=t_start, x1=t_braking, fillcolor="rgba(234, 179, 8, 0.12)", line_width=0)
        fig_force.add_vrect(x0=t_braking, x1=t_split, fillcolor="rgba(239, 68, 68, 0.12)", line_width=0)
        fig_force.add_vrect(x0=t_split, x1=t_takeoff, fillcolor="rgba(34, 197, 94, 0.12)", line_width=0)
        fig_force.add_vrect(x0=t_takeoff, x1=t_landing, fillcolor="rgba(148, 163, 184, 0.12)", line_width=0)

        fig_force.add_vline(x=t_start, line_width=1.5, line_dash="dash", line_color="#ca8a04")
        fig_force.add_vline(x=t_braking, line_width=1.5, line_dash="dash", line_color="#ef4444")
        fig_force.add_vline(x=t_split, line_width=1.5, line_dash="dash", line_color="#22c55e")
        fig_force.add_vline(x=t_takeoff, line_width=1.5, line_dash="dash", line_color="#dc2626")
        fig_force.add_vline(x=t_landing, line_width=1.5, line_dash="dash", line_color="#0284c7")

        max_y = float(np.max(sf)) * 1.15 if len(sf) > 0 else 3000.0
        fig_force.add_annotation(x=(t_start+t_braking)/2, y=max_y*0.98, text="Unweighting", showarrow=False, font=dict(size=11, color="#ca8a04", family="Arial Bold"))
        fig_force.add_annotation(x=(t_braking+t_split)/2, y=max_y*0.90, text="Braking", showarrow=False, font=dict(size=11, color="#ef4444", family="Arial Bold"))
        fig_force.add_annotation(x=(t_split+t_takeoff)/2, y=max_y*0.98, text="Propulsive", showarrow=False, font=dict(size=11, color="#22c55e", family="Arial Bold"))
        fig_force.add_annotation(x=(t_takeoff+t_landing)/2, y=max_y*0.90, text="Flight", showarrow=False, font=dict(size=11, color="#64748b", family="Arial Bold"))

        github_base = "https://raw.githubusercontent.com/RatTongiam/Force-Analysis/main"
        pictograms = [
            {"url": f"{github_base}/Standing.png", "x": max(display_x_min + 0.1, t_start - 0.2)},
            {"url": f"{github_base}/UP.png", "x": (t_start + t_braking) / 2.0},
            {"url": f"{github_base}/BP.png", "x": (t_braking + t_split) / 2.0},
            {"url": f"{github_base}/PP.png", "x": (t_split + t_takeoff) / 2.0},
            {"url": f"{github_base}/FP.png", "x": (t_takeoff + t_landing) / 2.0},
            {"url": f"{github_base}/LP.png", "x": min(t_max, t_landing + 0.2)},
        ]
        for pic in pictograms:
            fig_force.add_layout_image(
                dict(
                    source=pic["url"], xref="x", yref="y", x=pic["x"], y=max_y * 0.75,
                    sizex=0.15, sizey=max_y * 0.22, xanchor="center", yanchor="bottom", layer="above"
                )
            )

        fig_force.update_layout(
            title="FORCE-TIME ANALYSIS & SUB-PHASES",
            xaxis_title="Time (s)", yaxis_title="Force (N)",
            height=460, margin=dict(l=40, r=40, t=50, b=20)
        )
        fig_force.update_xaxes(range=[display_x_min, display_x_max])
        st.plotly_chart(fig_force, width="stretch")

    if "Coach" in app_theme:
        with col_s:
            st.markdown("### 📋 Coach Snapshot")
            p_asym_txt, _ = asym_badge(pr_net_l, pr_net_r)
            l_asym_txt, _ = asym_badge(pk_land_l, pk_land_r)
            p_diff = ((pr_net_l - pr_net_r) / max(pr_net_l, pr_net_r, 1e-6)) * 100.0
            l_diff = ((pk_land_l - pk_land_r) / max(pk_land_l, pk_land_r, 1e-6)) * 100.0

            if abs(p_diff) < 10 and abs(l_diff) >= 15:
                headline = f"Take-off net impulse สมดุล แต่ peak landing load เอนไปทาง {'Left' if l_diff > 0 else 'Right'}"
            elif abs(p_diff) >= 15:
                headline = f"พบ directional asymmetry ใน propulsion net impulse ไปทาง {'Left' if p_diff > 0 else 'Right'}"
            else:
                headline = "การกระจายแรงซ้าย–ขวาช่วง propulsion อยู่ในเกณฑ์สมดุล"

            st.markdown(f"""
                <div class="coach-box">
                    <b>{headline}</b><br><br>
                    JH (Impulse): <b>{jh_imp_val:.1f} cm</b> | RSImod: <b>{rsi_val:.2f}</b><br>
                    Propulsion Net Asymmetry: <b>{p_asym_txt}</b><br>
                    Landing Peak Asymmetry: <b>{l_asym_txt}</b>
                </div>
            """, unsafe_allow_html=True)

            if abs(l_diff) >= 15:
                st.markdown(f'<div class="coach-action">⚠️ <b>Landing Bias:</b> ขาข้าง {"Left" if l_diff > 0 else "Right"} รับแรงกระแทกสูงสุดมากกว่าปกติ ตรวจสอบเทคนิคการลงสู่พื้นร่วมด้วย</div>', unsafe_allow_html=True)
            if abs(p_diff) >= 10:
                st.markdown(f'<div class="coach-action">⚡ <b>Propulsion Asymmetry:</b> ด้าน {"Left" if p_diff > 0 else "Right"} เป็นข้างหลักในการสร้างแรงขับเคลื่อน (Dominant Limb)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="coach-action">✅ <b>Symmetry Verified:</b> แรงส่งทั้งสองข้างสมดุลดี ใช้เป็น Baseline อ้างอิงได้</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. Phase Boundary Timeline Controls
    # -------------------------------------------------------------
    st.markdown("##### 🎚️ Phase Boundary Timeline Controls")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        v1_min, v1_max = t_min, max(t_min + dt_val, t_braking - dt_val)
        new_start = st.slider("1. Start Onset (s)", min_value=v1_min, max_value=v1_max, value=max(v1_min, min(st.session_state.t_start, v1_max)), step=dt_val, format="%.3f", key="s_1")
    with c2:
        v2_min, v2_max = max(t_min + dt_val, new_start + dt_val), max(t_min + 2*dt_val, t_split - dt_val)
        new_braking = st.slider("2. Braking (s)", min_value=v2_min, max_value=v2_max, value=max(v2_min, min(st.session_state.t_braking, v2_max)), step=dt_val, format="%.3f", key="s_2")
    with c3:
        v3_min, v3_max = max(t_min + 2*dt_val, new_braking + dt_val), max(t_min + 3*dt_val, t_takeoff - dt_val)
        new_split = st.slider("3. Propulsive V=0 (s)", min_value=v3_min, max_value=v3_max, value=max(v3_min, min(st.session_state.t_split, v3_max)), step=dt_val, format="%.3f", key="s_3")
    with c4:
        v4_min, v4_max = max(t_min + 3*dt_val, new_split + dt_val), max(t_min + 4*dt_val, t_landing - dt_val)
        new_takeoff = st.slider("4. Take-off (s)", min_value=v4_min, max_value=v4_max, value=max(v4_min, min(st.session_state.t_takeoff, v4_max)), step=dt_val, format="%.3f", key="s_4")
    with c5:
        v5_min, v5_max = max(t_min + 4*dt_val, new_takeoff + dt_val), t_max
        new_landing = st.slider("5. Landing (s)", min_value=v5_min, max_value=v5_max, value=max(v5_min, min(st.session_state.t_landing, v5_max)), step=dt_val, format="%.3f", key="s_5")

    if (new_start, new_braking, new_split, new_takeoff, new_landing) != (t_start, t_braking, t_split, t_takeoff, t_landing):
        st.session_state.t_start = new_start
        st.session_state.t_braking = new_braking
        st.session_state.t_split = new_split
        st.session_state.t_takeoff = new_takeoff
        st.session_state.t_landing = new_landing
        st.rerun()

    col_btn1, _ = st.columns([1, 4])
    with col_btn1:
        if not st.session_state.get("is_confirmed", False):
            if st.button("✅ Confirm Phases & Crop Graph", type="primary"):
                st.session_state.is_confirmed = True
                st.rerun()
        else:
            if st.button("✏️ Edit / Show Full View"):
                st.session_state.is_confirmed = False
                st.rerun()

    # -------------------------------------------------------------
    # 5. Advanced Clinical & Research Modules (Coach Theme)
    # -------------------------------------------------------------
    if "Coach" in app_theme:
        st.markdown("---")
        
        # Row 1: QC Checks & Timing Strategy
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.markdown("### 🛡️ Data Quality Checks")
            cv_val = (bw_sd / bw) * 100.0
            jh_diff_pct = abs(jh_flt_val - jh_imp_val) / max(jh_imp_val, 1e-6) * 100.0
            qc_pass = (cv_val < 2.0) and (jh_diff_pct < 10.0)
            
            if qc_pass:
                st.markdown('<div class="qc-banner-pass">QUALITY STATUS: PASS</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="qc-banner-review">QUALITY STATUS: REVIEW / CHECK BASELINE</div>', unsafe_allow_html=True)
            
            qc_df = pd.DataFrame([
                {"Check Item": "Sampling Rate", "Value": f"{fs:.0f} Hz", "Status": "Valid"},
                {"Check Item": "Quiet-standing CV", "Value": f"{cv_val:.2f}% (SD {bw_sd:.1f} N)", "Status": "Good" if cv_val < 1.0 else "Review"},
                {"Check Item": "JH Method Disagreement", "Value": f"{abs(jh_flt_val - jh_imp_val):.2f} cm ({jh_diff_pct:.1f}%)", "Status": "Good" if jh_diff_pct < 5.0 else "Review"},
                {"Check Item": "Flight Zero-Offset Subtraction", "Value": f"L {offset_l:.1f} N | R {offset_r:.1f} N", "Status": "Corrected"}
            ])
            st.dataframe(qc_df, hide_index=True, width="stretch")

        with r1_c2:
            st.markdown("### ⏱️ Timing & Strategy")
            strat_df = pd.DataFrame([
                {"Strategy Metric": "Body Mass", "Value": f"{mass:.1f} kg"},
                {"Strategy Metric": "Time to Take-off (Contraction Time)", "Value": f"{ttt_val*1000:.0f} ms"},
                {"Strategy Metric": "Unweighting Duration", "Value": f"{(t_braking - t_start)*1000:.0f} ms"},
                {"Strategy Metric": "Braking Phase Duration", "Value": f"{(t_split - t_braking)*1000:.0f} ms"},
                {"Strategy Metric": "Propulsive Phase Duration", "Value": f"{(t_takeoff - t_split)*1000:.0f} ms"},
                {"Strategy Metric": "Flight Phase Duration", "Value": f"{flight_dur*1000:.0f} ms"},
                {"Strategy Metric": "Countermovement Dip Depth", "Value": f"{com_depth:.1f} cm"}
            ])
            st.dataframe(strat_df, hide_index=True, width="stretch")

        # Row 2: Braking / Propulsion Table
        st.markdown("### ⚖️ Braking & Propulsion — Left vs Right")
        bp_rows = [
            {"Phase": "Braking", "Metric": "Peak Force", "Left": f"{pk_br_l:.0f} N", "Right": f"{pk_br_r:.0f} N", "Asymmetry": asym_badge(pk_br_l, pk_br_r)[0]},
            {"Phase": "Braking", "Metric": "Net Impulse", "Left": f"{br_net_l:.1f} N·s", "Right": f"{br_net_r:.1f} N·s", "Asymmetry": asym_badge(br_net_l, br_net_r)[0]},
            {"Phase": "Braking", "Metric": "Gross Impulse", "Left": f"{br_gross_l:.1f} N·s", "Right": f"{br_gross_r:.1f} N·s", "Asymmetry": asym_badge(br_gross_l, br_gross_r)[0]},
            {"Phase": "Propulsion", "Metric": "Peak Force", "Left": f"{pk_pr_l:.0f} N", "Right": f"{pk_pr_r:.0f} N", "Asymmetry": asym_badge(pk_pr_l, pk_pr_r)[0]},
            {"Phase": "Propulsion", "Metric": "Net Impulse", "Left": f"{pr_net_l:.1f} N·s", "Right": f"{pr_net_r:.1f} N·s", "Asymmetry": asym_badge(pr_net_l, pr_net_r)[0]},
            {"Phase": "Propulsion", "Metric": "Gross Impulse", "Left": f"{pr_gross_l:.1f} N·s", "Right": f"{pr_gross_r:.1f} N·s", "Asymmetry": asym_badge(pr_gross_l, pr_gross_r)[0]}
        ]
        st.dataframe(pd.DataFrame(bp_rows), hide_index=True, width="stretch")

        # Row 3: Sub-charts (Velocity & Power)
        rc1, rc2 = st.columns(2)
        with rc1:
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=t[sIdx:lIdx+1], y=vel_total[sIdx:lIdx+1], line=dict(color="#11395f", width=1.5), name="COM Velocity"))
            fig_v.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            fig_v.update_layout(title="COM Velocity Curve (m/s)", height=260, margin=dict(l=30, r=30, t=40, b=20))
            st.plotly_chart(fig_v, width="stretch")
        with rc2:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=t[sIdx:tIdx+1], y=power_wkg[sIdx:tIdx+1], line=dict(color="#ed7d31", width=1.5), name="COM Power"))
            fig_p.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            fig_p.update_layout(title="COM Specific Power (W/kg)", height=260, margin=dict(l=30, r=30, t=40, b=20))
            st.plotly_chart(fig_p, width="stretch")

        # Row 4: Landing & Total Landing Load
        rl1, rl2 = st.columns(2)
        with rl1:
            st.markdown("### 🛬 Landing — Left vs Right")
            land_rows = [
                {"Metric": "Peak Landing Force", "Left": f"{pk_land_l:.0f} N", "Right": f"{pk_land_r:.0f} N", "Asymmetry": asym_badge(pk_land_l, pk_land_r)[0]},
                {"Metric": "Landing Impulse (0–250 ms)", "Left": f"{land_imp_250_l:.1f} N·s", "Right": f"{land_imp_250_r:.1f} N·s", "Asymmetry": asym_badge(land_imp_250_l, land_imp_250_r)[0]}
            ]
            st.dataframe(pd.DataFrame(land_rows), hide_index=True, width="stretch")
        with rl2:
            st.markdown("### 💥 Total Landing Load")
            tot_land_df = pd.DataFrame([
                {"Metric": "Peak Landing Force", "Value": f"{pk_land_tot:.0f} N ({pk_land_tot/bw:.2f} ×BW)"},
                {"Metric": "20–80% Loading Rate", "Value": f"{load_rate:.0f} N/s ({load_rate/bw:.1f} BW/s)"},
                {"Metric": "Landing Impulse (0–250 ms)", "Value": f"{land_imp_250_tot:.1f} N·s"},
                {"Metric": "Time to Peak Landing Force", "Value": f"{ttp_land_ms:.1f} ms"}
            ])
            st.dataframe(tot_land_df, hide_index=True, width="stretch")

        # Row 5: Research Derived Metrics & Clinical Limb Mapping
        rr1, rr2 = st.columns(2)
        with rr1:
            st.markdown("### 🔬 Research-grade Derived Metrics")
            mean_br_f = np.mean(sf[bIdx:zIdx+1]) if zIdx > bIdx else 0.0
            mean_pr_f = np.mean(sf[zIdx:tIdx+1]) if tIdx > zIdx else 0.0
            mean_br_p = np.mean(power_wkg[bIdx:zIdx+1]) if zIdx > bIdx else 0.0
            mean_pr_p = np.mean(power_wkg[zIdx:tIdx+1]) if tIdx > zIdx else 0.0
            pos_net_imp = float(np.trapezoid(np.maximum(0, sf[sIdx:tIdx+1] - bw), dx=dt_val))
            leg_stiff = (pk_br_l + pk_br_r) / (com_depth / 100.0) if com_depth > 0 else 0.0

            res_df = pd.DataFrame([
                {"Research Metric": "Mean Braking Force", "Value": f"{mean_br_f/mass:.2f} N/kg ({mean_br_f:.0f} N)"},
                {"Research Metric": "Mean Propulsive Force", "Value": f"{mean_pr_f/mass:.2f} N/kg ({mean_pr_f:.0f} N)"},
                {"Research Metric": "Mean Braking Power", "Value": f"{abs(mean_br_p):.2f} W/kg"},
                {"Research Metric": "Mean Propulsive Power", "Value": f"{mean_pr_p:.2f} W/kg"},
                {"Research Metric": "Positive Net Impulse", "Value": f"{pos_net_imp/mass:.3f} N·s/kg ({pos_net_imp:.1f} N·s)"},
                {"Research Metric": "Leg Stiffness (Exploratory)", "Value": f"{leg_stiff/mass:.1f} N/m/kg"}
            ])
            st.dataframe(res_df, hide_index=True, width="stretch")

        with rr2:
            st.markdown("### 🏥 Clinical Change & Limb Mapping")
            if involved_limb != "None / Athlete":
                inv_side = involved_limb
                uninv_side = "Right" if inv_side == "Left" else "Left"
                inv_p = pr_net_l if inv_side == "Left" else pr_net_r
                uninv_p = pr_net_r if inv_side == "Left" else pr_net_l
                p_def = ((inv_p - uninv_p) / abs(uninv_p)) * 100.0 if uninv_p != 0 else 0.0
                
                inv_l = pk_land_l if inv_side == "Left" else pk_land_r
                uninv_l = pk_land_r if inv_side == "Left" else pk_land_l
                l_def = ((inv_l - uninv_l) / abs(uninv_l)) * 100.0 if uninv_l != 0 else 0.0
                
                inv_p_str = f"{p_def:+.1f}%"
                inv_l_str = f"{l_def:+.1f}%"
            else:
                inv_p_str = "Select involved limb"
                inv_l_str = "Select involved limb"

            if baseline_jh > 0:
                ch_pct = ((jh_imp_val - baseline_jh) / baseline_jh) * 100.0
                ch_str = f"{ch_pct:+.1f}%"
                mdc_str = f"YES — exceeds {mdc_pct:.1f}%" if abs(ch_pct) >= mdc_pct else f"NO — within {mdc_pct:.1f}%"
            else:
                ch_str = "No baseline entered"
                mdc_str = "—"

            clin_df = pd.DataFrame([
                {"Clinical Variable": "Involved Propulsion Net Impulse Deficit", "Value": inv_p_str},
                {"Clinical Variable": "Involved Landing Peak Difference", "Value": inv_l_str},
                {"Clinical Variable": "Change from Baseline JH", "Value": ch_str},
                {"Clinical Variable": "Exceeds entered MDC?", "Value": mdc_str}
            ])
            st.dataframe(clin_df, hide_index=True, width="stretch")

        # Row 6: Interpretation for Coach
        st.markdown("### 💡 Interpretation for Coach")
        st.markdown(f"""
            * **Performance:** Jump Height จาก Impulse-Momentum = **{jh_imp_val:.1f} cm** (Flight Time = **{jh_flt_val:.1f} cm**), Time-to-takeoff = **{ttt_val*1000:.0f} ms**, RSImod = **{rsi_val:.2f}**
            * **Braking Strategy:** Net Impulse Asymmetry = **{asym_badge(br_net_l, br_net_r)[0]}** (สะท้อนการเปลี่ยนโมเมนตัมช่วงย่อตัว)
            * **Propulsion Strategy:** Net Impulse Asymmetry = **{asym_badge(pr_net_l, pr_net_r)[0]}** (แนะนำให้ดูควบคู่กับ Peak Force และ RFD)
            * **Landing Strategy:** Peak Landing Force Asymmetry = **{asym_badge(pk_land_l, pk_land_r)[0]}**, Loading Rate = **{load_rate/bw:.1f} BW/s**
        """)

    # -------------------------------------------------------------
    # 6. Asymmetry % Profile Graph
    # -------------------------------------------------------------
    st.markdown("---")
    max_sl_sr = np.maximum(sl, sr)
    deficits = np.where((sf >= 50) & (max_sl_sr > 0), ((sl - sr) / np.maximum(max_sl_sr, 1e-6)) * 100, 0)
    fig_deficit = go.Figure()

    fig_deficit.add_vrect(x0=t_start, x1=t_braking, fillcolor="rgba(234, 179, 8, 0.12)", line_width=0)
    fig_deficit.add_vrect(x0=t_braking, x1=t_split, fillcolor="rgba(239, 68, 68, 0.12)", line_width=0)
    fig_deficit.add_vrect(x0=t_split, x1=t_takeoff, fillcolor="rgba(34, 197, 94, 0.12)", line_width=0)
    fig_deficit.add_vrect(x0=t_takeoff, x1=t_landing, fillcolor="rgba(148, 163, 184, 0.12)", line_width=0)

    fig_deficit.add_vline(x=t_start, line_width=1.5, line_dash="dash", line_color="#ca8a04")
    fig_deficit.add_vline(x=t_braking, line_width=1.5, line_dash="dash", line_color="#ef4444")
    fig_deficit.add_vline(x=t_split, line_width=1.5, line_dash="dash", line_color="#22c55e")
    fig_deficit.add_vline(x=t_takeoff, line_width=1.5, line_dash="dash", line_color="#dc2626")
    fig_deficit.add_vline(x=t_landing, line_width=1.5, line_dash="dash", line_color="#0284c7")

    fig_deficit.add_hline(y=0, line_width=1.2, line_color="#6b7280")
    fig_deficit.add_trace(go.Scatter(x=t, y=deficits, name="Asymmetry", fill='tozeroy', fillcolor='rgba(17, 57, 95, 0.15)' if "Coach" in app_theme else 'rgba(77, 41, 148, 0.15)', line=dict(color=col_tot_hex, width=1.5)))
    fig_deficit.add_hrect(y0=-threshold_alert, y1=threshold_alert, fillcolor="rgba(34, 197, 94, 0.15)", line_width=0)

    fig_deficit.add_annotation(
        xref="paper", yref="y", x=0.01, y=38, text="<b>← Left Dominant (L > R)</b>", showarrow=False,
        font=dict(size=11, color=col_l_hex), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor=col_l_hex, borderwidth=1
    )
    fig_deficit.add_annotation(
        xref="paper", yref="y", x=0.01, y=-38, text="<b>← Right Dominant (R > L)</b>", showarrow=False,
        font=dict(size=11, color=col_r_hex), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor=col_r_hex, borderwidth=1
    )

    fig_deficit.update_layout(
        title="L/R ASYMMETRY % PROFILE (Threshold Alert & Limb Dominance)", 
        xaxis_title="Time (s)", yaxis_title="Deficit %", 
        yaxis_range=[-55, 55], height=340, margin=dict(l=40, r=40, t=50, b=20)
    )
    fig_deficit.update_xaxes(range=[display_x_min, display_x_max])
    st.plotly_chart(fig_deficit, width="stretch")

    # -------------------------------------------------------------
    # 7. Standard Report Table & PDF Export
    # -------------------------------------------------------------
    st.markdown(f"### 📋 Full Biomechanical Report ({'Movement Sub-phases' if group_mode_param=='phase' else 'Anicic et al., 2023'})")
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

    pdf_bytes = generate_pdf_report(
        report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing,
        threshold_alert=threshold_alert, crop_x_min=crop_x_min, crop_x_max=crop_x_max,
        theme=app_theme
    )
    st.download_button(
        label="📥 Download A4 PDF Report",
        data=pdf_bytes,
        file_name="Prima_Motion_CMJ_Report.pdf",
        mime="application/pdf",
        key="main_download_btn"
    )
else:
    st.info("Please upload data file(s) to begin analysis.")
