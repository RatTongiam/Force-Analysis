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
# 1. Sidebar: Data Import & File Upload
# -------------------------------------------------------------
st.sidebar.header("📁 1. Data Import & File Upload")

data_mode = st.sidebar.radio("Select Input Mode", [
    "MuscleLab CSV",
    "Single JSON (QTM)",
    "Dual TSV (QTM)", 
    "VALD ForceDecks CSV", 
    "C-Force Performance CSV"
])

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
                left_name = st.sidebar.selectbox("Left Limb Plate", plate_names, index=0)
                right_names = [n for n in plate_names if n != left_name]
                right_name = st.sidebar.selectbox("Right Limb Plate", right_names, index=0)
                
                swap_sides = st.sidebar.toggle("🔀 Swap Left / Right Sides", value=False)
                if swap_sides:
                    left_name, right_name = right_name, left_name
                
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

# -------------------------------------------------------------
# 2. Sidebar: Filtering & Settings
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ 2. Signal Filtering")
filter_type = st.sidebar.selectbox("Select Filter Algorithm", ["Butterworth LPF", "Moving Average", "Raw Data (None)"], index=0)
cutoff_freq = st.sidebar.slider("Cutoff Frequency (Hz)", min_value=5.0, max_value=100.0, value=50.0, step=5.0) if filter_type == "Butterworth LPF" else 50.0
filter_size = st.sidebar.selectbox("Window Size (Frames)", [1, 7, 15, 31], index=2) if filter_type == "Moving Average" else 15
threshold_alert = st.sidebar.number_input("Asymmetry Alert %", value=15.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("🎨 3. Presentation Theme")
app_theme = st.sidebar.selectbox("Select Dashboard Theme", ["Modern Coach (Dashboard)", "Classic Purple (Original)"], index=0)

if "Coach" in app_theme:
    group_mode_param = "phase"
    involved_limb = st.sidebar.selectbox("Involved Limb", ["None / Athlete", "Left", "Right"], index=0)
    baseline_jh = st.sidebar.number_input("Baseline JH (cm, optional)", value=0.0, step=0.5, format="%.1f")
    mdc_pct = st.sidebar.number_input("MDC / Meaningful Change (%)", value=5.0, step=0.5, format="%.1f")
else:
    group_mode_param = "variance"
    involved_limb = "None / Athlete"
    baseline_jh = 0.0
    mdc_pct = 5.0

col_l_hex = "#2f6fed" if "Coach" in app_theme else "#818cf8"
col_r_hex = "#ed7d31" if "Coach" in app_theme else "#f87171"
col_tot_hex = "#11395f" if "Coach" in app_theme else "#4d2994"

if "Coach" in app_theme:
    st.markdown("""
    <style>
    .coach-header { background: linear-gradient(120deg, #0d3154, #1c507f); color: #ffffff; padding: 18px 22px; border-radius: 12px; margin-bottom: 15px; }
    .kpi-card { background: #ffffff; border: 1px solid #e4e9f1; border-radius: 12px; padding: 12px 14px; box-shadow: 0 2px 6px rgba(16,24,40,.04); text-align: center; }
    .kpi-metric { font-size: 24px; font-weight: 800; color: #11395f; margin: 4px 0; }
    .kpi-sub { font-size: 11px; color: #667085; font-weight: 600; }
    .coach-box { background: #f7fbff; border-left: 5px solid #11395f; padding: 14px; border-radius: 8px; font-size: 13.5px; line-height: 1.5; margin-bottom: 8px; }
    .coach-action { background: #fff8f1; border: 1px solid #f5d2b7; padding: 9px 12px; border-radius: 8px; font-size: 12.5px; margin-top: 6px; color: #7c2d12; }
    .qc-banner-pass { background: #e9f7f0; color: #167a55; border: 1px solid #c9ead9; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; }
    .qc-banner-review { background: #fff4dc; color: #8a5a00; border: 1px solid #f2dcaa; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; }
    .status-pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 10.5px; font-weight: 800; }
    .status-low { background: #e9f7f0; color: #167a55; }
    .status-watch { background: #fff2d6; color: #9b6700; }
    .status-flag { background: #fdecec; color: #b83232; }
    </style>
    <div class="coach-header">
        <h2 style="margin:0; font-size:24px; color:#fff;">CMJ Coach Analyzer v4 — Research & Clinical</h2>
        <p style="margin:4px 0 0; font-size:12px; color:#dce8f5;">Dr.Chawin and PRIMA MOTION TECHNOLOGY • Comprehensive Biomechanics & Dual-Plate Asymmetry Engine</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("Free JumpAnz Team - Biomechanics Analysis")
    st.caption("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight")

def asym_badge(val_l, val_r):
    try:
        vl, vr = float(val_l), float(val_r)
        mx = max(abs(vl), abs(vr))
        if mx == 0:
            return "-", '<span class="status-pill status-low">Equal</span>'
        diff = ((vl - vr) / mx) * 100.0
        txt = f"{abs(diff):.1f}% {'Left' if diff > 0 else 'Right'} higher"
        cls = "status-low" if abs(diff) < 10 else ("status-watch" if abs(diff) < 15 else "status-flag")
        badge = f'<span class="status-pill {cls}">{"Low" if abs(diff)<10 else ("Monitor" if abs(diff)<15 else "Flag")}</span>'
        return txt, badge
    except Exception:
        return "-", "-"

if t is not None and f_total is not None and len(f_total) > 0:
    dt_val = float(dt) if dt is not None and dt > 0 else 0.001
    fs = 1.0 / dt_val
    n_samples = len(f_total)
    quiet_samples = max(1, min(int(0.5 / dt_val), n_samples))

    sIdx_auto, bIdx_auto, zIdx_auto, tIdx_auto, lIdx_auto, lIdx_l, lIdx_r, offsets = detect_phases_sequential(
        t, f_total, dt_val, quiet_samples=quiet_samples, filter_type=filter_type, cutoff=cutoff_freq, sl_raw=f_left, sr_raw=f_right
    )

    t_min, t_max = float(t[0]), float(t[-1])
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

    t_start, t_braking, t_split, t_takeoff, t_landing = (
        st.session_state.t_start, st.session_state.t_braking, st.session_state.t_split, st.session_state.t_takeoff, st.session_state.t_landing
    )

    sIdx = min(max(0, int(round((t_start - t[0]) / dt_val))), n_samples - 1)
    bIdx = min(max(0, int(round((t_braking - t[0]) / dt_val))), n_samples - 1)
    zIdx = min(max(0, int(round((t_split - t[0]) / dt_val))), n_samples - 1)
    tIdx = min(max(0, int(round((t_takeoff - t[0]) / dt_val))), n_samples - 1)
    lIdx = min(max(0, int(round((t_landing - t[0]) / dt_val))), n_samples - 1)

    # คำนวณผ่าน Engine เพียงครั้งเดียว
    report, m, series = calculate_metrics(
        t, f_total, f_left, f_right, dt_val, sIdx, bIdx, zIdx, tIdx, lIdx, 
        filter_type=filter_type, cutoff=cutoff_freq, offsets=offsets,
        lIdx_l=lIdx_l, lIdx_r=lIdx_r, group_by=group_mode_param
    )

    sf, sl, sr = series["sf_filtered"], series["sl_filtered"], series["sr_filtered"]
    vel_total, power_wkg = series["vel"], series["power_wkg"]

    total_duration = max(dt_val, t_landing - t_start)
    pad_time = 0.20 * total_duration
    crop_x_min, crop_x_max = max(t_min, t_start - pad_time), min(t_max, t_landing + pad_time)
    display_x_min = crop_x_min if st.session_state.get("is_confirmed", False) else t_min
    display_x_max = crop_x_max if st.session_state.get("is_confirmed", False) else t_max

    # ตรวจสอบความสอดคล้องของผลลัพธ์ (QC Agreement)
    cv_val = (m["bw_sd"] / m["bw"]) * 100.0 if m["bw"] > 0 else 0.0
    jh_diff_val = abs(m["jh_flt"] - m["jh_imp"])
    jh_diff_pct = (jh_diff_val / max(m["jh_imp"], 1e-6)) * 100.0
    qc_pass = (cv_val < 2.0) and (jh_diff_pct < 10.0)

    p_diff = ((m["pr_net_l"] - m["pr_net_r"]) / max(m["pr_net_l"], m["pr_net_r"], 1e-6)) * 100.0
    l_diff = ((m["pk_land_l"] - m["pk_land_r"]) / max(m["pk_land_l"], m["pk_land_r"], 1e-6)) * 100.0

    headline = (
        f"Take-off net impulse สมดุล แต่ peak landing load เอนไปทาง {'Left' if l_diff > 0 else 'Right'}" if abs(p_diff) < 10 and abs(l_diff) >= 15
        else (f"พบ directional asymmetry ใน propulsion net impulse ไปทาง {'Left' if p_diff > 0 else 'Right'}" if abs(p_diff) >= 15
        else "การแบ่งแรงซ้าย–ขวาช่วง propulsion อยู่ในช่วงค่อนข้างสมดุลของ trial นี้")
    )

    # KPI Summary Cards
    if "Coach" in app_theme:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-sub">JUMP HEIGHT (IMPULSE)</div><div class="kpi-metric">{m["jh_imp"]:.1f} cm</div><div class="kpi-sub">primary kinetics estimate</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-sub">JUMP HEIGHT (FLIGHT)</div><div class="kpi-metric">{m["jh_flt"]:.1f} cm</div><div class="kpi-sub">quality-check estimate</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-sub">RSImod</div><div class="kpi-metric">{m["rsi"]:.2f} m/s</div><div class="kpi-sub">JH (impulse) / contraction time</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-sub">PEAK CONCENTRIC POWER</div><div class="kpi-metric">{m["ppk_wkg"]:.1f} W/kg</div><div class="kpi-sub">{m["ppk_wkg"]*m["mass"]:.0f} W Total</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Force Curve Plot
    col_g, col_s = st.columns([8, 4]) if "Coach" in app_theme else (st.container(), None)
    with col_g:
        fig_force = go.Figure()
        fig_force.add_trace(go.Scatter(x=t, y=sl, name="Left", line=dict(color=col_l_hex, width=1.1 if "Coach" in app_theme else 0.8)))
        fig_force.add_trace(go.Scatter(x=t, y=sr, name="Right", line=dict(color=col_r_hex, width=1.1 if "Coach" in app_theme else 0.8)))
        fig_force.add_trace(go.Scatter(x=t, y=sf, name="Total", line=dict(color=col_tot_hex, width=1.8 if "Coach" in app_theme else 1.2)))

        for x0_val, x1_val, fill_col in [(t_start, t_braking, "rgba(234, 179, 8, 0.12)"), (t_braking, t_split, "rgba(239, 68, 68, 0.12)"), (t_split, t_takeoff, "rgba(34, 197, 94, 0.12)"), (t_takeoff, t_landing, "rgba(148, 163, 184, 0.12)")]:
            fig_force.add_vrect(x0=x0_val, x1=x1_val, fillcolor=fill_col, line_width=0)

        for line_x, line_col in [(t_start, "#ca8a04"), (t_braking, "#ef4444"), (t_split, "#22c55e"), (t_takeoff, "#dc2626"), (t_landing, "#0284c7")]:
            fig_force.add_vline(x=line_x, line_width=1.5, line_dash="dash", line_color=line_col)

        fig_force.update_layout(title="Interactive Vertical Force–Time Curve", xaxis_title="Time (s)", yaxis_title="Force (N)", height=460, margin=dict(l=40, r=40, t=50, b=20))
        fig_force.update_xaxes(range=[display_x_min, display_x_max])
        st.plotly_chart(fig_force, width="stretch")

    if "Coach" in app_theme and col_s:
        with col_s:
            st.markdown("### Coach snapshot")
            st.markdown(f'<div class="coach-box"><b>{headline}</b><br><br>JH {m["jh_imp"]:.1f} cm • RSImod {m["rsi"]:.2f} m/s • Propulsion net impulse asymmetry {abs(p_diff):.1f}% • Landing peak asymmetry {abs(l_diff):.1f}%.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="coach-action">Propulsion net impulse สมดุลใน trial นี้ใช้เป็น baseline เพื่อติดตาม fatigue / RTP ได้</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="coach-action">ใช้ค่าเฉลี่ย 3–5 valid trials และ CV% ก่อนสรุป pattern ระยะยาว</div>', unsafe_allow_html=True)

    # Timeline Controls
    st.markdown("##### 🎚️ Phase Boundary Timeline Controls")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        new_start = st.slider("1. Start Onset (s)", min_value=t_min, max_value=max(t_min + dt_val, t_braking - dt_val), value=st.session_state.t_start, step=dt_val, format="%.3f", key="s_1")
    with c2:
        new_braking = st.slider("2. Braking (s)", min_value=max(t_min + dt_val, new_start + dt_val), max_value=max(t_min + 2*dt_val, t_split - dt_val), value=st.session_state.t_braking, step=dt_val, format="%.3f", key="s_2")
    with c3:
        new_split = st.slider("3. Propulsive V=0 (s)", min_value=max(t_min + 2*dt_val, new_braking + dt_val), max_value=max(t_min + 3*dt_val, t_takeoff - dt_val), value=st.session_state.t_split, step=dt_val, format="%.3f", key="s_3")
    with c4:
        new_takeoff = st.slider("4. Take-off (s)", min_value=max(t_min + 3*dt_val, new_split + dt_val), max_value=max(t_min + 4*dt_val, t_landing - dt_val), value=st.session_state.t_takeoff, step=dt_val, format="%.3f", key="s_4")
    with c5:
        new_landing = st.slider("5. Landing (s)", min_value=max(t_min + 4*dt_val, new_takeoff + dt_val), max_value=t_max, value=st.session_state.t_landing, step=dt_val, format="%.3f", key="s_5")

    if (new_start, new_braking, new_split, new_takeoff, new_landing) != (t_start, t_braking, t_split, t_takeoff, t_landing):
        st.session_state.t_start, st.session_state.t_braking, st.session_state.t_split, st.session_state.t_takeoff, st.session_state.t_landing = new_start, new_braking, new_split, new_takeoff, new_landing
        st.rerun()

    if st.button("✅ Confirm Phases & Crop Graph" if not st.session_state.get("is_confirmed", False) else "✏️ Edit / Show Full View", type="primary"):
        st.session_state.is_confirmed = not st.session_state.get("is_confirmed", False)
        st.rerun()

    # QC Banner & Warning
    if "Coach" in app_theme:
        st.markdown("---")
        if not qc_pass:
            if jh_diff_pct >= 10.0:
                st.warning(f"⚠️ **QC Flag (JH Discrepancy > 10%):** ความสูงการกระโดดระหว่าง Impulse ({m['jh_imp']:.1f} cm) และ Flight ({m['jh_flt']:.1f} cm) ต่างกัน {jh_diff_pct:.1f}% ตรวจสอบท่ายกขา/งอเข่าตอน Landing หรือปรับจุด Takeoff/Landing ให้แม่นยำ")
            if cv_val >= 2.0:
                st.warning(f"⚠️ **QC Flag (Quiet Standing CV > 2.0%):** ค่าความแปรปรวนช่วงยืนนิ่งอยู่ที่ {cv_val:.2f}% ผู้ถูกทดสอบอาจมีการขยับตัวก่อนเริ่มกระโดด")

        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.markdown("### Data quality checks")
            st.markdown(f'<div class="{"qc-banner-pass" if qc_pass else "qc-banner-review"}">STATUS: {"PASS" if qc_pass else "REVIEW"}</div>', unsafe_allow_html=True)
            qc_df = pd.DataFrame([
                {"Check Item": "Sampling", "Value": f"{fs:.0f} Hz", "Status": "Calculated sampling frequency"},
                {"Check Item": "Flight zero-offset", "Value": f"{m['offset_l']:.1f} / {m['offset_r']:.1f} N", "Status": "Unloaded residual subtracted L / R"},
                {"Check Item": "Quiet-standing CV", "Value": f"{cv_val:.2f}%", "Status": f"BW {m['bw']:.2f} N • SD {m['bw_sd']:.2f} N"},
                {"Check Item": "JH method agreement", "Value": f"{jh_diff_val:.2f} cm", "Status": f"{jh_diff_pct:.1f}% difference: impulse vs flight-time"}
            ])
            st.dataframe(qc_df, hide_index=True, width="stretch")

        with r1_c2:
            st.markdown("### Timing & strategy")
            strat_df = pd.DataFrame([
                {"Metric": "Body mass", "Value": f"{m['mass']:.2f} kg"},
                {"Metric": "Time to take-off", "Value": f"{m['ttt']*1000:.0f} ms"},
                {"Metric": "Unweighting duration", "Value": f"{m['d_unw']*1000:.0f} ms"},
                {"Metric": "Braking duration", "Value": f"{m['d_brk']*1000:.0f} ms"},
                {"Metric": "Propulsion duration", "Value": f"{m['d_pro']*1000:.0f} ms"},
                {"Metric": "Flight time", "Value": f"{m['d_fly']*1000:.0f} ms"},
                {"Metric": "Countermovement depth", "Value": f"{m['com_depth']:.1f} cm"}
            ])
            st.dataframe(strat_df, hide_index=True, width="stretch")

    # Full Biomechanical Report Table
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

    # PDF Download
    pdf_bytes = generate_pdf_report(
        report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing,
        threshold_alert=threshold_alert, crop_x_min=crop_x_min, crop_x_max=crop_x_max,
        theme=app_theme, coach_context={}
    )
    st.sidebar.markdown("---")
    st.sidebar.download_button(label="📥 Download PDF Report", data=pdf_bytes, file_name="CMJ_Analysis_Report.pdf", mime="application/pdf")
else:
    st.info("Please upload data file(s) to begin analysis.")
