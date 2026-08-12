import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
import io
import re

# ReportLab Imports for PDF Generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

st.set_page_config(layout="wide", page_title="Free JumpAnz Team - Prima Motion Tech")

st.title("Free JumpAnz Team - Biomechanics Analysis")
st.caption("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight")

# --- SIDEBAR: FILE IMPORT & CONTROL ---
st.sidebar.header("Data Import & Settings")

data_mode = st.sidebar.radio("Select Input Mode", [
    "Dual TSV (Plate A + B)", 
    "VALD ForceDecks (CSV/TSV)", 
    "Single JSON (QTM)", 
    "Single CSV (C-Force)"
])

filter_size = st.sidebar.selectbox("Smoothing Filter", [1, 7, 15, 31], index=2)
threshold_alert = st.sidebar.number_input("Asymmetry Alert %", value=15.0, step=1.0)

# ==============================================================================
# 1. ROBUST PARSER ENGINES
# ==============================================================================

def parse_tsv(uploaded_file):
    lines = uploaded_file.getvalue().decode('utf-8', errors='ignore').splitlines()
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
        if len(parts) >= 3 and not line_str.startswith("FORCE_PLATE"):
            try:
                vals = [float(p) for p in parts]
                data.append(vals)
            except ValueError:
                pass
    dt = 1.0 / freq
    arr = np.array(data)
    return dt, arr

def parse_vald_forcedecks_exact(uploaded_file):
    uploaded_file.seek(0)
    raw_text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
    lines = raw_text.splitlines()
    
    header_idx = 0
    is_new_format = False
    for idx, line in enumerate(lines[:30]):
        if 'Time' in line and 'Z Left' in line:
            header_idx = idx
            is_new_format = True
            break
            
    if is_new_format:
        data_content = "\n".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(data_content))
        df.columns = [c.strip() for c in df.columns]
        
        t = df['Time'].values.astype(float)
        if len(t) > 0 and t[0] > 10.0:
            t = t - t[0]
        dt = t[1] - t[0] if len(t) > 1 and (t[1] - t[0]) > 0 else 0.001
        
        f_left = np.abs(df['Z Left'].values.astype(float))
        f_right = np.abs(df['Z Right'].values.astype(float))
        f_total = f_left + f_right
        return dt, t, f_left, f_right, f_total
    else:
        filename = uploaded_file.name.lower()
        sep = '\t' if filename.endswith('.tsv') else ','
        df = pd.read_csv(io.StringIO(raw_text), skiprows=10, header=None, sep=sep, on_bad_lines='skip')
        df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
        
        t = df.iloc[:, 0].values.astype(float) if df.shape[1] > 0 else np.arange(len(df)) * 0.001
        if len(t) > 0 and t[0] > 10.0:
            t = t - t[0]
        dt = t[1] - t[0] if len(t) > 1 and (t[1] - t[0]) > 0 else 0.001
        
        f_left = np.abs(df.iloc[:, 1].values.astype(float)) if df.shape[1] > 1 else np.zeros(len(df))
        f_right = np.abs(df.iloc[:, 4].values.astype(float)) if df.shape[1] > 4 else f_left
        f_total = f_left + f_right
        return dt, t, f_left, f_right, f_total

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
    f_left = f_left / 1000.0
    f_right = f_right / 1000.0
    f_total = f_left + f_right
    return dt, t, f_left, f_right, f_total

def parse_single_csv_cforce(uploaded_file):
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    df.columns = [c.strip().lower() for c in df.columns]
    
    time_col = next((c for c in df.columns if 'time' in c or c == 't'), df.columns[0])
    total_col = next((c for c in df.columns if 'force' in c and 'left' not in c and 'right' not in c), df.columns[1] if len(df.columns) > 1 else None)
    left_col = next((c for c in df.columns if 'left' in c), df.columns[2] if len(df.columns) > 2 else None)
    right_col = next((c for c in df.columns if 'right' in c), df.columns[3] if len(df.columns) > 3 else None)
    
    t = pd.to_numeric(df[time_col], errors='coerce').fillna(0).values
    if len(t) > 0 and t[0] > 10.0:
        t = t - t[0]
    dt = t[1] - t[0] if len(t) > 1 and (t[1] - t[0]) > 0 else 0.001
    
    f_total = np.abs(pd.to_numeric(df[total_col], errors='coerce').fillna(0).values) if total_col else np.zeros(len(df))
    f_left = np.abs(pd.to_numeric(df[left_col], errors='coerce').fillna(0).values) if left_col else f_total * 0.5
    f_right = np.abs(pd.to_numeric(df[right_col], errors='coerce').fillna(0).values) if right_col else f_total * 0.5
    
    return dt, t, f_left, f_right, f_total

# ==============================================================================
# 2. MATH & COMPUTATIONAL HELPERS
# ==============================================================================

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

def calc_deficit_str(val_l, val_r):
    try:
        vl = float(val_l)
        vr = float(val_r)
        max_val = max(abs(vl), abs(vr))
        if max_val == 0:
            return "-"
        diff_pct = ((vl - vr) / max_val) * 100.0
        return f"{diff_pct:+.1f}%"
    except (ValueError, TypeError):
        return "-"

# ==============================================================================
# 3. PDF REPORT GENERATOR
# ==============================================================================

def generate_pdf_report(report_data, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor('#4d2994'))
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=colors.HexColor('#1a0f30'))
    cell_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor('#1a0f30'))
    cell_bold = ParagraphStyle('TableCellBold', parent=cell_style, fontName='Helvetica-Bold')
    
    story = [
        Paragraph("BIOMECHANICAL ANALYSIS REPORT (CMJ)", title_style),
        Paragraph("FREE JUMPANZ TEAM — PRIMA MOTION TECHNOLOGY", subtitle_style),
        Spacer(1, 10)
    ]

    fig_plt, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
    ax.plot(t, sl, label='Left Limb', color='#818cf8', linewidth=1.5)
    ax.plot(t, sr, label='Right Limb', color='#f87171', linewidth=1.5)
    ax.plot(t, sf, label='Total Force', color='#4d2994', linewidth=2.5)

    ax.axvspan(t_start, t_braking, color='yellow', alpha=0.15)
    ax.axvspan(t_braking, t_split, color='red', alpha=0.15)
    ax.axvspan(t_split, t_takeoff, color='green', alpha=0.15)

    ax.axvline(t_start, color='#ca8a04', linestyle='--', linewidth=1)
    ax.axvline(t_braking, color='#ef4444', linestyle='--', linewidth=1)
    ax.axvline(t_split, color='#22c55e', linestyle='--', linewidth=1)
    ax.axvline(t_takeoff, color='#dc2626', linestyle='--', linewidth=1)

    ax.set_title("FORCE-TIME ANALYSIS & SUB-PHASES", fontsize=9, fontweight='bold', color='#1a0f30')
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Force (N)", fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=200)
    plt.close(fig_plt)
    img_buffer.seek(0)

    story.append(Image(img_buffer, width=520, height=208))
    story.append(Spacer(1, 10))
    
    table_data = [[
        Paragraph("<b>Biomechanical Metric</b>", cell_bold), 
        Paragraph("<b>Left</b>", cell_bold), 
        Paragraph("<b>Right</b>", cell_bold), 
        Paragraph("<b>TOTAL</b>", cell_bold),
        Paragraph("<b>Deficit %</b>", cell_bold)
    ]]
    
    for phase_name, metrics in report_data.items():
        table_data.append([
            Paragraph(f"<b>{phase_name.upper()}</b>", ParagraphStyle('PhaseHeader', parent=cell_bold, textColor=colors.HexColor('#4d2994'))),
            "", "", "", ""
        ])
        for m_name, vals in metrics.items():
            table_data.append([
                Paragraph(m_name, cell_style),
                Paragraph(str(vals["Left"]), cell_style),
                Paragraph(str(vals["Right"]), cell_style),
                Paragraph(f"<b>{vals['Total']}</b>", cell_bold),
                Paragraph(str(vals["Deficit"]), cell_style)
            ])
            
    t_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f6fb')),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 3),
        ('TOPPADDING', (0, 0), (-1, 0), 3),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ('SPAN', (0, 1), (4, 1)),
        ('SPAN', (0, 12), (4, 12)),
        ('SPAN', (0, 18), (4, 18)),
        ('SPAN', (0, 23), (4, 23)),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f1ebf9')),
        ('BACKGROUND', (0, 12), (-1, 12), colors.HexColor('#f1ebf9')),
        ('BACKGROUND', (0, 18), (-1, 18), colors.HexColor('#f1ebf9')),
        ('BACKGROUND', (0, 23), (-1, 23), colors.HexColor('#f1ebf9')),
    ])
    
    doc_table = Table(table_data, colWidths=[200, 75, 75, 85, 85])
    doc_table.setStyle(t_style)
    story.append(doc_table)
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

# ==============================================================================
# 4. MAIN EXECUTION FLOW
# ==============================================================================

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
            dt, t, f_left, f_right, f_total = parse_qtm_json(file_json)
        except Exception as e:
            st.error(f"Error parsing QTM JSON file: {e}")

elif data_mode == "Single CSV (C-Force)":
    file_csv = st.sidebar.file_uploader("Upload Single CSV File (.csv)", type=["csv"])
    if file_csv:
        try:
            dt, t, f_left, f_right, f_total = parse_single_csv_cforce(file_csv)
        except Exception as e:
            st.error(f"Error parsing Single CSV file: {e}")

# ==============================================================================
# 5. CORE ANALYSIS & REPORT RENDERING (24 RELIABLE METRICS)
# ==============================================================================

if t is not None and f_total is not None and len(f_total) > 0:
    
    sf = moving_average(f_total, filter_size)
    sl = moving_average(f_left, filter_size)
    sr = moving_average(f_right, filter_size)
    
    n_samples = len(sf)
    g = 9.80665
    quiet_samples = max(1, min(int(0.5 / dt), n_samples))
    
    bw = np.mean(sf[:quiet_samples])
    bw_left = np.mean(sl[:quiet_samples])
    bw_right = np.mean(sr[:quiet_samples])
    
    force_sd = np.std(sf[:quiet_samples])
    mass = bw / g
    mass_left = bw_left / g if bw_left > 0 else mass * 0.5
    mass_right = bw_right / g if bw_right > 0 else mass * 0.5
    
    # --- FOOLPROOF SUB-PHASE BOUNDARIES DETECTION ---
    # 1. Find the true main jump peak force
    peak_overall_idx = np.argmax(sf)
    flight_threshold = 15.0
    
    # 2. Find flight phase (force < 15N) strictly AFTER the peak force
    flight_candidates = np.where((t > t[peak_overall_idx]) & (sf < flight_threshold))[0]
    
    if len(flight_candidates) > 5:
        # Group contiguous flight frames to find the true airborne segment
        splits = np.where(np.diff(flight_candidates) > 1)[0]
        segments = np.split(flight_candidates, splits + 1)
        longest_seg = max(segments, key=len)
        tIdx_auto = longest_seg[0]  # Take-off frame (onset of flight)
        lIdx_auto = min(n_samples - 1, longest_seg[-1] + int(0.01 / dt)) # Landing frame
    else:
        # Fallback if no clean flight segment found
        tIdx_auto = int(n_samples * 0.6)
        lIdx_auto = int(n_samples * 0.7)

    tIdx_auto = min(tIdx_auto, n_samples - 1)
    lIdx_auto = min(lIdx_auto, n_samples - 1)

    # 3. Find Braking Onset (Minimum force between start and takeoff peak)
    search_start = max(0, int(quiet_samples))
    search_end = min(tIdx_auto, int(peak_overall_idx))
    if search_end > search_start:
        min_force_rel_idx = np.argmin(sf[search_start:search_end])
        bIdx_auto = search_start + min_force_rel_idx
    else:
        bIdx_auto = max(0, tIdx_auto - int(0.3 / dt))

    # 4. Find Unweighting Onset (Where force first drops below 98% of Bodyweight before braking)
    drop_candidates = np.where((t < t[bIdx_auto]) & (sf < bw * 0.98))[0]
    sIdx_auto = drop_candidates[0] if len(drop_candidates) > 0 else max(0, bIdx_auto - int(0.4 / dt))

    # 5. Find Propulsive Onset (V = 0 crossing between braking min and takeoff)
    vel_temp = np.cumsum((sf[sIdx_auto:tIdx_auto + 1] - bw) / mass) * dt
    b_rel = max(0, bIdx_auto - sIdx_auto)
    zero_vel_matches = np.where(vel_temp[b_rel:] >= 0)[0]
    zIdx_auto = (bIdx_auto + zero_vel_matches[0]) if len(zero_vel_matches) > 0 else bIdx_auto
    zIdx_auto = min(zIdx_auto, n_samples - 1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Phase Adjustment")
    
    t_start = st.sidebar.slider("Start (Unweighting Onset)", 0.0, float(t[-1]), float(t[sIdx_auto]), step=0.005)
    t_braking = st.sidebar.slider("Braking Onset (Min Force)", 0.0, float(t[-1]), float(t[bIdx_auto]), step=0.005)
    t_split = st.sidebar.slider("Propulsive Onset (V=0)", 0.0, float(t[-1]), float(t[zIdx_auto]), step=0.005)
    t_takeoff = st.sidebar.slider("Take-off", 0.0, float(t[-1]), float(t[tIdx_auto]), step=0.005)

    # Convert UI times to secure indices
    sIdx = min(max(0, int(round(t_start / dt))), n_samples - 1)
    bIdx = min(max(0, int(round(t_braking / dt))), n_samples - 1)
    zIdx = min(max(0, int(round(t_split / dt))), n_samples - 1)
    tIdx = min(max(0, int(round(t_takeoff / dt))), n_samples - 1)
    
    lIdx_matches = np.where((t > t[tIdx] + 0.05) & (sf >= flight_threshold))[0]
    lIdx = lIdx_matches[0] if len(lIdx_matches) > 0 else n_samples - 1
    lIdx = min(lIdx, n_samples - 1)

    # Integration
    vel_total = np.zeros(n_samples)
    vel_l = np.zeros(n_samples)
    vel_r = np.zeros(n_samples)
    disp_total = np.zeros(n_samples)

    cV = cVL = cVR = cD = 0.0

    for i in range(sIdx, min(tIdx + 1, n_samples)):
        cV += ((sf[i] - bw) / mass) * dt
        cVL += ((sl[i] - bw_left) / mass_left) * dt
        cVR += ((sr[i] - bw_right) / mass_right) * dt
        vel_total[i] = cV
        vel_l[i] = cVL
        vel_r[i] = cVR
        cD += cV * dt
        disp_total[i] = cD

    contraction_time = max(dt, t[tIdx] - t[sIdx])
    propulsive_dur = max(0.0, t[tIdx] - t[zIdx])
    flight_dur = max(0.0, t[lIdx] - t[tIdx])

    # 24 Reliable Metrics (Anicic et al., 2023)
    jh_flight = (g * (flight_dur ** 2)) / 8.0 * 100.0  # cm
    v_takeoff = vel_total[tIdx]  # m/s
    jh_impulse = ((v_takeoff ** 2) / (2.0 * g)) * 100.0  # cm
    rsi_modified = (jh_flight / 100.0) / contraction_time if contraction_time > 0 else 0.0
    peak_v_prop = np.max(vel_total[zIdx:min(tIdx + 1, n_samples)]) if len(vel_total[zIdx:min(tIdx + 1, n_samples)]) > 0 else 0.0

    unweight_f = sf[sIdx:min(bIdx + 1, n_samples)]
    unweight_impulse = calc_impulse(unweight_f, dt)

    brak_f = sf[bIdx:min(zIdx + 1, n_samples)]
    brak_fl = sl[bIdx:min(zIdx + 1, n_samples)]
    brak_fr = sr[bIdx:min(zIdx + 1, n_samples)]
    brak_v = vel_total[bIdx:min(zIdx + 1, n_samples)]
    brak_p = brak_f * brak_v

    avg_brak_fl = calc_avg(brak_fl)
    avg_brak_fr = calc_avg(brak_fr)
    avg_brak_f = calc_avg(brak_f)
    avg_brak_p = calc_avg(brak_p)
    brak_impulse = calc_impulse(brak_f, dt)
    brak_impulse_l = calc_impulse(brak_fl, dt)
    brak_impulse_r = calc_impulse(brak_fr, dt)
    peak_v_neg = np.min(vel_total[sIdx:min(zIdx + 1, n_samples)]) if len(vel_total[sIdx:min(zIdx + 1, n_samples)]) > 0 else 0.0

    prop_f = sf[zIdx:min(tIdx + 1, n_samples)]
    prop_fl = sl[zIdx:min(tIdx + 1, n_samples)]
    prop_fr = sr[zIdx:min(tIdx + 1, n_samples)]
    prop_p = prop_f * vel_total[zIdx:min(tIdx + 1, n_samples)]

    avg_prop_fl = calc_avg(prop_fl)
    avg_prop_fr = calc_avg(prop_fr)
    avg_prop_f = calc_avg(prop_f)
    peak_prop_fl = np.max(prop_fl) if len(prop_fl) > 0 else 0.0
    peak_prop_fr = np.max(prop_fr) if len(prop_fr) > 0 else 0.0
    peak_prop_f = np.max(prop_f) if len(prop_f) > 0 else 0.0
    
    peak_brak_fl = np.max(brak_fl) if len(brak_fl) > 0 else 0.0
    peak_brak_fr = np.max(brak_fr) if len(brak_fr) > 0 else 0.0
    peak_brak_f = np.max(brak_f) if len(brak_f) > 0 else 0.0

    peak_prop_p = np.max(prop_p) if len(prop_p) > 0 else 0.0
    avg_prop_p = calc_avg(prop_p)

    prop_impulse = calc_impulse(prop_f, dt)
    prop_impulse_l = calc_impulse(prop_fl, dt)
    prop_impulse_r = calc_impulse(prop_fr, dt)
    positive_impulse = brak_impulse + prop_impulse

    land_impulse = 0.0
    land_impulse_l = 0.0
    land_impulse_r = 0.0
    if lIdx > tIdx and lIdx < n_samples:
        land_end_idx = min(n_samples, lIdx + int(0.5 / dt))
        land_impulse = calc_impulse(sf[lIdx:land_end_idx], dt)
        land_impulse_l = calc_impulse(sl[lIdx:land_end_idx], dt)
        land_impulse_r = calc_impulse(sr[lIdx:land_end_idx], dt)

    com_depth = abs(disp_total[zIdx] - disp_total[sIdx]) * 100.0  # cm
    com_takeoff = disp_total[tIdx] * 100.0  # cm
    leg_stiffness = (peak_brak_f / (com_depth / 100.0)) if com_depth > 0 else 0.0
    flight_jump_ratio = flight_dur / contraction_time if contraction_time > 0 else 0.0

    # --- REPORT STRUCTURED BY 4 PCA COMPONENTS (ANICIC ET AL., 2023) ---
    report = {
        "1. Performance Component (59% Variance)": {
            "Jump Height - Flight Time (cm)": {"Left": "-", "Right": "-", "Total": f"{jh_flight:.1f}", "Deficit": "-"},
            "Jump Height - Impulse-Momentum (cm)": {"Left": "-", "Right": "-", "Total": f"{jh_impulse:.1f}", "Deficit": "-"},
            "Flight Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{flight_dur:.2f}", "Deficit": "-"},
            "Take-off Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{v_takeoff:.2f}", "Deficit": "-"},
            "Peak Propulsive Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{peak_v_prop:.2f}", "Deficit": "-"},
            "RSI Modified (AU)": {"Left": "-", "Right": "-", "Total": f"{rsi_modified:.2f}", "Deficit": "-"},
            "Landing Impulse (N·s)": {"Left": f"{land_impulse_l:.0f}", "Right": f"{land_impulse_r:.0f}", "Total": f"{land_impulse:.0f}", "Deficit": calc_deficit_str(land_impulse_l, land_impulse_r)},
            "Peak Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{peak_prop_p:.0f}", "Deficit": "-"},
            "Mean Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{avg_prop_p:.0f}", "Deficit": "-"},
            "Propulsive Impulse (N·s)": {"Left": f"{prop_impulse_l:.0f}", "Right": f"{prop_impulse_r:.0f}", "Total": f"{prop_impulse:.0f}", "Deficit": calc_deficit_str(prop_impulse_l, prop_impulse_r)},
            "Positive Impulse (N·s)": {"Left": "-", "Right": "-", "Total": f"{positive_impulse:.0f}", "Deficit": "-"},
            "COM Height at Take-off (cm)": {"Left": "-", "Right": "-", "Total": f"{com_takeoff:.1f}", "Deficit": "-"}
        },
        "2. Eccentric Component (16% Variance)": {
            "Mean Force during Breaking Phase (N)": {"Left": f"{avg_brak_fl:.0f}", "Right": f"{avg_brak_fr:.0f}", "Total": f"{avg_brak_f:.0f}", "Deficit": calc_deficit_str(avg_brak_fl, avg_brak_fr)},
            "Braking Impulse (N·s)": {"Left": f"{brak_impulse_l:.0f}", "Right": f"{brak_impulse_r:.0f}", "Total": f"{brak_impulse:.0f}", "Deficit": calc_deficit_str(brak_impulse_l, brak_impulse_r)},
            "Mean Power during Breaking Phase (W)": {"Left": "-", "Right": "-", "Total": f"{abs(avg_brak_p):.0f}", "Deficit": "-"},
            "Unloading Impulse (N·s)": {"Left": "-", "Right": "-", "Total": f"{unweight_impulse:.0f}", "Deficit": "-"},
            "Peak Negative Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{peak_v_neg:.2f}", "Deficit": "-"}
        },
        "3. Concentric Component (11% Variance)": {
            "Peak Force during Propulsive Phase (N)": {"Left": f"{peak_prop_fl:.0f}", "Right": f"{peak_prop_fr:.0f}", "Total": f"{peak_prop_f:.0f}", "Deficit": calc_deficit_str(peak_prop_fl, peak_prop_fr)},
            "Mean Force during Propulsive Phase (N)": {"Left": f"{avg_prop_fl:.0f}", "Right": f"{avg_prop_fr:.0f}", "Total": f"{avg_prop_f:.0f}", "Deficit": calc_deficit_str(avg_prop_fl, avg_prop_fr)},
            "Peak Force during Breaking Phase (N)": {"Left": f"{peak_brak_fl:.0f}", "Right": f"{peak_brak_fr:.0f}", "Total": f"{peak_brak_f:.0f}", "Deficit": calc_deficit_str(peak_brak_fl, peak_brak_fr)}
        },
        "4. Jump Strategy Component (6% Variance)": {
            "Propulsive Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{propulsive_dur:.2f}", "Deficit": "-"},
            "Countermovement Center of Mass Depth (cm)": {"Left": "-", "Right": "-", "Total": f"{com_depth:.1f}", "Deficit": "-"},
            "Leg Stiffness (N/m)": {"Left": "-", "Right": "-", "Total": f"{leg_stiffness:.0f}" if leg_stiffness > 0 else "N/A", "Deficit": "-"},
            "Flight Time to Jump Time Ratio (AU)": {"Left": "-", "Right": "-", "Total": f"{flight_jump_ratio:.2f}", "Deficit": "-"}
        }
    }

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
    fig_force.add_annotation(x=t_split + (t_takeoff - t_split)/2, y=max_f * 1.05, text="PROPULSIVE", showarrow=False, font=dict(color='#4d2994', size=10))

    fig_force.update_layout(title="FORCE-TIME ANALYSIS & SUB-PHASES", xaxis_title="Time (s)", yaxis_title="Force (N)", height=420)
    st.plotly_chart(fig_force, use_container_width=True)

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
    fig_deficit.add_trace(go.Scatter(x=t, y=deficits, name="Asymmetry", fill='tozeroy', fillcolor='rgba(77, 41, 148, 0.15)', line=dict(color='#4d2994', width=2)))
    fig_deficit.add_hrect(y0=-threshold_alert, y1=threshold_alert, fillcolor="rgba(34, 197, 94, 0.15)", line_width=0)
    fig_deficit.update_layout(title="L/R ASYMMETRY % (Threshold Alert)", xaxis_title="Time (s)", yaxis_title="Deficit %", yaxis_range=[-55, 55], height=260)
    st.plotly_chart(fig_deficit, use_container_width=True)

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
    
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.download_button(
        label="📥 Download A4 PDF Report",
        data=pdf_bytes,
        file_name="Prima_Motion_CMJ_Report.pdf",
        mime="application/pdf",
        key="main_download_btn"
    )

else:
    st.info("Please upload data file(s) to begin analysis.")
