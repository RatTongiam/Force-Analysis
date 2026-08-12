import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.ndimage import label

def butter_lowpass_filter(data, cutoff=10.0, fs=2000.0, order=4):
    """
    Standard Biomechanical 4th-order Zero-Lag Low-Pass Butterworth Filter
    (Anicic et al., 2023 / Street et al., 2018)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def moving_average(arr, window):
    if window <= 1:
        return arr
    return pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean().values

def calc_avg(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0

def calc_impulse(arr, dt):
    return np.sum(arr) * dt

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

def detect_phases_sequential(t, sf_raw, dt, quiet_samples):
    n_samples = len(sf_raw)
    fs = 1.0 / dt
    g = 9.80665

    # --- 1. FILTER RAW SIGNAL WITH 10Hz BUTTERWORTH FILTER ---
    sf = butter_lowpass_filter(sf_raw, cutoff=10.0, fs=fs, order=4)
    
    bw = np.mean(sf[:quiet_samples])
    force_sd = np.std(sf[:quiet_samples])
    mass = bw / g

    # --- 2. ANCHOR: MAIN AIRBORNE FLIGHT PHASE ---
    flight_threshold = 20.0
    flight_mask = sf < flight_threshold
    labeled, num_features = label(flight_mask)
    
    if num_features > 0:
        block_lengths = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        main_flight_label = np.argmax(block_lengths) + 1
        flight_indices = np.where(labeled == main_flight_label)[0]
        
        tIdx_auto = flight_indices[0]  # Take-off
        lIdx_auto = flight_indices[-1] # Landing
    else:
        tIdx_auto = int(n_samples * 0.7)
        lIdx_auto = int(n_samples * 0.8)

    # --- 3. PEAK PROPULSION BEFORE TAKEOFF ---
    prop_search_start = max(0, tIdx_auto - int(1.5 * fs))
    if tIdx_auto > prop_search_start:
        peak_prop_idx = prop_search_start + np.argmax(sf[prop_search_start:tIdx_auto])
    else:
        peak_prop_idx = max(0, tIdx_auto - 1)

    # --- 4. BRAKING MINIMUM FORCE ---
    if peak_prop_idx > prop_search_start:
        bIdx_auto = prop_search_start + np.argmin(sf[prop_search_start:peak_prop_idx])
    else:
        bIdx_auto = max(0, peak_prop_idx - int(0.2 * fs))

    # --- 5. UNWEIGHTING ONSET (BACKWARDS FROM BRAKING MIN FORCE) ---
    # Threshold: Force drops below BW - 5*SD or BW - 2.5% BW
    dev = max(force_sd * 5.0, bw * 0.025, 10.0)
    bw_crossings = np.where((np.arange(n_samples) < bIdx_auto) & (sf >= bw - dev))[0]
    
    if len(bw_crossings) > 0:
        sIdx_auto = bw_crossings[-1]
    else:
        sIdx_auto = max(0, bIdx_auto - int(0.4 * fs))

    # --- 6. PROPULSIVE ONSET (V = 0 CROSSING) ---
    vel_temp = np.cumsum((sf[sIdx_auto:tIdx_auto + 1] - bw) / mass) * dt
    b_rel = max(0, bIdx_auto - sIdx_auto)
    zero_crossings = np.where(vel_temp[b_rel:] >= 0)[0]
    if len(zero_crossings) > 0:
        zIdx_auto = bIdx_auto + zero_crossings[0]
    else:
        zIdx_auto = bIdx_auto

    # Logical Constraints Order Bounding
    tIdx_auto = min(tIdx_auto, n_samples - 1)
    lIdx_auto = min(lIdx_auto, n_samples - 1)
    zIdx_auto = min(zIdx_auto, tIdx_auto)
    bIdx_auto = min(bIdx_auto, zIdx_auto)
    sIdx_auto = min(sIdx_auto, bIdx_auto)

    return sIdx_auto, bIdx_auto, zIdx_auto, tIdx_auto, lIdx_auto

def calculate_metrics(t, sf_raw, sl_raw, sr_raw, dt, sIdx, bIdx, zIdx, tIdx, lIdx):
    n_samples = len(sf_raw)
    fs = 1.0 / dt
    g = 9.80665

    # Apply Butterworth Filter to raw signals
    sf = butter_lowpass_filter(sf_raw, cutoff=10.0, fs=fs, order=4)
    sl = butter_lowpass_filter(sl_raw, cutoff=10.0, fs=fs, order=4)
    sr = butter_lowpass_filter(sr_raw, cutoff=10.0, fs=fs, order=4)

    quiet_samples = max(1, min(int(0.5 / dt), n_samples))
    
    bw = np.mean(sf[:quiet_samples])
    bw_left = np.mean(sl[:quiet_samples])
    bw_right = np.mean(sr[:quiet_samples])
    
    mass = bw / g
    mass_left = bw_left / g if bw_left > 0 else mass * 0.5
    mass_right = bw_right / g if bw_right > 0 else mass * 0.5

    vel_total = np.zeros(n_samples)
    disp_total = np.zeros(n_samples)
    cV = cD = 0.0

    for i in range(sIdx, min(tIdx + 1, n_samples)):
        cV += ((sf[i] - bw) / mass) * dt
        vel_total[i] = cV
        cD += cV * dt
        disp_total[i] = cD

    contraction_time = max(dt, t[tIdx] - t[sIdx])
    propulsive_dur = max(0.0, t[tIdx] - t[zIdx])
    flight_dur = max(0.0, t[lIdx] - t[tIdx])

    jh_flight = (g * (flight_dur ** 2)) / 8.0 * 100.0
    v_takeoff = vel_total[tIdx]
    jh_impulse = ((v_takeoff ** 2) / (2.0 * g)) * 100.0
    rsi_modified = (jh_flight / 100.0) / contraction_time if contraction_time > 0 else 0.0
    peak_v_prop = np.max(vel_total[zIdx:min(tIdx + 1, n_samples)]) if len(vel_total[zIdx:min(tIdx + 1, n_samples)]) > 0 else 0.0

    unweight_f = sf[sIdx:min(bIdx + 1, n_samples)]
    unweight_impulse = calc_impulse(unweight_f, dt)

    brak_f = sf[bIdx:min(zIdx + 1, n_samples)]
    brak_fl = sl[bIdx:min(zIdx + 1, n_samples)]
    brak_fr = sr[bIdx:min(zIdx + 1, n_samples)]
    brak_p = brak_f * vel_total[bIdx:min(zIdx + 1, n_samples)]

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

    land_impulse = land_impulse_l = land_impulse_r = 0.0
    if lIdx > tIdx and lIdx < n_samples:
        land_end_idx = min(n_samples, lIdx + int(0.5 / dt))
        land_impulse = calc_impulse(sf[lIdx:land_end_idx], dt)
        land_impulse_l = calc_impulse(sl[lIdx:land_end_idx], dt)
        land_impulse_r = calc_impulse(sr[lIdx:land_end_idx], dt)

    com_depth = abs(disp_total[zIdx] - disp_total[sIdx]) * 100.0
    com_takeoff = disp_total[tIdx] * 100.0
    leg_stiffness = (peak_brak_f / (com_depth / 100.0)) if com_depth > 0 else 0.0
    flight_jump_ratio = flight_dur / contraction_time if contraction_time > 0 else 0.0

    return {
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
            "Mean Force during Braking Phase (N)": {"Left": f"{avg_brak_fl:.0f}", "Right": f"{avg_brak_fr:.0f}", "Total": f"{avg_brak_f:.0f}", "Deficit": calc_deficit_str(avg_brak_fl, avg_brak_fr)},
            "Braking Impulse (N·s)": {"Left": f"{brak_impulse_l:.0f}", "Right": f"{brak_impulse_r:.0f}", "Total": f"{brak_impulse:.0f}", "Deficit": calc_deficit_str(brak_impulse_l, brak_impulse_r)},
            "Mean Power during Braking Phase (W)": {"Left": "-", "Right": "-", "Total": f"{abs(avg_brak_p):.0f}", "Deficit": "-"},
            "Unloading Impulse (N·s)": {"Left": "-", "Right": "-", "Total": f"{unweight_impulse:.0f}", "Deficit": "-"},
            "Peak Negative Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{peak_v_neg:.2f}", "Deficit": "-"}
        },
        "3. Concentric Component (11% Variance)": {
            "Peak Force during Propulsive Phase (N)": {"Left": f"{peak_prop_fl:.0f}", "Right": f"{peak_prop_fr:.0f}", "Total": f"{peak_prop_f:.0f}", "Deficit": calc_deficit_str(peak_prop_fl, peak_prop_fr)},
            "Mean Force during Propulsive Phase (N)": {"Left": f"{avg_prop_fl:.0f}", "Right": f"{avg_prop_fr:.0f}", "Total": f"{avg_prop_f:.0f}", "Deficit": calc_deficit_str(avg_prop_fl, avg_prop_fr)},
            "Peak Force during Braking Phase (N)": {"Left": f"{peak_brak_fl:.0f}", "Right": f"{peak_brak_fr:.0f}", "Total": f"{peak_brak_f:.0f}", "Deficit": calc_deficit_str(peak_brak_fl, peak_brak_fr)}
        },
        "4. Jump Strategy Component (6% Variance)": {
            "Propulsive Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{propulsive_dur:.2f}", "Deficit": "-"},
            "Countermovement Center of Mass Depth (cm)": {"Left": "-", "Right": "-", "Total": f"{com_depth:.1f}", "Deficit": "-"},
            "Leg Stiffness (N/m)": {"Left": "-", "Right": "-", "Total": f"{leg_stiffness:.0f}" if leg_stiffness > 0 else "N/A", "Deficit": "-"},
            "Flight Time to Jump Time Ratio (AU)": {"Left": "-", "Right": "-", "Total": f"{flight_jump_ratio:.2f}", "Deficit": "-"}
        }
    }
