import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.ndimage import label

def butter_lowpass_filter(data, cutoff=50.0, fs=2000.0, order=4):
    if len(data) <= order * 3:
        return np.array(data, dtype=float)
    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    elif normal_cutoff <= 0.0:
        normal_cutoff = 0.01

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    pad_l = min(150, len(data) - 1)
    y = filtfilt(b, a, data, padlen=pad_l)
    return y

def apply_signal_filter(data, filter_type="Butterworth LPF", cutoff=50.0, fs=1000.0, window_size=15):
    data_arr = np.array(data, dtype=float)
    if len(data_arr) == 0:
        return data_arr

    if filter_type == "Butterworth LPF":
        filtered = butter_lowpass_filter(data_arr, cutoff=cutoff, fs=fs, order=4)
    elif filter_type == "Moving Average":
        if window_size <= 1:
            filtered = data_arr
        else:
            filtered = pd.Series(data_arr).rolling(window=window_size, center=True, min_periods=1).mean().values
    else:
        filtered = data_arr

    return filtered

def moving_average(arr, window):
    if window <= 1:
        return arr
    return pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean().values

def calc_avg(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0

def calc_impulse(arr, dt):
    """
    คำนวณ Impulse ด้วยวิธี Trapezoidal Rule
    """
    if len(arr) < 2:
        return float(np.sum(arr) * dt)
    return float(np.trapezoid(arr, dx=dt))

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

def detect_phases_sequential(t, sf_raw, dt, quiet_samples=None, filter_type="Butterworth LPF", cutoff=50.0, sl_raw=None, sr_raw=None):
    n_samples = len(sf_raw)
    fs = 1.0 / dt
    g = 9.80665

    # 1. กรองสัญญาณเพื่อค้นหา Flight Block ด้วย scipy.ndimage.label
    sf = apply_signal_filter(sf_raw, filter_type=filter_type, cutoff=cutoff, fs=fs, window_size=15)
    
    win_bw = min(int(1.0 * fs), n_samples) if quiet_samples is None else quiet_samples
    bw = np.mean(sf[:win_bw])
    force_sd = np.std(sf[:win_bw])
    mass = bw / g if bw > 0 else 70.0

    flight_threshold = 25.0
    flight_mask = sf < flight_threshold
    labeled, num_features = label(flight_mask)
    
    tIdx_auto = None
    lIdx_auto = None

    if num_features > 0:
        valid_blocks = []
        for i in range(1, num_features + 1):
            idxs = np.where(labeled == i)[0]
            if len(idxs) > int(0.10 * fs) and idxs[0] > int(0.5 * fs):
                valid_blocks.append((i, len(idxs), idxs[0], idxs[-1]))
        
        if valid_blocks:
            best_block = max(valid_blocks, key=lambda x: x[1])
            tIdx_auto = best_block[2]
            lIdx_auto = best_block[3]

    if tIdx_auto is None:
        global_peak_idx = int(0.5 * fs) + np.argmax(sf[int(0.5 * fs):])
        tIdx_auto = max(int(0.5 * fs), global_peak_idx - int(0.2 * fs))
        lIdx_auto = min(n_samples - 1, tIdx_auto + int(0.4 * fs))

    # คำนวณ Residual Force Offset ในช่วง Flight Phase เพื่อนำไปหักล้าง Linear Baseline Drift
    offset_l, offset_r = 0.0, 0.0
    if sl_raw is not None and sr_raw is not None and lIdx_auto > tIdx_auto:
        f_mid_start = tIdx_auto + int((lIdx_auto - tIdx_auto) * 0.25)
        f_mid_end = tIdx_auto + int((lIdx_auto - tIdx_auto) * 0.75)
        if f_mid_end > f_mid_start:
            offset_l = float(np.mean(sl_raw[f_mid_start:f_mid_end]))
            offset_r = float(np.mean(sr_raw[f_mid_start:f_mid_end]))

    # 2. ค้นหา Propulsive, Braking, Start
    prop_search_start = max(0, tIdx_auto - int(1.5 * fs))
    if tIdx_auto > prop_search_start:
        peak_prop_idx = prop_search_start + np.argmax(sf[prop_search_start:tIdx_auto])
    else:
        peak_prop_idx = max(0, tIdx_auto - 1)

    if peak_prop_idx > prop_search_start:
        bIdx_auto = prop_search_start + np.argmin(sf[prop_search_start:peak_prop_idx])
    else:
        bIdx_auto = max(0, peak_prop_idx - int(0.2 * fs))

    threshold_bw = max(bw * 0.98, bw - 3 * force_sd)
    back_search_window = sf[:bIdx_auto]
    bw_crossings = np.where(back_search_window >= threshold_bw)[0]
    
    if len(bw_crossings) > 0:
        sIdx_auto = bw_crossings[-1]
    else:
        sIdx_auto = max(0, bIdx_auto - int(0.4 * fs))

    # อินทิเกรตหาจุด Zero-Velocity Crossing
    net_acc = (sf[sIdx_auto:tIdx_auto + 1] - bw) / mass
    vel_temp = np.zeros(len(net_acc))
    for k in range(1, len(net_acc)):
        vel_temp[k] = vel_temp[k - 1] + 0.5 * (net_acc[k - 1] + net_acc[k]) * dt

    b_rel = max(0, bIdx_auto - sIdx_auto)
    zero_crossings = np.where(vel_temp[b_rel:] >= 0)[0]
    if len(zero_crossings) > 0:
        zIdx_auto = bIdx_auto + zero_crossings[0]
    else:
        zIdx_auto = bIdx_auto

    # 3. ตรวจสอบ Touchdown แยกขาซ้าย-ขวา บน Baseline Corrected Signal
    lIdx_l, lIdx_r = lIdx_auto, lIdx_auto
    if sl_raw is not None and sr_raw is not None:
        sl_zeroed = sl_raw - offset_l
        sr_zeroed = sr_raw - offset_r
        
        for i in range(tIdx_auto + int(0.05 * fs), min(n_samples, tIdx_auto + int(1.2 * fs))):
            if sl_zeroed[i] >= 15.0:
                lIdx_l = i
                break
        for i in range(tIdx_auto + int(0.05 * fs), min(n_samples, tIdx_auto + int(1.2 * fs))):
            if sr_zeroed[i] >= 15.0:
                lIdx_r = i
                break

    tIdx_auto = min(tIdx_auto, n_samples - 1)
    lIdx_auto = min(lIdx_auto, n_samples - 1)
    zIdx_auto = min(zIdx_auto, tIdx_auto)
    bIdx_auto = min(bIdx_auto, zIdx_auto)
    sIdx_auto = min(sIdx_auto, bIdx_auto)

    return sIdx_auto, bIdx_auto, zIdx_auto, tIdx_auto, lIdx_auto, lIdx_l, lIdx_r, (offset_l, offset_r)

def calculate_metrics(t, sf_raw, sl_raw, sr_raw, dt, sIdx, bIdx, zIdx, tIdx, lIdx, filter_type="Butterworth LPF", cutoff=50.0, offsets=(0.0, 0.0), lIdx_l=None, lIdx_r=None):
    n_samples = len(sf_raw)
    fs = 1.0 / dt
    g = 9.80665

    offset_l, offset_r = offsets
    sl_zeroed = sl_raw - offset_l
    sr_zeroed = sr_raw - offset_r
    sf_zeroed = sl_zeroed + sr_zeroed

    # กรองสัญญาณหลังชดเชย Baseline Drift
    sf = apply_signal_filter(sf_zeroed, filter_type=filter_type, cutoff=cutoff, fs=fs, window_size=15)
    sl = apply_signal_filter(sl_zeroed, filter_type=filter_type, cutoff=cutoff, fs=fs, window_size=15)
    sr = apply_signal_filter(sr_zeroed, filter_type=filter_type, cutoff=cutoff, fs=fs, window_size=15)

    quiet_samples = max(1, min(int(0.5 / dt), n_samples))
    
    bw = np.mean(sf[:quiet_samples])
    bw_left = np.mean(sl[:quiet_samples])
    bw_right = np.mean(sr[:quiet_samples])
    
    mass = bw / g if bw > 0 else 70.0

    # คำนวณความเร็ว (Velocity) และการกระจัด (Displacement) ด้วย Trapezoidal Numerical Integration
    vel_total = np.zeros(n_samples)
    disp_total = np.zeros(n_samples)
    
    net_acc = (sf - bw) / mass
    for i in range(sIdx + 1, min(tIdx + 1, n_samples)):
        vel_total[i] = vel_total[i - 1] + 0.5 * (net_acc[i - 1] + net_acc[i]) * dt
        disp_total[i] = disp_total[i - 1] + 0.5 * (vel_total[i - 1] + vel_total[i]) * dt

    contraction_time = max(dt, t[tIdx] - t[sIdx])
    propulsive_dur = max(0.0, t[tIdx] - t[zIdx])
    flight_dur = max(0.0, t[lIdx] - t[tIdx])

    jh_flight = (g * (flight_dur ** 2)) / 8.0 * 100.0
    v_takeoff = vel_total[tIdx]
    jh_impulse = ((v_takeoff ** 2) / (2.0 * g)) * 100.0
    rsi_modified = (jh_flight / 100.0) / contraction_time if contraction_time > 0 else 0.0
    peak_v_prop = np.max(vel_total[zIdx:min(tIdx + 1, n_samples)]) if len(vel_total[zIdx:min(tIdx + 1, n_samples)]) > 0 else 0.0

    unweight_fl = sl[sIdx:min(bIdx + 1, n_samples)]
    unweight_fr = sr[sIdx:min(bIdx + 1, n_samples)]
    unweight_impulse = calc_impulse(sf[sIdx:min(bIdx + 1, n_samples)], dt)
    unweight_impulse_l = calc_impulse(unweight_fl, dt)
    unweight_impulse_r = calc_impulse(unweight_fr, dt)

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

    rfd_brak_tot = (brak_f[-1] - brak_f[0]) / (len(brak_f) * dt) if len(brak_f) > 1 else 0.0
    rfd_brak_l = (brak_fl[-1] - brak_fl[0]) / (len(brak_fl) * dt) if len(brak_fl) > 1 else 0.0
    rfd_brak_r = (brak_fr[-1] - brak_fr[0]) / (len(brak_fr) * dt) if len(brak_fr) > 1 else 0.0

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
    positive_impulse_l = brak_impulse_l + prop_impulse_l
    positive_impulse_r = brak_impulse_r + prop_impulse_r

    peak_force_idx = np.argmax(sf[sIdx:tIdx + 1]) + sIdx
    time_to_peak_force = (t[peak_force_idx] - t[sIdx]) * 1000.0

    idx_50ms = min(n_samples, zIdx + int(0.05 / dt))
    idx_100ms = min(n_samples, zIdx + int(0.10 / dt))
    
    p1_imp = calc_impulse(sf[zIdx:idx_50ms] - bw, dt) if idx_50ms > zIdx else 0.0
    p1_imp_l = calc_impulse(sl[zIdx:idx_50ms] - bw_left, dt) if idx_50ms > zIdx else 0.0
    p1_imp_r = calc_impulse(sr[zIdx:idx_50ms] - bw_right, dt) if idx_50ms > zIdx else 0.0

    p2_imp = calc_impulse(sf[idx_50ms:idx_100ms] - bw, dt) if idx_100ms > idx_50ms else 0.0
    p2_imp_l = calc_impulse(sl[idx_50ms:idx_100ms] - bw_left, dt) if idx_100ms > idx_50ms else 0.0
    p2_imp_r = calc_impulse(sr[idx_50ms:idx_100ms] - bw_right, dt) if idx_100ms > idx_50ms else 0.0

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

    touchdown_delay_ms = "-"
    if lIdx_l is not None and lIdx_r is not None:
        diff_td = (t[lIdx_l] - t[lIdx_r]) * 1000.0
        touchdown_delay_ms = f"{diff_td:+.1f} ms"

    return {
        "1. Performance Component (59% Variance)": {
            "Jump Height - Flight Time (cm)": {"Left": "-", "Right": "-", "Total": f"{jh_flight:.1f}", "Deficit": "-"},
            "Jump Height - Impulse-Momentum (cm)": {"Left": "-", "Right": "-", "Total": f"{jh_impulse:.1f}", "Deficit": "-"},
            "Flight Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{flight_dur:.3f}", "Deficit": "-"},
            "Take-off Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{v_takeoff:.2f}", "Deficit": "-"},
            "Peak Propulsive Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{peak_v_prop:.2f}", "Deficit": "-"},
            "RSI Modified (AU)": {"Left": "-", "Right": "-", "Total": f"{rsi_modified:.2f}", "Deficit": "-"},
            "Landing Impulse (N·s)": {"Left": f"{land_impulse_l:.0f}", "Right": f"{land_impulse_r:.0f}", "Total": f"{land_impulse:.0f}", "Deficit": calc_deficit_str(land_impulse_l, land_impulse_r)},
            "Initial Touchdown Delay (L-R)": {"Left": f"{t[lIdx_l]:.3f} s" if lIdx_l else "-", "Right": f"{t[lIdx_r]:.3f} s" if lIdx_r else "-", "Total": touchdown_delay_ms, "Deficit": "-"},
            "Peak Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{peak_prop_p:.0f}", "Deficit": "-"},
            "Mean Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{avg_prop_p:.0f}", "Deficit": "-"},
            "Propulsive Impulse (N·s)": {"Left": f"{prop_impulse_l:.0f}", "Right": f"{prop_impulse_r:.0f}", "Total": f"{prop_impulse:.0f}", "Deficit": calc_deficit_str(prop_impulse_l, prop_impulse_r)},
            "Positive Impulse (N·s)": {"Left": f"{positive_impulse_l:.0f}", "Right": f"{positive_impulse_r:.0f}", "Total": f"{positive_impulse:.0f}", "Deficit": calc_deficit_str(positive_impulse_l, positive_impulse_r)},
            "COM Height at Take-off (cm)": {"Left": "-", "Right": "-", "Total": f"{com_takeoff:.1f}", "Deficit": "-"}
        },
        "2. Eccentric Component (16% Variance)": {
            "Mean Force during Braking Phase (N)": {"Left": f"{avg_brak_fl:.0f}", "Right": f"{avg_brak_fr:.0f}", "Total": f"{avg_brak_f:.0f}", "Deficit": calc_deficit_str(avg_brak_fl, avg_brak_fr)},
            "Braking Impulse (N·s)": {"Left": f"{brak_impulse_l:.0f}", "Right": f"{brak_impulse_r:.0f}", "Total": f"{brak_impulse:.0f}", "Deficit": calc_deficit_str(brak_impulse_l, brak_impulse_r)},
            "Eccentric Braking RFD (N/s)": {"Left": f"{rfd_brak_l:.0f}", "Right": f"{rfd_brak_r:.0f}", "Total": f"{rfd_brak_tot:.0f}", "Deficit": calc_deficit_str(rfd_brak_l, rfd_brak_r)},
            "Mean Power during Braking Phase (W)": {"Left": "-", "Right": "-", "Total": f"{abs(avg_brak_p):.0f}", "Deficit": "-"},
            "Unloading Impulse (N·s)": {"Left": f"{unweight_impulse_l:.0f}", "Right": f"{unweight_impulse_r:.0f}", "Total": f"{unweight_impulse:.0f}", "Deficit": calc_deficit_str(unweight_impulse_l, unweight_impulse_r)},
            "Peak Negative Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{peak_v_neg:.2f}", "Deficit": "-"}
        },
        "3. Concentric Component (11% Variance)": {
            "Peak Force during Propulsive Phase (N)": {"Left": f"{peak_prop_fl:.0f}", "Right": f"{peak_prop_fr:.0f}", "Total": f"{peak_prop_f:.0f}", "Deficit": calc_deficit_str(peak_prop_fl, peak_prop_fr)},
            "Mean Force during Propulsive Phase (N)": {"Left": f"{avg_prop_fl:.0f}", "Right": f"{avg_prop_fr:.0f}", "Total": f"{avg_prop_f:.0f}", "Deficit": calc_deficit_str(avg_prop_fl, avg_prop_fr)},
            "Peak Force during Braking Phase (N)": {"Left": f"{peak_brak_fl:.0f}", "Right": f"{peak_brak_fr:.0f}", "Total": f"{peak_brak_f:.0f}", "Deficit": calc_deficit_str(peak_brak_fl, peak_brak_fr)},
            "Time to Peak Force (ms)": {"Left": "-", "Right": "-", "Total": f"{time_to_peak_force:.0f}", "Deficit": "-"},
            "P1 Concentric Impulse (0-50ms) (N·s)": {"Left": f"{p1_imp_l:.1f}", "Right": f"{p1_imp_r:.1f}", "Total": f"{p1_imp:.1f}", "Deficit": calc_deficit_str(p1_imp_l, p1_imp_r)},
            "P2 Concentric Impulse (50-100ms) (N·s)": {"Left": f"{p2_imp_l:.1f}", "Right": f"{p2_imp_r:.1f}", "Total": f"{p2_imp:.1f}", "Deficit": calc_deficit_str(p2_imp_l, p2_imp_r)}
        },
        "4. Jump Strategy Component (6% Variance)": {
            "Propulsive Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{propulsive_dur:.2f}", "Deficit": "-"},
            "Countermovement Center of Mass Depth (cm)": {"Left": "-", "Right": "-", "Total": f"{com_depth:.1f}", "Deficit": "-"},
            "Leg Stiffness (N/m)": {"Left": "-", "Right": "-", "Total": f"{leg_stiffness:.0f}" if leg_stiffness > 0 else "N/A", "Deficit": "-"},
            "Flight Time to Jump Time Ratio (AU)": {"Left": "-", "Right": "-", "Total": f"{flight_jump_ratio:.2f}", "Deficit": "-"}
        }
    }
