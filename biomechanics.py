import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.ndimage import label
from typing import Tuple, Dict, Any, Optional

def butter_lowpass_filter(data: np.ndarray, cutoff: float = 50.0, fs: float = 2000.0, order: int = 4) -> np.ndarray:
    data_arr = np.asarray(data, dtype=float)
    if len(data_arr) <= order * 3:
        return data_arr
    
    nyq = 0.5 * fs
    normal_cutoff = max(0.01, min(0.99, cutoff / nyq))
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Dynamic padlen ตาม Sampling frequency (1.0 วินาที หรือความยาวข้อมูล - 1)
    pad_l = min(len(data_arr) - 1, int(1.0 * fs))
    y = filtfilt(b, a, data_arr, padlen=pad_l)
    return y

def apply_signal_filter(data: np.ndarray, filter_type: str = "Butterworth LPF", cutoff: float = 50.0, fs: float = 1000.0, window_size: int = 15) -> np.ndarray:
    data_arr = np.asarray(data, dtype=float)
    if len(data_arr) == 0:
        return data_arr

    if filter_type == "Butterworth LPF":
        return butter_lowpass_filter(data_arr, cutoff=cutoff, fs=fs, order=4)
    elif filter_type == "Moving Average":
        if window_size <= 1:
            return data_arr
        return pd.Series(data_arr).rolling(window=window_size, center=True, min_periods=1).mean().values
    return data_arr

def calc_avg(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if len(arr) > 0 else 0.0

def calc_impulse(arr: np.ndarray, dt: float) -> float:
    if len(arr) < 2:
        return float(np.sum(arr) * dt)
    return float(np.trapezoid(arr, dx=dt))

def calc_deficit_str(val_l: Any, val_r: Any) -> str:
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

def calc_rfd_start(arr: np.ndarray, s_idx: int, dt: float, ms: float) -> float:
    if arr is None or len(arr) == 0 or s_idx < 0 or s_idx >= len(arr):
        return 0.0
    n_pts = int((ms / 1000.0) / dt)
    if n_pts <= 0:
        return 0.0
    target_idx = s_idx + n_pts
    if target_idx < len(arr):
        return float((arr[target_idx] - arr[s_idx]) / (ms / 1000.0))
    return 0.0

def calc_max_win_rfd(arr: np.ndarray, s_idx: int, e_idx: int, dt: float, win_ms: float = 20.0) -> float:
    if arr is None or len(arr) == 0 or s_idx < 0 or e_idx >= len(arr) or e_idx <= s_idx:
        return 0.0
    w = max(1, int((win_ms / 1000.0) / dt))
    if (e_idx - s_idx) <= w:
        return 0.0
    seg = arr[s_idx:e_idx + 1]
    if len(seg) <= w:
        return 0.0
    slopes = (seg[w:] - seg[:-w]) / (w * dt)
    return float(np.max(slopes)) if len(slopes) > 0 else 0.0

def detect_phases_sequential(t: np.ndarray, sf_raw: np.ndarray, dt: float, quiet_samples: Optional[int] = None, filter_type: str = "Butterworth LPF", cutoff: float = 50.0, sl_raw: Optional[np.ndarray] = None, sr_raw: Optional[np.ndarray] = None) -> Tuple[int, int, int, int, int, int, int, Tuple[float, float]]:
    n_samples = len(sf_raw)
    fs = 1.0 / dt
    g = 9.80665

    sf = apply_signal_filter(sf_raw, filter_type=filter_type, cutoff=cutoff, fs=fs, window_size=15)
    win_bw = min(int(1.0 * fs), n_samples) if quiet_samples is None else quiet_samples
    bw = np.mean(sf[:win_bw])
    force_sd = np.std(sf[:win_bw])
    mass = bw / g if bw > 0 else 70.0

    flight_threshold = 25.0
    flight_mask = sf < flight_threshold
    labeled, num_features = label(flight_mask)
    
    tIdx_auto, lIdx_auto = None, None
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

    offset_l, offset_r = 0.0, 0.0
    if sl_raw is not None and sr_raw is not None and lIdx_auto > tIdx_auto:
        f_mid_start = tIdx_auto + int((lIdx_auto - tIdx_auto) * 0.25)
        f_mid_end = tIdx_auto + int((lIdx_auto - tIdx_auto) * 0.75)
        if f_mid_end > f_mid_start:
            offset_l = float(np.mean(sl_raw[f_mid_start:f_mid_end]))
            offset_r = float(np.mean(sr_raw[f_mid_start:f_mid_end]))

    prop_search_start = max(0, tIdx_auto - int(1.5 * fs))
    peak_prop_idx = prop_search_start + np.argmax(sf[prop_search_start:tIdx_auto]) if tIdx_auto > prop_search_start else max(0, tIdx_auto - 1)
    bIdx_auto = prop_search_start + np.argmin(sf[prop_search_start:peak_prop_idx]) if peak_prop_idx > prop_search_start else max(0, peak_prop_idx - int(0.2 * fs))

    threshold_bw = max(bw * 0.98, bw - 3 * force_sd)
    back_search_window = sf[:bIdx_auto]
    bw_crossings = np.where(back_search_window >= threshold_bw)[0]
    sIdx_auto = bw_crossings[-1] if len(bw_crossings) > 0 else max(0, bIdx_auto - int(0.4 * fs))

    net_acc = (sf[sIdx_auto:tIdx_auto + 1] - bw) / mass
    vel_temp = np.zeros(len(net_acc))
    for k in range(1, len(net_acc)):
        vel_temp[k] = vel_temp[k - 1] + 0.5 * (net_acc[k - 1] + net_acc[k]) * dt

    b_rel = max(0, bIdx_auto - sIdx_auto)
    zero_crossings = np.where(vel_temp[b_rel:] >= 0)[0]
    zIdx_auto = bIdx_auto + zero_crossings[0] if len(zero_crossings) > 0 else bIdx_auto

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

    return (
        min(sIdx_auto, bIdx_auto),
        min(bIdx_auto, zIdx_auto),
        min(zIdx_auto, tIdx_auto),
        min(tIdx_auto, n_samples - 1),
        min(lIdx_auto, n_samples - 1),
        lIdx_l, lIdx_r, (offset_l, offset_r)
    )

def calculate_metrics(t: np.ndarray, sf_raw: np.ndarray, sl_raw: np.ndarray, sr_raw: np.ndarray, dt: float, sIdx: int, bIdx: int, zIdx: int, tIdx: int, lIdx: int, filter_type: str = "Butterworth LPF", cutoff: float = 50.0, offsets: Tuple[float, float] = (0.0, 0.0), lIdx_l: Optional[int] = None, lIdx_r: Optional[int] = None, group_by: str = "phase") -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    n_samples = len(sf_raw)
    fs = 1.0 / dt
    g = 9.80665

    offset_l, offset_r = offsets
    sl_zeroed = sl_raw - offset_l
    sr_zeroed = sr_raw - offset_r
    sf_zeroed = sl_zeroed + sr_zeroed

    sf = apply_signal_filter(sf_zeroed, filter_type=filter_type, cutoff=cutoff, fs=fs, window_size=15)
    sl = apply_signal_filter(sl_zeroed, filter_type=filter_type, cutoff=cutoff, fs=fs, window_size=15)
    sr = apply_signal_filter(sr_zeroed, filter_type=filter_type, cutoff=cutoff, fs=fs, window_size=15)

    quiet_samples = max(1, min(int(0.5 / dt), n_samples))
    bw = float(np.mean(sf[:quiet_samples]))
    bw_sd = float(np.std(sf[:quiet_samples]))
    bw_left = float(np.mean(sl[:quiet_samples]))
    bw_right = float(np.mean(sr[:quiet_samples]))
    mass = bw / g if bw > 0 else 70.0

    vel_total = np.zeros(n_samples)
    disp_total = np.zeros(n_samples)
    net_acc = (sf - bw) / mass
    for i in range(sIdx + 1, min(tIdx + 1, n_samples)):
        vel_total[i] = vel_total[i - 1] + 0.5 * (net_acc[i - 1] + net_acc[i]) * dt
        disp_total[i] = disp_total[i - 1] + 0.5 * (vel_total[i - 1] + vel_total[i]) * dt

    power_total = sf * vel_total
    power_wkg = power_total / mass

    contraction_time = max(dt, t[tIdx] - t[sIdx])
    propulsive_dur = max(0.0, t[tIdx] - t[zIdx])
    unweight_dur = max(0.0, t[bIdx] - t[sIdx])
    braking_dur = max(0.0, t[zIdx] - t[bIdx])
    flight_dur = max(0.0, t[lIdx] - t[tIdx])

    v_takeoff = float(vel_total[tIdx])
    jh_impulse = float(((v_takeoff ** 2) / (2.0 * g)) * 100.0)
    jh_flight = float((g * (flight_dur ** 2)) / 8.0 * 100.0)
    
    # คำนวณ RSI modified โดยใช้ Impulse-Momentum Method ตามมาตรฐานสากล
    rsi_modified = float((jh_impulse / 100.0) / contraction_time) if contraction_time > 0 else 0.0
    peak_v_prop = float(np.max(vel_total[zIdx:min(tIdx + 1, n_samples)])) if tIdx > zIdx else 0.0
    peak_v_neg = float(np.min(vel_total[sIdx:min(zIdx + 1, n_samples)])) if zIdx > sIdx else 0.0

    # Impulse and Forces
    unweight_impulse = calc_impulse(sf[sIdx:min(bIdx + 1, n_samples)], dt)
    unweight_impulse_l = calc_impulse(sl[sIdx:min(bIdx + 1, n_samples)], dt)
    unweight_impulse_r = calc_impulse(sr[sIdx:min(bIdx + 1, n_samples)], dt)

    brak_f = sf[bIdx:min(zIdx + 1, n_samples)]
    brak_fl = sl[bIdx:min(zIdx + 1, n_samples)]
    brak_fr = sr[bIdx:min(zIdx + 1, n_samples)]
    avg_brak_fl, avg_brak_fr, avg_brak_f = calc_avg(brak_fl), calc_avg(brak_fr), calc_avg(brak_f)
    avg_brak_p = calc_avg(power_wkg[bIdx:min(zIdx + 1, n_samples)])
    brak_impulse = calc_impulse(brak_f, dt)
    brak_impulse_l = calc_impulse(brak_fl, dt)
    brak_impulse_r = calc_impulse(brak_fr, dt)
    br_net_l = float(np.trapezoid(sl[bIdx:zIdx+1] - bw_left, dx=dt)) if zIdx > bIdx else 0.0
    br_net_r = float(np.trapezoid(sr[bIdx:zIdx+1] - bw_right, dx=dt)) if zIdx > bIdx else 0.0

    rfd_brak_tot = (brak_f[-1] - brak_f[0]) / (len(brak_f) * dt) if len(brak_f) > 1 else 0.0
    rfd_brak_l = (brak_fl[-1] - brak_fl[0]) / (len(brak_fl) * dt) if len(brak_fl) > 1 else 0.0
    rfd_brak_r = (brak_fr[-1] - brak_fr[0]) / (len(brak_fr) * dt) if len(brak_fr) > 1 else 0.0

    prop_f = sf[zIdx:min(tIdx + 1, n_samples)]
    prop_fl = sl[zIdx:min(tIdx + 1, n_samples)]
    prop_fr = sr[zIdx:min(tIdx + 1, n_samples)]
    avg_prop_fl, avg_prop_fr, avg_prop_f = calc_avg(prop_fl), calc_avg(prop_fr), calc_avg(prop_f)
    peak_prop_fl = float(np.max(prop_fl)) if len(prop_fl) > 0 else 0.0
    peak_prop_fr = float(np.max(prop_fr)) if len(prop_fr) > 0 else 0.0
    peak_prop_f = float(np.max(prop_f)) if len(prop_f) > 0 else 0.0

    peak_brak_fl = float(np.max(brak_fl)) if len(brak_fl) > 0 else 0.0
    peak_brak_fr = float(np.max(brak_fr)) if len(brak_fr) > 0 else 0.0
    peak_brak_f = float(np.max(brak_f)) if len(brak_f) > 0 else 0.0

    peak_prop_p = float(np.max(power_wkg[zIdx:tIdx+1])) if tIdx > zIdx else 0.0
    avg_prop_p = calc_avg(power_wkg[zIdx:tIdx+1]) if tIdx > zIdx else 0.0

    prop_impulse = calc_impulse(prop_f, dt)
    prop_impulse_l = calc_impulse(prop_fl, dt)
    prop_impulse_r = calc_impulse(prop_fr, dt)
    pr_net_l = float(np.trapezoid(sl[zIdx:tIdx+1] - bw_left, dx=dt)) if tIdx > zIdx else 0.0
    pr_net_r = float(np.trapezoid(sr[zIdx:tIdx+1] - bw_right, dx=dt)) if tIdx > zIdx else 0.0

    positive_impulse = brak_impulse + prop_impulse
    positive_impulse_l = brak_impulse_l + prop_impulse_l
    positive_impulse_r = brak_impulse_r + prop_impulse_r
    pos_net_imp = float(np.trapezoid(np.maximum(0, sf[sIdx:tIdx+1] - bw), dx=dt))

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

    # Landing metrics
    land_search_end = min(n_samples, lIdx + int(0.5 / dt))
    pk_land_tot = float(np.max(sf[lIdx:land_search_end])) if land_search_end > lIdx else 0.0
    pk_land_l = float(np.max(sl[lIdx:land_search_end])) if land_search_end > lIdx else 0.0
    pk_land_r = float(np.max(sr[lIdx:land_search_end])) if land_search_end > lIdx else 0.0
    ttp_land_ms = float(np.argmax(sf[lIdx:land_search_end]) * dt * 1000.0) if land_search_end > lIdx else 0.0
    
    imp_250_end = min(n_samples, lIdx + int(0.25 / dt))
    land_imp_250_l = float(np.trapezoid(sl[lIdx:imp_250_end], dx=dt)) if imp_250_end > lIdx else 0.0
    land_imp_250_r = float(np.trapezoid(sr[lIdx:imp_250_end], dx=dt)) if imp_250_end > lIdx else 0.0
    land_imp_250_tot = land_imp_250_l + land_imp_250_r

    pk_idx = lIdx + np.argmax(sf[lIdx:land_search_end]) if land_search_end > lIdx else lIdx
    f_20 = 0.20 * pk_land_tot
    f_80 = 0.80 * pk_land_tot
    sub_land = sf[lIdx:pk_idx+1]
    idx_20 = np.where(sub_land >= f_20)[0]
    idx_80 = np.where(sub_land >= f_80)[0]
    load_rate = float((f_80 - f_20) / ((idx_80[0] - idx_20[0]) * dt)) if len(idx_20) > 0 and len(idx_80) > 0 and idx_80[0] > idx_20[0] else 0.0

    com_depth = abs(disp_total[zIdx] - disp_total[sIdx]) * 100.0
    com_takeoff = disp_total[tIdx] * 100.0
    leg_stiffness = (peak_brak_f / (com_depth / 100.0)) if com_depth > 0 else 0.0
    flight_jump_ratio = flight_dur / contraction_time if contraction_time > 0 else 0.0

    touchdown_delay_ms = "-"
    if lIdx_l is not None and lIdx_r is not None:
        touchdown_delay_ms = f"{(t[lIdx_l] - t[lIdx_r]) * 1000.0:+.1f} ms"

    # บรรจุข้อมูล Scalar ทั้งหมด
    scalar_metrics = {
        "bw": bw, "bw_sd": bw_sd, "mass": mass, "fs": fs,
        "jh_imp": jh_impulse, "jh_flt": jh_flight, "rsi": rsi_modified, "ppk_wkg": peak_prop_p,
        "ttt": contraction_time, "d_unw": unweight_dur, "d_brk": braking_dur, "d_pro": propulsive_dur, "d_fly": flight_dur,
        "com_depth": com_depth, "com_takeoff": com_takeoff, "v_to": v_takeoff,
        "pk_br_l": peak_brak_fl, "pk_br_r": peak_brak_fr, "br_net_l": br_net_l, "br_net_r": br_net_r,
        "br_gross_l": brak_impulse_l, "br_gross_r": brak_impulse_r,
        "avg_rfd_br_l": (peak_brak_fl - sl[bIdx]) / ((zIdx - bIdx) * dt) if zIdx > bIdx else 0.0,
        "avg_rfd_br_r": (peak_brak_fr - sr[bIdx]) / ((zIdx - bIdx) * dt) if zIdx > bIdx else 0.0,
        "win_rfd_br_l": calc_max_win_rfd(sl, bIdx, zIdx, dt, 20),
        "win_rfd_br_r": calc_max_win_rfd(sr, bIdx, zIdx, dt, 20),
        "rfd50_br_l": calc_rfd_start(sl, bIdx, dt, 50), "rfd50_br_r": calc_rfd_start(sr, bIdx, dt, 50),
        "rfd100_br_l": calc_rfd_start(sl, bIdx, dt, 100), "rfd100_br_r": calc_rfd_start(sr, bIdx, dt, 100),
        "rfd200_br_l": calc_rfd_start(sl, bIdx, dt, 200), "rfd200_br_r": calc_rfd_start(sr, bIdx, dt, 200),
        "pk_pr_l": peak_prop_fl, "pk_pr_r": peak_prop_fr, "pr_net_l": pr_net_l, "pr_net_r": pr_net_r,
        "pr_gross_l": prop_impulse_l, "pr_gross_r": prop_impulse_r,
        "avg_rfd_pr_l": (peak_prop_fl - sl[zIdx]) / ((tIdx - zIdx) * dt) if tIdx > zIdx else 0.0,
        "avg_rfd_pr_r": (peak_prop_fr - sr[zIdx]) / ((tIdx - zIdx) * dt) if tIdx > zIdx else 0.0,
        "win_rfd_pr_l": calc_max_win_rfd(sl, zIdx, tIdx, dt, 20),
        "win_rfd_pr_r": calc_max_win_rfd(sr, zIdx, tIdx, dt, 20),
        "rfd50_pr_l": calc_rfd_start(sl, zIdx, dt, 50), "rfd50_pr_r": calc_rfd_start(sr, zIdx, dt, 50),
        "rfd100_pr_l": calc_rfd_start(sl, zIdx, dt, 100), "rfd100_pr_r": calc_rfd_start(sr, zIdx, dt, 100),
        "rfd200_pr_l": calc_rfd_start(sl, zIdx, dt, 200), "rfd200_pr_r": calc_rfd_start(sr, zIdx, dt, 200),
        "pk_land_l": pk_land_l, "pk_land_r": pk_land_r, "pk_land_tot": pk_land_tot,
        "load_rate": load_rate, "land_imp_250_l": land_imp_250_l, "land_imp_250_r": land_imp_250_r,
        "land_imp_250_tot": land_imp_250_tot, "ttp_land_ms": ttp_land_ms,
        "mean_br_f": avg_brak_f, "mean_pr_f": avg_prop_f, "mean_br_p": avg_brak_p, "mean_pr_p": avg_prop_p,
        "pos_net_imp": pos_net_imp, "leg_stiff": leg_stiffness,
        "offset_l": offset_l, "offset_r": offset_r
    }

    derived_series = {
        "vel": vel_total,
        "disp": disp_total,
        "power_wkg": power_wkg,
        "sf_filtered": sf,
        "sl_filtered": sl,
        "sr_filtered": sr
    }

    # Grouping Report สำหรับแสดงในตาราง
    if group_by == "phase":
        report = {
            "1. Weighing & Onset Phase": {
                "Body Mass (kg)": {"Left": f"{bw_left/g:.1f}", "Right": f"{bw_right/g:.1f}", "Total": f"{mass:.1f}", "Deficit": calc_deficit_str(bw_left, bw_right)},
                "Body Weight (N)": {"Left": f"{bw_left:.0f}", "Right": f"{bw_right:.0f}", "Total": f"{bw:.0f}", "Deficit": calc_deficit_str(bw_left, bw_right)},
                "Time to Peak Force (ms)": {"Left": "-", "Right": "-", "Total": f"{time_to_peak_force:.0f}", "Deficit": "-"}
            },
            "2. Unweighting Phase": {
                "Unloading Duration (s)": {"Left": "-", "Right": "-", "Total": f"{unweight_dur:.3f}", "Deficit": "-"},
                "Unloading Impulse (N·s)": {"Left": f"{unweight_impulse_l:.0f}", "Right": f"{unweight_impulse_r:.0f}", "Total": f"{unweight_impulse:.0f}", "Deficit": calc_deficit_str(unweight_impulse_l, unweight_impulse_r)},
                "Peak Negative Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{peak_v_neg:.2f}", "Deficit": "-"}
            },
            "3. Braking (Eccentric) Phase": {
                "Braking Duration (s)": {"Left": "-", "Right": "-", "Total": f"{braking_dur:.3f}", "Deficit": "-"},
                "Peak Force during Braking Phase (N)": {"Left": f"{peak_brak_fl:.0f}", "Right": f"{peak_brak_fr:.0f}", "Total": f"{peak_brak_f:.0f}", "Deficit": calc_deficit_str(peak_brak_fl, peak_brak_fr)},
                "Mean Force during Braking Phase (N)": {"Left": f"{avg_brak_fl:.0f}", "Right": f"{avg_brak_fr:.0f}", "Total": f"{avg_brak_f:.0f}", "Deficit": calc_deficit_str(avg_brak_fl, avg_brak_fr)},
                "Braking Impulse (N·s)": {"Left": f"{brak_impulse_l:.0f}", "Right": f"{brak_impulse_r:.0f}", "Total": f"{brak_impulse:.0f}", "Deficit": calc_deficit_str(brak_impulse_l, brak_impulse_r)},
                "Eccentric Braking RFD (N/s)": {"Left": f"{rfd_brak_l:.0f}", "Right": f"{rfd_brak_r:.0f}", "Total": f"{rfd_brak_tot:.0f}", "Deficit": calc_deficit_str(rfd_brak_l, rfd_brak_r)},
                "Mean Power during Braking Phase (W)": {"Left": "-", "Right": "-", "Total": f"{abs(avg_brak_p*mass):.0f}", "Deficit": "-"}
            },
            "4. Propulsive (Concentric) Phase": {
                "Propulsive Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{propulsive_dur:.3f}", "Deficit": "-"},
                "Peak Force during Propulsive Phase (N)": {"Left": f"{peak_prop_fl:.0f}", "Right": f"{peak_prop_fr:.0f}", "Total": f"{peak_prop_f:.0f}", "Deficit": calc_deficit_str(peak_prop_fl, peak_prop_fr)},
                "Mean Force during Propulsive Phase (N)": {"Left": f"{avg_prop_fl:.0f}", "Right": f"{avg_prop_fr:.0f}", "Total": f"{avg_prop_f:.0f}", "Deficit": calc_deficit_str(avg_prop_fl, avg_prop_fr)},
                "Propulsive Impulse (N·s)": {"Left": f"{prop_impulse_l:.0f}", "Right": f"{prop_impulse_r:.0f}", "Total": f"{prop_impulse:.0f}", "Deficit": calc_deficit_str(prop_impulse_l, prop_impulse_r)},
                "Peak Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{peak_prop_p*mass:.0f}", "Deficit": "-"},
                "Mean Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{avg_prop_p*mass:.0f}", "Deficit": "-"},
                "P1 Concentric Impulse (0-50ms) (N·s)": {"Left": f"{p1_imp_l:.1f}", "Right": f"{p1_imp_r:.1f}", "Total": f"{p1_imp:.1f}", "Deficit": calc_deficit_str(p1_imp_l, p1_imp_r)},
                "P2 Concentric Impulse (50-100ms) (N·s)": {"Left": f"{p2_imp_l:.1f}", "Right": f"{p2_imp_r:.1f}", "Total": f"{p2_imp:.1f}", "Deficit": calc_deficit_str(p2_imp_l, p2_imp_r)}
            },
            "5. Flight & Performance Phase": {
                "Jump Height - Impulse-Momentum (cm)": {"Left": "-", "Right": "-", "Total": f"{jh_impulse:.1f}", "Deficit": "-"},
                "Jump Height - Flight Time (cm)": {"Left": "-", "Right": "-", "Total": f"{jh_flight:.1f}", "Deficit": "-"},
                "Flight Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{flight_dur:.3f}", "Deficit": "-"},
                "Take-off Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{v_takeoff:.2f}", "Deficit": "-"},
                "Peak Propulsive Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{peak_v_prop:.2f}", "Deficit": "-"},
                "RSI Modified (AU)": {"Left": "-", "Right": "-", "Total": f"{rsi_modified:.2f}", "Deficit": "-"},
                "Positive Impulse (N·s)": {"Left": f"{positive_impulse_l:.0f}", "Right": f"{positive_impulse_r:.0f}", "Total": f"{positive_impulse:.0f}", "Deficit": calc_deficit_str(positive_impulse_l, positive_impulse_r)},
                "COM Height at Take-off (cm)": {"Left": "-", "Right": "-", "Total": f"{com_takeoff:.1f}", "Deficit": "-"}
            },
            "6. Landing & Strategy Phase": {
                "Landing Impulse (N·s)": {"Left": f"{land_imp_250_l:.0f}", "Right": f"{land_imp_250_r:.0f}", "Total": f"{land_imp_250_tot:.0f}", "Deficit": calc_deficit_str(land_imp_250_l, land_imp_250_r)},
                "Initial Touchdown Delay (L-R)": {"Left": f"{t[lIdx_l]:.3f} s" if lIdx_l else "-", "Right": f"{t[lIdx_r]:.3f} s" if lIdx_r else "-", "Total": touchdown_delay_ms, "Deficit": "-"},
                "Countermovement Center of Mass Depth (cm)": {"Left": "-", "Right": "-", "Total": f"{com_depth:.1f}", "Deficit": "-"},
                "Leg Stiffness (N/m)": {"Left": "-", "Right": "-", "Total": f"{leg_stiffness:.0f}" if leg_stiffness > 0 else "N/A", "Deficit": "-"},
                "Flight Time to Jump Time Ratio (AU)": {"Left": "-", "Right": "-", "Total": f"{flight_jump_ratio:.2f}", "Deficit": "-"}
            }
        }
    else:
        report = {
            "1. Performance Component (59% Variance)": {
                "Jump Height - Flight Time (cm)": {"Left": "-", "Right": "-", "Total": f"{jh_flight:.1f}", "Deficit": "-"},
                "Jump Height - Impulse-Momentum (cm)": {"Left": "-", "Right": "-", "Total": f"{jh_impulse:.1f}", "Deficit": "-"},
                "Flight Phase Duration (s)": {"Left": "-", "Right": "-", "Total": f"{flight_dur:.3f}", "Deficit": "-"},
                "Take-off Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{v_takeoff:.2f}", "Deficit": "-"},
                "Peak Propulsive Velocity (m/s)": {"Left": "-", "Right": "-", "Total": f"{peak_v_prop:.2f}", "Deficit": "-"},
                "RSI Modified (AU)": {"Left": "-", "Right": "-", "Total": f"{rsi_modified:.2f}", "Deficit": "-"},
                "Landing Impulse (N·s)": {"Left": f"{land_imp_250_l:.0f}", "Right": f"{land_imp_250_r:.0f}", "Total": f"{land_imp_250_tot:.0f}", "Deficit": calc_deficit_str(land_imp_250_l, land_imp_250_r)},
                "Initial Touchdown Delay (L-R)": {"Left": f"{t[lIdx_l]:.3f} s" if lIdx_l else "-", "Right": f"{t[lIdx_r]:.3f} s" if lIdx_r else "-", "Total": touchdown_delay_ms, "Deficit": "-"},
                "Peak Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{peak_prop_p*mass:.0f}", "Deficit": "-"},
                "Mean Propulsive Power (W)": {"Left": "-", "Right": "-", "Total": f"{avg_prop_p*mass:.0f}", "Deficit": "-"},
                "Propulsive Impulse (N·s)": {"Left": f"{prop_impulse_l:.0f}", "Right": f"{prop_impulse_r:.0f}", "Total": f"{prop_impulse:.0f}", "Deficit": calc_deficit_str(prop_impulse_l, prop_impulse_r)},
                "Positive Impulse (N·s)": {"Left": f"{positive_impulse_l:.0f}", "Right": f"{positive_impulse_r:.0f}", "Total": f"{positive_impulse:.0f}", "Deficit": calc_deficit_str(positive_impulse_l, positive_impulse_r)},
                "COM Height at Take-off (cm)": {"Left": "-", "Right": "-", "Total": f"{com_takeoff:.1f}", "Deficit": "-"}
            },
            "2. Eccentric Component (16% Variance)": {
                "Mean Force during Braking Phase (N)": {"Left": f"{avg_brak_fl:.0f}", "Right": f"{avg_brak_fr:.0f}", "Total": f"{avg_brak_f:.0f}", "Deficit": calc_deficit_str(avg_brak_fl, avg_brak_fr)},
                "Braking Impulse (N·s)": {"Left": f"{brak_impulse_l:.0f}", "Right": f"{brak_impulse_r:.0f}", "Total": f"{brak_impulse:.0f}", "Deficit": calc_deficit_str(brak_impulse_l, brak_impulse_r)},
                "Eccentric Braking RFD (N/s)": {"Left": f"{rfd_brak_l:.0f}", "Right": f"{rfd_brak_r:.0f}", "Total": f"{rfd_brak_tot:.0f}", "Deficit": calc_deficit_str(rfd_brak_l, rfd_brak_r)},
                "Mean Power during Braking Phase (W)": {"Left": "-", "Right": "-", "Total": f"{abs(avg_brak_p*mass):.0f}", "Deficit": "-"},
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

    return report, scalar_metrics, derived_series
