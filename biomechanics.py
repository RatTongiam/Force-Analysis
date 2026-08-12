import numpy as np
import pandas as pd

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

def detect_phases_sequential(t, sf, dt, quiet_samples):
    n_samples = len(sf)
    g = 9.80665
    bw = np.mean(sf[:quiet_samples])
    force_sd = np.std(sf[:quiet_samples])
    mass = bw / g

    unweight_cands = np.where(sf < bw - 3 * force_sd)[0]
    sIdx_auto = unweight_cands[0] if len(unweight_cands) > 0 else int(quiet_samples)

    search_end_b = min(sIdx_auto + int(1.2 / dt), n_samples)
    bIdx_auto = (sIdx_auto + np.argmin(sf[sIdx_auto:search_end_b])) if search_end_b > sIdx_auto else sIdx_auto

    vel_temp = np.cumsum((sf[sIdx_auto:] - bw) / mass) * dt
    b_rel = max(0, bIdx_auto - sIdx_auto)
    zero_crossings = np.where(vel_temp[b_rel:] >= 0)[0]
    zIdx_auto = (bIdx_auto + zero_crossings[0]) if len(zero_crossings) > 0 else bIdx_auto

    flight_threshold = 15.0
    flight_cands = np.where((np.arange(n_samples) > zIdx_auto) & (sf < flight_threshold))[0]
    if len(flight_cands) > 0:
        tIdx_auto = flight_cands[0]
        l_cands = np.where((np.arange(n_samples) > tIdx_auto + int(0.05 / dt)) & (sf >= flight_threshold))[0]
        lIdx_auto = l_cands[0] if len(l_cands) > 0 else n_samples - 1
    else:
        tIdx_auto = min(n_samples - 1, zIdx_auto + int(0.3 / dt))
        lIdx_auto = min(n_samples - 1, tIdx_auto + int(0.1 / dt))

    return sIdx_auto, bIdx_auto, zIdx_auto, tIdx_auto, lIdx_auto

def calculate_metrics(t, sf, sl, sr, dt, sIdx, bIdx, zIdx, tIdx, lIdx):
    n_samples = len(sf)
    g = 9.80665
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
