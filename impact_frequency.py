"""
impact_frequency.py
ประเมินความถี่ของ impact ตอน landing และเทียบกับ natural frequency (fn) ของ force plate

วิธี (ใช้สัญญาณ RAW ที่ปรับ zero-offset แล้ว ห้ามใช้สัญญาณที่ผ่าน LPF 50 Hz):
  A. FFT   : f_c99  = ความถี่ที่สะสมพลังงาน 99% ของช่วง impact (ใช้ตัดสินใจเทียบ plate)
             f_snr  = ความถี่ที่ spectrum ตกลงต่ำกว่า noise floor x k (noise floor จากช่วง flight)
                      เป็นขอบเขตบนแบบ conservative ขึ้นกับ dynamic range ไม่ใช้ตัดสินใจ
  B. เวลา  : f_rise = 0.35 / t_rise(10-90%)   (ประมาณแบบ 1st-order / Gaussian)
             f_dur  = 1 / (4 * t_peak)        (ช่วง onset->peak เป็น 1/4 cycle ของ half-sine)
  C. เทียบ plate: ความคลาดเคลื่อน amplitude ของระบบ 2nd-order (undamped) = 1/(1-r^2) - 1, r = f/fn

ข้อจำกัด: ค่าที่ได้เป็นค่าประมาณ ไม่ใช่การวัด transfer function ของแผ่น
fn ในสภาพใช้งานจริง (แผ่น+พื้น+ผู้ทดสอบ) ต้องวัดด้วย impact (tap/hammer) test ไม่ได้ประมาณจากสัญญาณกระโดด
"""
from typing import Dict, Optional

import numpy as np
from scipy.signal.windows import tukey
from scipy.ndimage import uniform_filter1d


def _amp_spectrum(x: np.ndarray, dt: float, nfft: int):
    """Amplitude spectrum แบบ transient (หน่วย N·s) หน้าต่าง Tukey ตัด mean ก่อน"""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    w = tukey(len(x), alpha=0.2)
    X = np.fft.rfft(x * w, n=nfft) * dt
    return np.fft.rfftfreq(nfft, dt), np.abs(X)


def _cross_time(y: np.ndarray, k: int, level: float, dt: float) -> float:
    """เวลาที่ y ตัด level ระหว่าง k และ k+1 (linear interpolation)"""
    y0, y1 = y[k], y[k + 1]
    den = (y1 - y0) if y1 != y0 else 1e-12
    return (k + (level - y0) / den) * dt


def _last_below(y: np.ndarray, level: float, i_pk: int) -> Optional[int]:
    idx = np.where(y[: i_pk + 1] < level)[0]
    return int(idx[-1]) if len(idx) > 0 and idx[-1] < i_pk else None


def landing_impact_frequency(
    force: np.ndarray, fs: float, t_idx: int, l_idx: int, *,
    pre_ms: float = 40.0, post_ms: float = 250.0, peak_search_ms: float = 100.0,
    snr_k: float = 3.0, onset_floor_n: float = 10.0,
    ring_hp_hz: float = 30.0, ring_ms: float = 150.0, ring_prom: float = 4.0,
) -> Dict[str, object]:
    """
    force : แรง Fz RAW (zero-offset แล้ว) ของ 1 แผ่น หรือผลรวม
    t_idx : index take-off, l_idx : index landing (touchdown)
    """
    f = np.asarray(force, dtype=float)
    n, dt = len(f), 1.0 / fs
    out: Dict[str, object] = {"notes": []}
    notes = out["notes"]

    i0 = max(0, l_idx - int(pre_ms * fs / 1000))
    i1 = min(n, l_idx + int(post_ms * fs / 1000))
    if i1 - i0 < 32:
        notes.append("ช่วง landing สั้นเกินไปสำหรับ FFT")
        return out

    # ---- noise floor จากช่วง flight (กลาง 50%) ----
    fl0 = t_idx + int(0.25 * (l_idx - t_idx))
    fl1 = t_idx + int(0.75 * (l_idx - t_idx))
    noise = f[fl0:fl1] if (fl1 - fl0) >= 30 else None
    sigma = float(np.std(noise)) if noise is not None else None
    if noise is None:
        notes.append("flight สั้นเกินไป ใช้ noise floor ไม่ได้ (ข้าม f_snr)")

    # ---- B. Time-domain: rise time / time-to-peak ----
    j_end = min(n, l_idx + int(peak_search_ms * fs / 1000))
    i_pk = i0 + int(np.argmax(f[i0:j_end]))
    pk = float(f[i_pk])
    onset_lvl = max(onset_floor_n, 5.0 * sigma) if sigma else onset_floor_n
    k10 = _last_below(f, 0.10 * pk, i_pk)
    k90 = _last_below(f, 0.90 * pk, i_pk)
    k_on = _last_below(f, onset_lvl, i_pk)
    out["peak_force_n"] = pk
    out["t_peak_idx"] = i_pk
    if k10 is not None and k90 is not None and k_on is not None:
        t10 = _cross_time(f, k10, 0.10 * pk, dt)
        t90 = _cross_time(f, k90, 0.90 * pk, dt)
        t_on = _cross_time(f, k_on, onset_lvl, dt)
        tr = t90 - t10
        t_peak = i_pk * dt - t_on
        out["t_rise_ms"] = tr * 1000.0
        out["t_peak_ms"] = t_peak * 1000.0
        out["f_rise_hz"] = 0.35 / tr if tr > 0 else None
        out["f_dur_hz"] = 1.0 / (4.0 * t_peak) if t_peak > 0 else None
        if tr < 5 * dt:
            notes.append(f"t_rise {tr*1000:.1f} ms < 5 sample ({5*dt*1000:.0f} ms): ความละเอียดเวลาไม่พอ f_rise เป็นค่าประมาณหยาบ")
    else:
        notes.append("หา onset/10%/90% ไม่ได้ (สัญญาณเริ่มสูงอยู่แล้ว หรือไม่มี impact ชัด)")

    # ---- A. FFT: bandwidth เหนือ noise floor ----
    seg = f[i0:i1]
    nfft = 1 << int(np.ceil(np.log2(len(seg) * 4)))
    freqs, amp = _amp_spectrum(seg, dt, nfft)
    df = freqs[1] - freqs[0]
    out["freqs"], out["amp"] = freqs, amp
    cum = np.cumsum(amp ** 2)
    out["f_c99_hz"] = float(freqs[min(len(freqs) - 1, int(np.searchsorted(cum / cum[-1], 0.99)))])
    if noise is not None:
        _, namp = _amp_spectrum(noise, dt, nfft)
        floor = float(np.median(namp[freqs > 20.0])) * np.sqrt(len(seg) / len(noise))
        thr = snr_k * floor
        sm = uniform_filter1d(amp, size=max(1, int(round(5.0 / df))))
        start = int(np.argmax(sm * (freqs >= 1.0)))
        below = np.where(sm[start:] < thr)[0]
        out["noise_thr"] = thr
        if len(below) > 0:
            out["f_snr_hz"] = float(freqs[start + below[0]])
        else:
            out["f_snr_hz"] = fs / 2.0
            notes.append("spectrum ไม่ตกต่ำกว่า noise floor ก่อน Nyquist: f_snr = Nyquist (อาจมี aliasing/noise สูง)")

    return out


def plate_response_check(f_hz: float, fn_hz: float, low_ratio: float = 10.0, flag_ratio: float = 4.5) -> Dict[str, object]:
    """ระบบ 2nd-order undamped: amplitude error = 1/(1-r^2) - 1 , r = f/fn"""
    r = f_hz / fn_hz
    if r >= 1.0:
        return {"ratio_fn_over_f": 1.0 / r, "amp_error_pct": float("inf"), "status": "Flag"}
    err = (1.0 / (1.0 - r * r) - 1.0) * 100.0
    ratio = fn_hz / f_hz
    status = "Low" if ratio >= low_ratio else ("Monitor" if ratio >= flag_ratio else "Flag")
    return {"ratio_fn_over_f": ratio, "amp_error_pct": err, "status": status}


def analyze_impact(channels: Dict[str, np.ndarray], fs: float, t_idx: int, l_idx: int,
                   fn_hz: Optional[float] = None, **kw) -> Dict[str, Dict[str, object]]:
    """channels เช่น {"Left": f_left_zeroed, "Right": f_right_zeroed, "Total": f_total_zeroed}"""
    res: Dict[str, Dict[str, object]] = {}
    for name, x in channels.items():
        r = landing_impact_frequency(x, fs, t_idx, l_idx, **kw)
        cand = [r.get(k) for k in ("f_c99_hz", "f_rise_hz", "f_dur_hz") if r.get(k)]
        r["f_worst_hz"] = max(cand) if cand else None
        if fn_hz and r["f_worst_hz"]:
            r["check"] = plate_response_check(r["f_worst_hz"], fn_hz)
        res[name] = r
    return res


def summarize_impact(res: Dict[str, Dict[str, object]], fn_hz: Optional[float] = None):
    """แปลงผล analyze_impact เป็นแถวข้อความสำหรับตารางใน UI / PDF"""
    def fmt(v, nd=1, unit=""):
        return "-" if v is None else f"{float(v):.{nd}f}{unit}"
    rows = []
    for name, r in res.items():
        chk = r.get("check")
        rows.append({
            "Channel": name,
            "Peak (N)": fmt(r.get("peak_force_n"), 0),
            "Rise 10-90% (ms)": fmt(r.get("t_rise_ms")),
            "f_rise (Hz)": fmt(r.get("f_rise_hz")),
            "f_dur (Hz)": fmt(r.get("f_dur_hz")),
            "f_99% energy (Hz)": fmt(r.get("f_c99_hz")),
            "f_used (Hz)": fmt(r.get("f_worst_hz")),
            "fn / f": fmt(chk["ratio_fn_over_f"]) if chk else "-",
            "Amp. error (%)": (fmt(chk["amp_error_pct"], 2) if chk and chk["amp_error_pct"] != float("inf") else (">100" if chk else "-")),
            "Status": chk["status"] if chk else "fn not set",
        })
    return rows
