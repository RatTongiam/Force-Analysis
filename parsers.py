import pandas as pd
import numpy as np
import json
import io
import re

def parse_tsv(uploaded_file):
    """
    อ่านไฟล์ TSV จาก QTM Dual Force Plate
    รองรับทั้งไฟล์แบบ 9 คอลัมน์ (ไม่มี Frame/Time) และ 11 คอลัมน์ (มี Frame/Time)
    """
    uploaded_file.seek(0)
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
                # กรองไม่เอาบรรทัดข้อมูลที่มีค่า NaN ติดมา
                if not any(np.isnan(v) for v in vals):
                    data.append(vals)
            except ValueError:
                pass

    dt = 1.0 / freq
    arr = np.array(data)
    num_cols = arr.shape[1] if len(arr) > 0 else 0

    # Auto-detect ตำแหน่งคอลัมน์ Fz ตามประเภทโครงสร้างไฟล์ TSV
    if num_cols >= 11:
        fz_idx = 4  # โครงสร้างแบบมี Frame, Time, Fx, Fy, Fz...
    elif num_cols == 9:
        fz_idx = 2  # โครงสร้างแบบไม่มี Frame/Time (เริ่มที่ Fx, Fy, Fz...)
    else:
        fz_idx = 2  # Fallback Default

    return dt, arr, fz_idx

def parse_vald_forcedecks_exact(uploaded_file):
    """
    อ่านไฟล์ CSV/TSV จาก VALD ForceDecks
    """
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
        
        t_raw = df['Time'].values.astype(float)
        t = t_raw - t_raw[0] if len(t_raw) > 0 else np.array([])
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
        
        t_raw = df.iloc[:, 0].values.astype(float) if df.shape[1] > 0 else np.arange(len(df)) * 0.001
        t = t_raw - t_raw[0] if len(t_raw) > 0 else np.arange(len(df)) * 0.001
        dt = t[1] - t[0] if len(t) > 1 and (t[1] - t[0]) > 0 else 0.001
        
        f_left = np.abs(df.iloc[:, 1].values.astype(float)) if df.shape[1] > 1 else np.zeros(len(df))
        f_right = np.abs(df.iloc[:, 4].values.astype(float)) if df.shape[1] > 4 else f_left
        f_total = f_left + f_right
        return dt, t, f_left, f_right, f_total

def parse_qtm_json(uploaded_file):
    """
    อ่านไฟล์ Single JSON จาก Qualisys QTM
    การันตีคำนวณแกนเวลา Relative Time เริ่มต้นที่ 0.0s เสมอ
    """
    uploaded_file.seek(0)
    content = json.load(uploaded_file)
    root = content[0] if isinstance(content, list) else content
    plates = root.get("ForcePlates", [])
    
    if len(plates) == 0:
        return []
        
    plate_data = []
    for idx, plate in enumerate(plates):
        parts = plate.get("Parts", [])
        if len(parts) > 0 and len(parts[0].get("Values", [])) > 0:
            vals = np.array(parts[0]["Values"])
            if vals.shape[1] >= 9:
                col_means = [np.mean(np.abs(vals[:, c])) for c in [2, 5, 8] if c < vals.shape[1]]
                best_col = [2, 5, 8][np.argmax(col_means)] if col_means else 2
                fz = np.abs(vals[:, best_col])
                
                range_info = parts[0].get("Range", {})
                n_start = range_info.get("Start", 1)
                n_end = range_info.get("End", len(fz))
                total_frames = max(1, n_end - n_start + 1)
                
                timestamps = vals[:, 0] if vals.shape[1] > 0 else np.arange(total_frames)
                if len(timestamps) > 1 and (timestamps[1] - timestamps[0]) > 0:
                    dt = float(timestamps[1] - timestamps[0])
                else:
                    cam_freq = root.get("Timebase", {}).get("Frequency", 120.0)
                    duration = total_frames / 2000.0 if total_frames > 5000 else total_frames / cam_freq
                    dt = duration / total_frames if total_frames > 0 else 1.0 / 2000.0
                
                plate_data.append({
                    "id": idx,
                    "name": plate.get("Name", f"Force-plate {idx+1}"),
                    "fz": fz,
                    "dt": dt
                })
                
    return plate_data

def parse_single_csv_cforce(uploaded_file):
    """
    อ่านไฟล์ Single CSV จาก C-Force Performance
    """
    uploaded_file.seek(0)
    lines = uploaded_file.getvalue().decode('utf-8', errors='ignore').splitlines()
    
    data = []
    for line in lines[1:]:
        line_str = line.strip()
        if not line_str:
            continue
        parts = line_str.split(',')
        if len(parts) >= 4:
            try:
                t_val = float(parts[0].strip())
                
                def clean_val(val_str):
                    val_str = val_str.strip()
                    match = re.findall(r'-?\d+\.?\d*', val_str)
                    if len(match) == 1:
                        return float(match[0])
                    elif len(match) > 1:
                        return float("".join(match[1:])) if len(val_str.split()) > 1 else float(match[0])
                    return 0.0

                f_val = clean_val(parts[1])
                l_val = clean_val(parts[2])
                r_val = clean_val(parts[3])
                
                data.append([t_val, f_val, l_val, r_val])
            except Exception:
                pass
                
    if len(data) == 0:
        return 0.001, np.array([]), np.array([]), np.array([]), np.array([])
        
    df = pd.DataFrame(data, columns=['time', 'force', 'left', 'right'])
    t_raw = df['time'].values.astype(float)
    t = t_raw - t_raw[0] if len(t_raw) > 0 else np.array([])
    dt = t[1] - t[0] if len(t) > 1 and (t[1] - t[0]) > 0 else 0.001
    
    f_total = np.abs(df['force'].values.astype(float))
    f_left = np.abs(df['left'].values.astype(float))
    f_right = np.abs(df['right'].values.astype(float))
    
    return dt, t, f_left, f_right, f_total

def parse_musclelab_csv(uploaded_file):
    """
    อ่านไฟล์ CSV จาก MuscleLab Force Plate
    """
    uploaded_file.seek(0)
    raw_text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
    
    df = pd.read_csv(io.StringIO(raw_text), sep=';')
    
    col_left = [c for c in df.columns if 'Left Newton' in c or ('Left' in c and 'Newton' in c)]
    col_right = [c for c in df.columns if 'Right Newton' in c or ('Right' in c and 'Newton' in c)]
    
    if not col_left or not col_right:
        raise ValueError("ไม่พบคอลัมน์ข้อมูลแรง (Newton) ของ MuscleLab Force Plate")
        
    f_left_raw = pd.to_numeric(df[col_left[0]], errors='coerce').values
    f_right_raw = pd.to_numeric(df[col_right[0]], errors='coerce').values
    
    dt = 0.001  # 1000 Hz
    
    valid_mask = ~np.isnan(f_left_raw) & ~np.isnan(f_right_raw) & (f_left_raw > 0) & (f_right_raw > 0)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        last_valid_idx = valid_indices[-1]
        f_left = np.abs(f_left_raw[:last_valid_idx + 1])
        f_right = np.abs(f_right_raw[:last_valid_idx + 1])
    else:
        f_left = np.abs(np.nan_to_num(f_left_raw))
        f_right = np.abs(np.nan_to_num(f_right_raw))
        
    f_total = f_left + f_right
    t = np.arange(len(f_total)) * dt
    
    return dt, t, f_left, f_right, f_total
