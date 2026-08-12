import pandas as pd
import numpy as np
import json
import io
import re

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
        dt = t[1] - t[0] if len(t) > 1 and (t[1] - t[0]) > 0 else 0.001
        
        f_left = np.abs(df.iloc[:, 1].values.astype(float)) if df.shape[1] > 1 else np.zeros(len(df))
        f_right = np.abs(df.iloc[:, 4].values.astype(float)) if df.shape[1] > 4 else f_left
        f_total = f_left + f_right
        return dt, t, f_left, f_right, f_total

def parse_qtm_json(uploaded_file):
    content = json.load(uploaded_file)
    root = content[0] if isinstance(content, list) else content
    
    freq_camera = root.get("Timebase", {}).get("Frequency", 120.0)
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
        
    if num_frames > 5000 and freq_camera <= 200.0:
        freq_force_plate = 2000.0
    else:
        freq_force_plate = freq_camera

    dt = 1.0 / freq_force_plate
    t = np.arange(num_frames) * dt
    
    f_total_raw = f_left + f_right
    quiet_check = np.mean(f_total_raw[:min(100, num_frames)])
    if quiet_check > 3000.0:
        f_left = f_left / 1000.0
        f_right = f_right / 1000.0
        f_total = f_total_raw / 1000.0
    else:
        f_total = f_total_raw

    return dt, t, f_left, f_right, f_total

def parse_single_csv_cforce(uploaded_file):
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
                        # Handle formatting artifacts like "3 481.97" by taking the primary decimal number part
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
    t = df['time'].values.astype(float)
    dt = t[1] - t[0] if len(t) > 1 and (t[1] - t[0]) > 0 else 0.001
    
    f_total = np.abs(df['force'].values.astype(float))
    f_left = np.abs(df['left'].values.astype(float))
    f_right = np.abs(df['right'].values.astype(float))
    
    return dt, t, f_left, f_right, f_total
