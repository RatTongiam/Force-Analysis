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
    dt = t[1] - t[0] if len(t) > 1 and (t[1] - t[0]) > 0 else 0.001
    
    f_total = np.abs(pd.to_numeric(df[total_col], errors='coerce').fillna(0).values) if total_col else np.zeros(len(df))
    f_left = np.abs(pd.to_numeric(df[left_col], errors='coerce').fillna(0).values) if left_col else f_total * 0.5
    f_right = np.abs(pd.to_numeric(df[right_col], errors='coerce').fillna(0).values) if right_col else f_total * 0.5
    
    return dt, t, f_left, f_right, f_total
