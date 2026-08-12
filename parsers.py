def parse_qtm_json(uploaded_file):
    content = json.load(uploaded_file)
    root = content[0] if isinstance(content, list) else content
    
    # 1. Camera Frequency vs Analog Force Plate Frequency Resolution
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
        
    # Auto-detect true analog force plate frequency (e.g. 2000Hz or 1000Hz) if frame count is huge
    if num_frames > 5000 and freq_camera <= 200.0:
        freq_force_plate = 2000.0  # Standard QTM analog force plate sampling rate
    else:
        freq_force_plate = freq_camera

    dt = 1.0 / freq_force_plate
    t = np.arange(num_frames) * dt
    
    # 2. Smart Force Unit Detection (mN vs N)
    f_total_raw = f_left + f_right
    quiet_check = np.mean(f_total_raw[:min(100, num_frames)])
    if quiet_check > 3000.0: # Unit is mN
        f_left = f_left / 1000.0
        f_right = f_right / 1000.0
        f_total = f_total_raw / 1000.0
    else: # Unit is already N
        f_total = f_total_raw

    return dt, t, f_left, f_right, f_total
