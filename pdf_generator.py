import io
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

def load_image_from_github(url):
    try:
        req = urllib.request.urlopen(url, timeout=3)
        img = PILImage.open(io.BytesIO(req.read())).convert("RGBA")
        arr = np.array(img)
        white_pixels = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
        arr[white_pixels, 3] = 0
        return PILImage.fromarray(arr, mode="RGBA")
    except Exception:
        return None

def create_combined_charts_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, threshold_alert=15.0, crop_x_min=None, crop_x_max=None):
    # สร้างรูปที่มี 2 Subplots (กราฟ Force-Time ด้านบน และกราฟ Deficit % ด้านล่าง)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 4.5), dpi=200, gridspec_kw={'height_ratios': [1.3, 1.0]})
    
    # --- Subplot 1: Force-Time Analysis ---
    ax1.plot(t, sl, color='#818cf8', linewidth=0.9, label='Left Limb')
    ax1.plot(t, sr, color='#f87171', linewidth=0.9, label='Right Limb')
    ax1.plot(t, sf, color='#4d2994', linewidth=1.5, label='Total Force')
    
    ax1.axvspan(t_start, t_braking, color='#eab308', alpha=0.15)
    ax1.axvspan(t_braking, t_split, color='#ef4444', alpha=0.15)
    ax1.axvspan(t_split, t_takeoff, color='#22c55e', alpha=0.15)
    
    ax1.axvline(x=t_start, color='#ca8a04', linestyle='--', linewidth=1.1)
    ax1.axvline(x=t_braking, color='#ef4444', linestyle='--', linewidth=1.1)
    ax1.axvline(x=t_split, color='#22c55e', linestyle='--', linewidth=1.1)
    ax1.axvline(x=t_takeoff, color='#dc2626', linestyle='--', linewidth=1.1)
    
    max_y = float(np.max(sf)) * 1.15 if len(sf) > 0 else 3000.0
    ax1.text((t_start + t_braking) / 2.0, max_y * 0.95, 'Unweighting', color='#ca8a04', fontsize=7.5, fontweight='bold', ha='center')
    ax1.text((t_braking + t_split) / 2.0, max_y * 0.88, 'Braking', color='#ef4444', fontsize=7.5, fontweight='bold', ha='center')
    ax1.text((t_split + t_takeoff) / 2.0, max_y * 0.95, 'Propulsive', color='#22c55e', fontsize=7.5, fontweight='bold', ha='center')
    
    t_min = float(t[0]) if len(t) > 0 else 0.0
    t_max = float(t[-1]) if len(t) > 0 else 10.0
    x_start = crop_x_min if crop_x_min is not None else max(t_min, t_start - 1.0)
    x_end = crop_x_max if crop_x_max is not None else min(t_max, t_takeoff + 1.5)
    
    ax1.set_xlim(x_start, x_end)
    ax1.set_ylim(0, max_y)
    ax1.set_ylabel('Force (N)', fontsize=8.5)
    ax1.set_title('FORCE-TIME ANALYSIS & ASYMMETRY PROFILE', fontsize=10, fontweight='bold', color='#1e1b4b', pad=6)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=7.5)

    # Load and Plot Pictograms with Correct Aspect Ratio (ไม่เตี้ยลง)
    mid_unweight = (t_start + t_braking) / 2.0
    mid_brake = (t_braking + t_split) / 2.0
    mid_prop = (t_split + t_takeoff) / 2.0
    mid_flight = t_takeoff + 0.25
    mid_landing = t_takeoff + 0.6

    github_base = "https://raw.githubusercontent.com/RatTongiam/Force-Analysis/main"
    pic_configs = [
        {"file": "Standing.png", "x": max(x_start + 0.1, t_start - 0.2)},
        {"file": "UP.png", "x": mid_unweight},
        {"file": "BP.png", "x": mid_brake},
        {"file": "PP.png", "x": mid_prop},
        {"file": "FP.png", "x": mid_flight},
        {"file": "LP.png", "x": mid_landing},
    ]

    for pic in pic_configs:
        img = load_image_from_github(f"{github_base}/{pic['file']}")
        if img is not None:
            orig_w, orig_h = img.size
            # ล็อคสัดส่วนความกว้างต่อความสูงตามภาพต้นฉบับจริง (ไม่ให้ถูกบีบแบน)
            im_w = (x_end - x_start) * 0.075
            im_h = im_w * (orig_h / orig_w) * (max_y / (x_end - x_start)) * 0.65
            im_x = pic["x"] - im_w / 2.0
            im_y = max_y * 0.55
            ax1.imshow(img, extent=[im_x, im_x + im_w, im_y, im_y + im_h], aspect='auto', zorder=5)

    # --- Subplot 2: L/R Asymmetry % Profile ---
    max_sl_sr = np.maximum(sl, sr)
    deficits = np.where((sf >= 50.0) & (max_sl_sr > 0), ((sl - sr) / np.maximum(max_sl_sr, 1e-6)) * 100, 0.0)

    ax2.axvspan(t_start, t_braking, color='#eab308', alpha=0.15)
    ax2.axvspan(t_braking, t_split, color='#ef4444', alpha=0.15)
    ax2.axvspan(t_split, t_takeoff, color='#22c55e', alpha=0.15)

    ax2.axvline(x=t_start, color='#ca8a04', linestyle='--', linewidth=1.1)
    ax2.axvline(x=t_braking, color='#ef4444', linestyle='--', linewidth=1.1)
    ax2.axvline(x=t_split, color='#22c55e', linestyle='--', linewidth=1.1)
    ax2.axvline(x=t_takeoff, color='#dc2626', linestyle='--', linewidth=1.1)

    ax2.axhline(y=0, color='#6b7280', linewidth=1.0)
    ax2.plot(t, deficits, color='#4d2994', linewidth=1.3, label='Asymmetry %')
    ax2.axhspan(-threshold_alert, threshold_alert, color='#22c55e', alpha=0.15)

    ax2.set_xlim(x_start, x_end)
    ax2.set_ylim(-55, 55)
    ax2.set_xlabel('Time (s)', fontsize=8.5)
    ax2.set_ylabel('Deficit %', fontsize=8.5)
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=200, transparent=True)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

def generate_pdf_report(report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, threshold_alert=15.0, crop_x_min=None, crop_x_max=None):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=20,
        bottomMargin=20
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=16,
        textColor=colors.HexColor('#1e1b4b')
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#64748b')
    )
    header_style = ParagraphStyle(
        'HeaderStyle',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9,
        textColor=colors.whitesmoke
    )
    cell_style = ParagraphStyle(
        'CellStyle',
        fontName='Helvetica',
        fontSize=7,
        leading=8.5,
        textColor=colors.HexColor('#1e293b')
    )
    cat_style = ParagraphStyle(
        'CatStyle',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#4d2994')
    )
    
    story = []
    
    story.append(Paragraph("Free JumpAnz Team - Biomechanics Analysis", title_style))
    story.append(Paragraph("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight", subtitle_style))
    story.append(Spacer(1, 4))
    
    # ใส่กราฟคู่ (Force-Time + Asymmetry Profile)
    chart_buf = create_combined_charts_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, threshold_alert, crop_x_min, crop_x_max)
    story.append(RLImage(chart_buf, width=535, height=230))
    story.append(Spacer(1, 4))
    
    table_data = [[
        Paragraph("Biomechanical Metric", header_style),
        Paragraph("Left", header_style),
        Paragraph("Right", header_style),
        Paragraph("TOTAL", header_style),
        Paragraph("Deficit %", header_style)
    ]]
    
    for phase_name, metrics in report.items():
        table_data.append([
            Paragraph(f"<b>=== {phase_name.upper()} ===</b>", cat_style),
            "", "", "", ""
        ])
        for metric_name, vals in metrics.items():
            table_data.append([
                Paragraph(metric_name, cell_style),
                Paragraph(str(vals["Left"]), cell_style),
                Paragraph(str(vals["Right"]), cell_style),
                Paragraph(str(vals["Total"]), cell_style),
                Paragraph(str(vals["Deficit"]), cell_style)
            ])
            
    t_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4d2994')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ])
    
    table = Table(table_data, colWidths=[215, 80, 80, 80, 80])
    table.setStyle(t_style)
    story.append(table)
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
