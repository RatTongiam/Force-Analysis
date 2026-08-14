import io
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image as PILImage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

def load_image_from_github(url):
    try:
        req = urllib.request.urlopen(url, timeout=3)
        img = PILImage.open(io.BytesIO(req.read())).convert("RGBA")
        arr = np.array(img)
        white_pixels = (arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235)
        arr[white_pixels, 3] = 0
        return PILImage.fromarray(arr, mode="RGBA")
    except Exception:
        return None

def create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, crop_x_min=None, crop_x_max=None):
    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=300)
    
    ax.plot(t, sl, color='#818cf8', linewidth=1.3, label='Left Limb')
    ax.plot(t, sr, color='#f87171', linewidth=1.3, label='Right Limb')
    ax.plot(t, sf, color='#4d2994', linewidth=2.0, label='Total Force')
    
    # ไฮไลต์ Phase ต่างๆ รวมถึง Flight Phase
    ax.axvspan(t_start, t_braking, color='#eab308', alpha=0.15)
    ax.axvspan(t_braking, t_split, color='#ef4444', alpha=0.15)
    ax.axvspan(t_split, t_takeoff, color='#22c55e', alpha=0.15)
    ax.axvspan(t_takeoff, t_landing, color='#94a3b8', alpha=0.15)
    
    # เส้นแบ่ง Phase พร้อมเส้น Landing สีฟ้า (#0284c7)
    ax.axvline(x=t_start, color='#ca8a04', linestyle='--', linewidth=1.3)
    ax.axvline(x=t_braking, color='#ef4444', linestyle='--', linewidth=1.3)
    ax.axvline(x=t_split, color='#22c55e', linestyle='--', linewidth=1.3)
    ax.axvline(x=t_takeoff, color='#dc2626', linestyle='--', linewidth=1.3)
    ax.axvline(x=t_landing, color='#0284c7', linestyle='--', linewidth=1.3)
    
    max_y = float(np.max(sf)) * 1.30 if len(sf) > 0 else 3000.0
    
    # กำกับข้อความชื่อ Phase
    ax.text((t_start + t_braking) / 2.0, max_y * 0.94, 'Unweighting', color='#ca8a04', fontsize=9.0, fontweight='bold', ha='center')
    ax.text((t_braking + t_split) / 2.0, max_y * 0.86, 'Braking', color='#ef4444', fontsize=9.0, fontweight='bold', ha='center')
    ax.text((t_split + t_takeoff) / 2.0, max_y * 0.94, 'Propulsive', color='#22c55e', fontsize=9.0, fontweight='bold', ha='center')
    ax.text((t_takeoff + t_landing) / 2.0, max_y * 0.86, 'Flight', color='#64748b', fontsize=9.0, fontweight='bold', ha='center')
    
    t_min = float(t[0]) if len(t) > 0 else 0.0
    t_max = float(t[-1]) if len(t) > 0 else 10.0
    x_start = crop_x_min if crop_x_min is not None else max(t_min, t_start - 0.5)
    x_end = crop_x_max if crop_x_max is not None else min(t_max, t_landing + 0.5)
    
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(0, max_y)
    ax.set_ylabel('Force (N)', fontsize=10.5)
    ax.set_title('FORCE-TIME ANALYSIS & SUB-PHASES', fontsize=12, fontweight='bold', color='#1e1b4b', pad=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9.0)

    mid_unweight = (t_start + t_braking) / 2.0
    mid_brake = (t_braking + t_split) / 2.0
    mid_prop = (t_split + t_takeoff) / 2.0
    mid_flight = (t_takeoff + t_landing) / 2.0
    mid_landing = min(t_max, t_landing + 0.15)

    github_base = "https://raw.githubusercontent.com/RatTongiam/Force-Analysis/main"
    pic_configs = [
        {"file": "Standing.png", "x": max(x_start + 0.05, t_start - 0.15)},
        {"file": "UP.png", "x": mid_unweight},
        {"file": "BP.png", "x": mid_brake},
        {"file": "PP.png", "x": mid_prop},
        {"file": "FP.png", "x": mid_flight},
        {"file": "LP.png", "x": mid_landing},
    ]

    for pic in pic_configs:
        img = load_image_from_github(github_base + "/" + pic["file"])
        if img is not None:
            imagebox = OffsetImage(img, zoom=0.15)
            ab = AnnotationBbox(imagebox, (pic["x"], max_y * 0.52), frameon=False, box_alignment=(0.5, 0.0))
            ax.add_artist(ab)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

def create_deficit_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert=15.0, crop_x_min=None, crop_x_max=None):
    fig, ax = plt.subplots(figsize=(9.0, 2.3), dpi=300)
    
    max_sl_sr = np.maximum(sl, sr)
    deficits = np.where((sf >= 50.0) & (max_sl_sr > 0), ((sl - sr) / np.maximum(max_sl_sr, 1e-6)) * 100, 0.0)

    ax.axvspan(t_start, t_braking, color='#eab308', alpha=0.15)
    ax.axvspan(t_braking, t_split, color='#ef4444', alpha=0.15)
    ax.axvspan(t_split, t_takeoff, color='#22c55e', alpha=0.15)
    ax.axvspan(t_takeoff, t_landing, color='#94a3b8', alpha=0.15)

    ax.axvline(x=t_start, color='#ca8a04', linestyle='--', linewidth=1.3)
    ax.axvline(x=t_braking, color='#ef4444', linestyle='--', linewidth=1.3)
    ax.axvline(x=t_split, color='#22c55e', linestyle='--', linewidth=1.3)
    ax.axvline(x=t_takeoff, color='#dc2626', linestyle='--', linewidth=1.3)
    ax.axvline(x=t_landing, color='#0284c7', linestyle='--', linewidth=1.3)

    ax.axhline(y=0, color='#6b7280', linewidth=1.1)
    ax.plot(t, deficits, color='#4d2994', linewidth=1.6, label='Asymmetry %')
    ax.axhspan(-threshold_alert, threshold_alert, color='#22c55e', alpha=0.15)

    t_min = float(t[0]) if len(t) > 0 else 0.0
    t_max = float(t[-1]) if len(t) > 0 else 10.0
    x_start = crop_x_min if crop_x_min is not None else max(t_min, t_start - 0.5)
    x_end = crop_x_max if crop_x_max is not None else min(t_max, t_landing + 0.5)

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(-55, 55)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Deficit %', fontsize=10)
    ax.set_title('L/R ASYMMETRY % PROFILE', fontsize=11, fontweight='bold', color='#1e1b4b', pad=6)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

def generate_pdf_report(report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert=15.0, crop_x_min=None, crop_x_max=None):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        leftMargin=25,
        rightMargin=25,
        topMargin=15,
        bottomMargin=15
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
        fontSize=8,
        leading=10,
        textColor=colors.whitesmoke
    )
    cell_style = ParagraphStyle(
        'CellStyle',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1e293b')
    )
    cat_style = ParagraphStyle(
        'CatStyle',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#4d2994')
    )
    
    story = []
    
    story.append(Paragraph("Free JumpAnz Team - Biomechanics Analysis", title_style))
    story.append(Paragraph("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight", subtitle_style))
    story.append(Spacer(1, 4))
    
    # 1. กราฟ Force-Time
    force_chart_buf = create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, crop_x_min, crop_x_max)
    story.append(RLImage(force_chart_buf, width=545, height=165))
    story.append(Spacer(1, 4))
    
    # 2. ตารางค่าชีวกลศาสตร์
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
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
        ('TOPPADDING', (0, 0), (-1, -1), 1.6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ])
    
    table = Table(table_data, colWidths=[315, 55, 55, 55, 65])
    table.setStyle(t_style)
    story.append(table)
    story.append(Spacer(1, 4))

    # 3. กราฟ L/R Asymmetry Deficit %
    deficit_chart_buf = create_deficit_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max)
    story.append(RLImage(deficit_chart_buf, width=545, height=105))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
