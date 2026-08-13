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
        
        # แปลงพื้นหลังสีขาวให้โปร่งใส (Transparent Background)
        arr = np.array(img)
        # ตรวจสอบพิกเซลที่เป็นสีขาว (หรือใกล้เคียงขาว > 240) แล้วปรับ Alpha เป็น 0
        white_pixels = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
        arr[white_pixels, 3] = 0
        
        return PILImage.fromarray(arr, mode="RGBA")
    except Exception:
        return None

def create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, crop_x_min=None, crop_x_max=None):
    fig, ax = plt.subplots(figsize=(8.0, 2.8), dpi=200)
    
    # Plot Lines
    ax.plot(t, sl, color='#818cf8', linewidth=1.0, label='Left Limb')
    ax.plot(t, sr, color='#f87171', linewidth=1.0, label='Right Limb')
    ax.plot(t, sf, color='#4d2994', linewidth=1.8, label='Total Force')
    
    # Phase Rectangles
    ax.axvspan(t_start, t_braking, color='#eab308', alpha=0.15)
    ax.axvspan(t_braking, t_split, color='#ef4444', alpha=0.15)
    ax.axvspan(t_split, t_takeoff, color='#22c55e', alpha=0.15)
    
    # Vertical Phase Boundary Lines
    ax.axvline(x=t_start, color='#ca8a04', linestyle='--', linewidth=1.2)
    ax.axvline(x=t_braking, color='#ef4444', linestyle='--', linewidth=1.2)
    ax.axvline(x=t_split, color='#22c55e', linestyle='--', linewidth=1.2)
    ax.axvline(x=t_takeoff, color='#dc2626', linestyle='--', linewidth=1.2)
    
    # Dynamic Phase Labels
    max_y = float(np.max(sf)) * 1.15 if len(sf) > 0 else 3000.0
    ax.text((t_start + t_braking) / 2.0, max_y * 0.95, 'Unweighting', color='#ca8a04', fontsize=8, fontweight='bold', ha='center')
    ax.text((t_braking + t_split) / 2.0, max_y * 0.88, 'Braking', color='#ef4444', fontsize=8, fontweight='bold', ha='center')
    ax.text((t_split + t_takeoff) / 2.0, max_y * 0.95, 'Propulsive', color='#22c55e', fontsize=8, fontweight='bold', ha='center')
    
    # Crop X-axis Range if provided
    t_min = float(t[0]) if len(t) > 0 else 0.0
    t_max = float(t[-1]) if len(t) > 0 else 10.0
    x_start = crop_x_min if crop_x_min is not None else max(t_min, t_start - 1.0)
    x_end = crop_x_max if crop_x_max is not None else min(t_max, t_takeoff + 1.5)
    
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Force (N)', fontsize=9)
    ax.set_title('FORCE-TIME ANALYSIS & SUB-PHASES', fontsize=11, fontweight='bold', color='#1e1b4b', pad=8)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)

    # Load and Plot Pictograms with Transparent Background
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
            im_w = (x_end - x_start) * 0.08
            im_h = max_y * 0.20
            im_x = pic["x"] - im_w / 2.0
            im_y = max_y * 0.58
            ax.imshow(img, extent=[im_x, im_x + im_w, im_y, im_y + im_h], aspect='auto', zorder=5)

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
        topMargin=25,
        bottomMargin=25
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#1e1b4b')
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
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
    story.append(Spacer(1, 6))
    
    chart_buf = create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, crop_x_min, crop_x_max)
    story.append(RLImage(chart_buf, width=535, height=185))
    story.append(Spacer(1, 6))
    
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
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ])
    
    table = Table(table_data, colWidths=[215, 80, 80, 80, 80])
    table.setStyle(t_style)
    story.append(table)
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
