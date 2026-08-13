import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def get_pictogram_image(filename):
    """โหลดรูปภาพ PNG จาก GitHub Raw และบังคับแปลง Alpha Channel ให้โปร่งแสงจริง 100%"""
    url = f"https://raw.githubusercontent.com/RatTongiam/Force-Analysis/main/{filename}"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            img = PILImage.open(io.BytesIO(resp.content)).convert("RGBA")
            img.thumbnail((40, 60))
            return OffsetImage(img, zoom=0.4)
    except Exception:
        pass
    return None

def generate_pdf_report(report_data, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor('#4d2994'))
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=colors.HexColor('#1a0f30'))
    cell_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#1a0f30'))
    cell_bold = ParagraphStyle('TableCellBold', parent=cell_style, fontName='Helvetica-Bold')
    
    story = [
        Paragraph("BIOMECHANICAL ANALYSIS REPORT (CMJ)", title_style),
        Paragraph("FREE JUMPANZ TEAM — PRIMA MOTION TECHNOLOGY", subtitle_style),
        Spacer(1, 8)
    ]

    # --- PLOT WITH TRANSPARENT PICTOGRAMS & SUB-PHASES ---
    fig_plt, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
    ax.plot(t, sl, label='Left Limb', color='#818cf8', linewidth=0.8, zorder=3)
    ax.plot(t, sr, label='Right Limb', color='#f87171', linewidth=0.8, zorder=3)
    ax.plot(t, sf, label='Total Force', color='#4d2994', linewidth=1.2, zorder=4)

    # Shading Areas
    ax.axvspan(t_start, t_braking, color='#fef08a', alpha=0.35, zorder=1)
    ax.axvspan(t_braking, t_split, color='#fca5a5', alpha=0.35, zorder=1)
    ax.axvspan(t_split, t_takeoff, color='#bbf7d0', alpha=0.35, zorder=1)

    # Vertical Dotted Boundary Lines
    ax.axvline(t_start, color='#ca8a04', linestyle='--', linewidth=0.8, zorder=2)
    ax.axvline(t_braking, color='#ef4444', linestyle='--', linewidth=0.8, zorder=2)
    ax.axvline(t_split, color='#22c55e', linestyle='--', linewidth=0.8, zorder=2)
    ax.axvline(t_takeoff, color='#dc2626', linestyle='--', linewidth=0.8, zorder=2)

    # Dynamic Positioning
    mid_unweight = (t_start + t_braking) / 2.0
    mid_brake = (t_braking + t_split) / 2.0
    mid_prop = (t_split + t_takeoff) / 2.0
    
    dt = t[1] - t[0] if len(t) > 1 else 0.001
    n_samples = len(sf)
    tIdx_curr = min(max(0, int(round((t_takeoff - t[0]) / dt))), n_samples - 1)
    airborne = np.where((np.arange(n_samples) >= tIdx_curr) & (sf < 25.0))[0]
    lIdx_curr = n_samples - 1
    if len(airborne) > 0:
        non_air = np.where((np.arange(n_samples) > airborne[0]) & (sf >= 25.0))[0]
        if len(non_air) > 0:
            lIdx_curr = non_air[0]

    mid_flight = (t_takeoff + t[lIdx_curr]) / 2.0
    mid_landing = min(t[-1], t[lIdx_curr] + 0.2)

    max_y = float(np.max(sf)) * 1.25
    ax.set_ylim(0, max_y)

    # Text Annotations
    ax.text(mid_unweight, max_y * 0.94, "Unweighting", fontsize=7, fontweight='bold', color='#ca8a04', ha='center', zorder=5)
    ax.text(mid_brake, max_y * 0.87, "Braking", fontsize=7, fontweight='bold', color='#ef4444', ha='center', zorder=5)
    ax.text(mid_prop, max_y * 0.94, "Propulsive", fontsize=7, fontweight='bold', color='#22c55e', ha='center', zorder=5)

    # Add Pictogram Images to Matplotlib Axis with Z-Order 10 for True Transparency
    pics_config = [
        ("Standing.png", max(t[0], t_start - 0.15)),
        ("UP.png", mid_unweight),
        ("BP.png", mid_brake),
        ("PP.png", mid_prop),
        ("FP.png", mid_flight),
        ("LP.png", mid_landing)
    ]

    for fname, x_pos in pics_config:
        img_box = get_pictogram_image(fname)
        if img_box:
            ab = AnnotationBbox(
                img_box, 
                (x_pos, max_y * 0.72), 
                frameon=False, 
                pad=0,
                zorder=10
            )
            ax.add_artist(ab)

    ax.set_title("FORCE-TIME ANALYSIS & SUB-PHASES", fontsize=9, fontweight='bold', color='#1a0f30')
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Force (N)", fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=200, transparent=True)
    plt.close(fig_plt)
    img_buffer.seek(0)

    story.append(Image(img_buffer, width=520, height=208))
    story.append(Spacer(1, 6))
    
    table_data = [[
        Paragraph("<b>Biomechanical Metric</b>", cell_bold), 
        Paragraph("<b>Left</b>", cell_bold), 
        Paragraph("<b>Right</b>", cell_bold), 
        Paragraph("<b>TOTAL</b>", cell_bold),
        Paragraph("<b>Deficit %</b>", cell_bold)
    ]]
    
    span_rows = []
    current_row = 1
    
    for phase_name, metrics in report_data.items():
        table_data.append([
            Paragraph(f"<b>{phase_name.upper()}</b>", ParagraphStyle('PhaseHeader', parent=cell_bold, textColor=colors.HexColor('#4d2994'))),
            "", "", "", ""
        ])
        span_rows.append(current_row)
        current_row += 1
        
        for m_name, vals in metrics.items():
            table_data.append([
                Paragraph(m_name, cell_style),
                Paragraph(str(vals.get("Left", "-")), cell_style),
                Paragraph(str(vals.get("Right", "-")), cell_style),
                Paragraph(f"<b>{vals.get('Total', '-')}</b>", cell_bold),
                Paragraph(str(vals.get("Deficit", "-")), cell_style)
            ])
            current_row += 1
            
    t_styles_list = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f6fb')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
    ]
    
    for s_row in span_rows:
        t_styles_list.append(('SPAN', (0, s_row), (4, s_row)))
        t_styles_list.append(('BACKGROUND', (0, s_row), (-1, s_row), colors.HexColor('#f1ebf9')))
        
    doc_table = Table(table_data, colWidths=[190, 75, 75, 90, 90])
    doc_table.setStyle(TableStyle(t_styles_list))
    story.append(doc_table)
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer
