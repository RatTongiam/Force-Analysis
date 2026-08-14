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

def create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, crop_x_min=None, crop_x_max=None, theme="Modern Coach (Dashboard)"):
    is_coach = "Coach" in theme
    fig, ax = plt.subplots(figsize=(9.0, 3.2 if is_coach else 3.6), dpi=300)
    
    col_l = '#2f6fed' if is_coach else '#818cf8'
    col_r = '#ed7d31' if is_coach else '#f87171'
    col_tot = '#11395f' if is_coach else '#4d2994'

    ax.plot(t, sl, color=col_l, linewidth=0.65, label='Left Limb')
    ax.plot(t, sr, color=col_r, linewidth=0.65, label='Right Limb')
    ax.plot(t, sf, color=col_tot, linewidth=1.0, label='Total Force')
    
    ax.axvspan(t_start, t_braking, color='#eab308', alpha=0.15)
    ax.axvspan(t_braking, t_split, color='#ef4444', alpha=0.15)
    ax.axvspan(t_split, t_takeoff, color='#22c55e', alpha=0.15)
    ax.axvspan(t_takeoff, t_landing, color='#94a3b8', alpha=0.15)
    
    ax.axvline(x=t_start, color='#ca8a04', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_braking, color='#ef4444', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_split, color='#22c55e', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_takeoff, color='#dc2626', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_landing, color='#0284c7', linestyle='--', linewidth=0.65)
    
    max_y = float(np.max(sf)) * 1.30 if len(sf) > 0 else 3000.0
    
    ax.text((t_start + t_braking) / 2.0, max_y * 0.94, 'Unweighting', color='#ca8a04', fontsize=8.5, fontweight='bold', ha='center')
    ax.text((t_braking + t_split) / 2.0, max_y * 0.86, 'Braking', color='#ef4444', fontsize=8.5, fontweight='bold', ha='center')
    ax.text((t_split + t_takeoff) / 2.0, max_y * 0.94, 'Propulsive', color='#22c55e', fontsize=8.5, fontweight='bold', ha='center')
    ax.text((t_takeoff + t_landing) / 2.0, max_y * 0.86, 'Flight', color='#64748b', fontsize=8.5, fontweight='bold', ha='center')
    
    t_min = float(t[0]) if len(t) > 0 else 0.0
    t_max = float(t[-1]) if len(t) > 0 else 10.0
    x_start = crop_x_min if crop_x_min is not None else max(t_min, t_start - 0.5)
    x_end = crop_x_max if crop_x_max is not None else min(t_max, t_landing + 0.5)
    
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(0, max_y)
    ax.set_ylabel('Force (N)', fontsize=9.5)
    ax.set_title('FORCE-TIME ANALYSIS & SUB-PHASES', fontsize=11, fontweight='bold', color=col_tot, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
    ax.legend(loc='upper right', fontsize=8.5)

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
            imagebox = OffsetImage(img, zoom=0.13 if is_coach else 0.15)
            ab = AnnotationBbox(imagebox, (pic["x"], max_y * 0.52), frameon=False, box_alignment=(0.5, 0.0))
            ax.add_artist(ab)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

def create_deficit_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert=15.0, crop_x_min=None, crop_x_max=None, theme="Modern Coach (Dashboard)"):
    is_coach = "Coach" in theme
    fig, ax = plt.subplots(figsize=(9.0, 2.0 if is_coach else 2.3), dpi=300)
    
    col_tot = '#11395f' if is_coach else '#4d2994'
    max_sl_sr = np.maximum(sl, sr)
    deficits = np.where((sf >= 50.0) & (max_sl_sr > 0), ((sl - sr) / np.maximum(max_sl_sr, 1e-6)) * 100, 0.0)

    ax.axvspan(t_start, t_braking, color='#eab308', alpha=0.15)
    ax.axvspan(t_braking, t_split, color='#ef4444', alpha=0.15)
    ax.axvspan(t_split, t_takeoff, color='#22c55e', alpha=0.15)
    ax.axvspan(t_takeoff, t_landing, color='#94a3b8', alpha=0.15)

    ax.axvline(x=t_start, color='#ca8a04', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_braking, color='#ef4444', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_split, color='#22c55e', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_takeoff, color='#dc2626', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_landing, color='#0284c7', linestyle='--', linewidth=0.65)

    ax.axhline(y=0, color='#6b7280', linewidth=0.55)
    ax.plot(t, deficits, color=col_tot, linewidth=0.8, label='Asymmetry %')
    ax.axhspan(-threshold_alert, threshold_alert, color='#22c55e', alpha=0.15)

    t_min = float(t[0]) if len(t) > 0 else 0.0
    t_max = float(t[-1]) if len(t) > 0 else 10.0
    x_start = crop_x_min if crop_x_min is not None else max(t_min, t_start - 0.5)
    x_end = crop_x_max if crop_x_max is not None else min(t_max, t_landing + 0.5)

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(-55, 55)
    ax.set_xlabel('Time (s)', fontsize=9.0)
    ax.set_ylabel('Deficit %', fontsize=9.0)
    ax.set_title('L/R ASYMMETRY % PROFILE', fontsize=10, fontweight='bold', color=col_tot, pad=5)
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

# -------------------------------------------------------------
# Template 1: Classic Purple Report (Original Style)
# -------------------------------------------------------------
def build_classic_pdf(doc, report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=14, leading=16, textColor=colors.HexColor('#1e1b4b'))
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor('#64748b'))
    header_style = ParagraphStyle('HeaderStyle', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.whitesmoke)
    cell_style = ParagraphStyle('CellStyle', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#1e293b'))
    cat_style = ParagraphStyle('CatStyle', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor('#4d2994'))
    
    story = []
    story.append(Paragraph("Free JumpAnz Team - Biomechanics Analysis", title_style))
    story.append(Paragraph("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight", subtitle_style))
    story.append(Spacer(1, 4))
    
    force_chart_buf = create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, crop_x_min, crop_x_max, theme="Classic Purple (Original)")
    story.append(RLImage(force_chart_buf, width=545, height=165))
    story.append(Spacer(1, 4))
    
    table_data = [[
        Paragraph("Biomechanical Metric", header_style),
        Paragraph("Left", header_style),
        Paragraph("Right", header_style),
        Paragraph("TOTAL", header_style),
        Paragraph("Deficit %", header_style)
    ]]
    
    for phase_name, metrics in report.items():
        table_data.append([Paragraph(f"<b>=== {phase_name.upper()} ===</b>", cat_style), "", "", "", ""])
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

    deficit_chart_buf = create_deficit_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max, theme="Classic Purple (Original)")
    story.append(RLImage(deficit_chart_buf, width=545, height=105))

    doc.build(story)

# -------------------------------------------------------------
# Template 2: Modern Coach Dashboard Report
# -------------------------------------------------------------
def build_modern_coach_pdf(doc, report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max, coach_context=None):
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('MTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=13, leading=15, textColor=colors.HexColor('#ffffff'))
    sub_style = ParagraphStyle('MSub', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=9, textColor=colors.HexColor('#dce8f5'))
    
    kpi_title_style = ParagraphStyle('KPITitle', fontName='Helvetica-Bold', fontSize=6.5, leading=8, textColor=colors.HexColor('#64748b'), alignment=1)
    kpi_val_style = ParagraphStyle('KPIVal', fontName='Helvetica-Bold', fontSize=13, leading=15, textColor=colors.HexColor('#11395f'), alignment=1)
    
    header_style = ParagraphStyle('MHeader', fontName='Helvetica-Bold', fontSize=7.5, leading=9, textColor=colors.whitesmoke)
    cell_style = ParagraphStyle('MCell', fontName='Helvetica', fontSize=7.0, leading=8.5, textColor=colors.HexColor('#1e293b'))
    cat_style = ParagraphStyle('MCat', fontName='Helvetica-Bold', fontSize=7.5, leading=9, textColor=colors.HexColor('#11395f'))
    
    box_text_style = ParagraphStyle('MBox', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#1e293b'))
    act_text_style = ParagraphStyle('MAct', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#7c2d12'))
    
    story = []

    # 1. Coach Header Banner
    header_data = [[
        Paragraph("<b>CMJ COACH ANALYZER — RESEARCH & CLINICAL REPORT</b>", title_style),
    ], [
        Paragraph("PRIMA MOTION TECHNOLOGY • Bilateral Asymmetry & Phase Dynamics Engine", sub_style)
    ]]
    head_table = Table(header_data, colWidths=[545])
    head_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0d3154')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(head_table)
    story.append(Spacer(1, 4))

    # 2. Extract KPI Values
    jh_val = report.get("5. Flight & Performance Phase", {}).get("Jump Height - Impulse-Momentum (cm)", {}).get("Total", "-")
    if jh_val == "-":
        jh_val = report.get("1. Performance Component (59% Variance)", {}).get("Jump Height - Impulse-Momentum (cm)", {}).get("Total", "-")
    
    rsi_val = report.get("5. Flight & Performance Phase", {}).get("RSI Modified (AU)", {}).get("Total", "-")
    if rsi_val == "-":
        rsi_val = report.get("1. Performance Component (59% Variance)", {}).get("RSI Modified (AU)", {}).get("Total", "-")
    
    peak_p_val = report.get("4. Propulsive (Concentric) Phase", {}).get("Peak Propulsive Power (W)", {}).get("Total", "-")
    if peak_p_val == "-":
        peak_p_val = report.get("1. Performance Component (59% Variance)", {}).get("Peak Propulsive Power (W)", {}).get("Total", "-")
    
    ttt_val = f"{(t_takeoff - t_start)*1000.0:.0f}"

    # 3. Top 4 KPI Cards Table
    kpi_table_data = [
        [
            Paragraph("JUMP HEIGHT (IMPULSE)", kpi_title_style),
            Paragraph("RSI MODIFIED", kpi_title_style),
            Paragraph("PEAK CONCENTRIC POWER", kpi_title_style),
            Paragraph("TIME TO TAKE-OFF", kpi_title_style)
        ],
        [
            Paragraph(f"{jh_val} cm", kpi_val_style),
            Paragraph(f"{rsi_val}", kpi_val_style),
            Paragraph(f"{peak_p_val} W", kpi_val_style),
            Paragraph(f"{ttt_val} ms", kpi_val_style)
        ]
    ]
    kpi_table = Table(kpi_table_data, colWidths=[136, 136, 136, 137])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 3))

    # 4. Coach Snapshot Box
    headline_txt = coach_context.get("headline", "การกระจายแรงซ้าย–ขวาอยู่ในเกณฑ์ปกติ") if coach_context else "Bilateral performance analysis complete."
    action_txt = coach_context.get("action", "ติดตามความสม่ำเสมอของตัวแปรใน Trial ถัดไป") if coach_context else "Standard training monitoring."
    
    coach_box_data = [
        [Paragraph(f"<b>Coach Snapshot:</b> {headline_txt}", box_text_style)],
        [Paragraph(f"<b>Action / Interpretation:</b> {action_txt}", act_text_style)]
    ]
    coach_table = Table(coach_box_data, colWidths=[545])
    coach_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f7ff')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#fff7ed')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(coach_table)
    story.append(Spacer(1, 3))

    # 5. Charts
    force_chart_buf = create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, crop_x_min, crop_x_max, theme="Modern Coach (Dashboard)")
    story.append(RLImage(force_chart_buf, width=545, height=135))
    story.append(Spacer(1, 3))

    # 6. Biomechanical Data Table
    table_data = [[
        Paragraph("Biomechanical Metric", header_style),
        Paragraph("Left", header_style),
        Paragraph("Right", header_style),
        Paragraph("TOTAL", header_style),
        Paragraph("Deficit %", header_style)
    ]]
    
    for phase_name, metrics in report.items():
        table_data.append([Paragraph(f"<b>=== {phase_name.upper()} ===</b>", cat_style), "", "", "", ""])
        for metric_name, vals in metrics.items():
            table_data.append([
                Paragraph(metric_name, cell_style),
                Paragraph(str(vals["Left"]), cell_style),
                Paragraph(str(vals["Right"]), cell_style),
                Paragraph(str(vals["Total"]), cell_style),
                Paragraph(str(vals["Deficit"]), cell_style)
            ])
    
    t_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#11395f')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.1),
        ('TOPPADDING', (0, 0), (-1, -1), 1.1),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ])
    
    table = Table(table_data, colWidths=[315, 55, 55, 55, 65])
    table.setStyle(t_style)
    story.append(table)
    story.append(Spacer(1, 3))

    # 7. Asymmetry Deficit Chart
    deficit_chart_buf = create_deficit_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max, theme="Modern Coach (Dashboard)")
    story.append(RLImage(deficit_chart_buf, width=545, height=85))

    doc.build(story)

# -------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------
def generate_pdf_report(report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert=15.0, crop_x_min=None, crop_x_max=None, theme="Modern Coach (Dashboard)", coach_context=None):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        leftMargin=25,
        rightMargin=25,
        topMargin=15,
        bottomMargin=15
    )
    
    if "Coach" in theme:
        build_modern_coach_pdf(doc, report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max, coach_context)
    else:
        build_classic_pdf(doc, report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
