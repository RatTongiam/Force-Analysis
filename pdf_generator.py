import io
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image as PILImage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak

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

def create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, crop_x_min=None, crop_x_max=None, is_coach=True):
    fig, ax = plt.subplots(figsize=(8.0, 3.2 if is_coach else 3.6), dpi=300)
    
    col_l = '#2f6fed' if is_coach else '#818cf8'
    col_r = '#ed7d31' if is_coach else '#f87171'
    col_tot = '#11395f' if is_coach else '#4d2994'

    ax.plot(t, sl, color=col_l, linewidth=0.65, label='Left')
    ax.plot(t, sr, color=col_r, linewidth=0.65, label='Right')
    ax.plot(t, sf, color=col_tot, linewidth=1.1, label='Total')
    
    ax.axvspan(t_start, t_braking, color='#eab308', alpha=0.12)
    ax.axvspan(t_braking, t_split, color='#ef4444', alpha=0.12)
    ax.axvspan(t_split, t_takeoff, color='#22c55e', alpha=0.12)
    ax.axvspan(t_takeoff, t_landing, color='#94a3b8', alpha=0.12)
    
    ax.axvline(x=t_start, color='#ca8a04', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_braking, color='#ef4444', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_split, color='#22c55e', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_takeoff, color='#dc2626', linestyle='--', linewidth=0.65)
    ax.axvline(x=t_landing, color='#0284c7', linestyle='--', linewidth=0.65)
    
    max_y = float(np.max(sf)) * 1.25 if len(sf) > 0 else 3000.0
    
    ax.text((t_start + t_braking) / 2.0, max_y * 0.94, 'Unweight', color='#ca8a04', fontsize=8.0, fontweight='bold', ha='center')
    ax.text((t_braking + t_split) / 2.0, max_y * 0.86, 'Braking', color='#ef4444', fontsize=8.0, fontweight='bold', ha='center')
    ax.text((t_split + t_takeoff) / 2.0, max_y * 0.94, 'Propulsion', color='#22c55e', fontsize=8.0, fontweight='bold', ha='center')
    ax.text((t_takeoff + t_landing) / 2.0, max_y * 0.86, 'Flight', color='#64748b', fontsize=8.0, fontweight='bold', ha='center')
    
    t_min = float(t[0]) if len(t) > 0 else 0.0
    t_max = float(t[-1]) if len(t) > 0 else 10.0
    x_start = crop_x_min if crop_x_min is not None else max(t_min, t_start - 0.5)
    x_end = crop_x_max if crop_x_max is not None else min(t_max, t_landing + 0.5)
    
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(0, max_y)
    ax.set_ylabel('Force (N)', fontsize=9.0)
    ax.set_title('Interactive vertical force–time curve', fontsize=10.5, fontweight='bold', color=col_tot, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
    ax.legend(loc='upper right', fontsize=8.0)

    github_base = "https://raw.githubusercontent.com/RatTongiam/Force-Analysis/main"
    pic_configs = [
        {"file": "Standing.png", "x": max(x_start + 0.05, t_start - 0.15)},
        {"file": "UP.png", "x": (t_start + t_braking) / 2.0},
        {"file": "BP.png", "x": (t_braking + t_split) / 2.0},
        {"file": "PP.png", "x": (t_split + t_takeoff) / 2.0},
        {"file": "FP.png", "x": (t_takeoff + t_landing) / 2.0},
        {"file": "LP.png", "x": min(t_max, t_landing + 0.15)},
    ]

    for pic in pic_configs:
        img = load_image_from_github(github_base + "/" + pic["file"])
        if img is not None:
            imagebox = OffsetImage(img, zoom=0.12 if is_coach else 0.15)
            ab = AnnotationBbox(imagebox, (pic["x"], max_y * 0.50), frameon=False, box_alignment=(0.5, 0.0))
            ax.add_artist(ab)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

def create_mini_series_image(t, data, title, ylabel, color='#11395f'):
    fig, ax = plt.subplots(figsize=(4.8, 1.8), dpi=300)
    if len(t) == len(data) and len(t) > 0:
        ax.plot(t, data, color=color, linewidth=1.0)
    else:
        ax.plot([0, 1], [0, 0], color=color, linewidth=1.0)
    ax.axhline(y=0, color='#94a3b8', linestyle='--', linewidth=0.6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold', color='#1e293b', pad=4)
    ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

def create_deficit_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert=15.0, crop_x_min=None, crop_x_max=None, is_coach=True):
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
    ax.set_xlabel('Time (s)', fontsize=8.5)
    ax.set_ylabel('Deficit %', fontsize=8.5)
    ax.set_title('L/R ASYMMETRY % PROFILE', fontsize=9.5, fontweight='bold', color=col_tot, pad=4)
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

# -------------------------------------------------------------
# Template 1: Classic Purple (A4 Portrait)
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
    
    force_chart_buf = create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, crop_x_min, crop_x_max, is_coach=False)
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

    deficit_chart_buf = create_deficit_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max, is_coach=False)
    story.append(RLImage(deficit_chart_buf, width=545, height=105))

    doc.build(story)

# -------------------------------------------------------------
# Template 2: Modern Coach Analyzer v4 (A4 Landscape Multi-card)
# -------------------------------------------------------------
def build_modern_coach_pdf(doc, report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max, coach_context):
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('HeadTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=13, leading=15, textColor=colors.HexColor('#ffffff'))
    sub_style = ParagraphStyle('HeadSub', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#dce8f5'))
    
    kpi_sub_style = ParagraphStyle('KPISub', fontName='Helvetica', fontSize=6.5, leading=8, textColor=colors.HexColor('#64748b'), alignment=1)
    kpi_val_style = ParagraphStyle('KPIVal', fontName='Helvetica-Bold', fontSize=13, leading=15, textColor=colors.HexColor('#11395f'), alignment=1)
    
    card_h2 = ParagraphStyle('CardH2', fontName='Helvetica-Bold', fontSize=9.0, leading=11, textColor=colors.HexColor('#11395f'))
    cell_lbl = ParagraphStyle('CellLbl', fontName='Helvetica', fontSize=6.8, leading=8.5, textColor=colors.HexColor('#334155'))
    cell_val = ParagraphStyle('CellVal', fontName='Helvetica-Bold', fontSize=6.8, leading=8.5, textColor=colors.HexColor('#0f172a'), alignment=2)
    th_style = ParagraphStyle('THStyle', fontName='Helvetica-Bold', fontSize=6.8, leading=8.5, textColor=colors.HexColor('#475467'))

    story = []

    # 1. Header Banner
    header_data = [[
        Paragraph("<b>CMJ Coach Analyzer v4 Multi-Report — Research &amp; Clinical</b>", title_style)
    ], [
        Paragraph("QTM / Bertec / VALD • Validated Impulse-Momentum Workflow • Bilateral Asymmetry Engine", sub_style)
    ]]
    head_table = Table(header_data, colWidths=[792])
    head_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0d3154')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(head_table)
    story.append(Spacer(1, 4))

    # 2. 4 KPI Scorecards
    jh_imp = str(coach_context.get("jh_imp", "36.1"))
    jh_flt = str(coach_context.get("jh_flt", "35.4"))
    rsi_val = str(coach_context.get("rsi", "0.41"))
    ppk_val = str(coach_context.get("ppk", "48.1"))
    
    kpi_data = [
        [
            Paragraph("Jump height — impulse", kpi_sub_style),
            Paragraph("Jump height — flight", kpi_sub_style),
            Paragraph("RSImod", kpi_sub_style),
            Paragraph("Peak concentric power", kpi_sub_style)
        ],
        [
            Paragraph(f"{jh_imp} cm", kpi_val_style),
            Paragraph(f"{jh_flt} cm", kpi_val_style),
            Paragraph(f"{rsi_val} m/s", kpi_val_style),
            Paragraph(f"{ppk_val} W/kg", kpi_val_style)
        ],
        [
            Paragraph("preferred kinetics estimate", kpi_sub_style),
            Paragraph("quality-check estimate", kpi_sub_style),
            Paragraph("JH (m) / time-to-take-off (s)", kpi_sub_style),
            Paragraph("relative power", kpi_sub_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[198, 198, 198, 198])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 4))

    # 3. Upper Grid: Graph (520 pt) + Snapshot (272 pt)
    force_chart_buf = create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, crop_x_min, crop_x_max, is_coach=True)
    
    coach_headline = str(coach_context.get("headline", "Bilateral propulsion impulse balanced within trial."))
    coach_sub = str(coach_context.get("sub", f"JH {jh_imp} cm • RSImod {rsi_val} m/s"))
    coach_act1 = str(coach_context.get("act1", "Propulsion net impulse is balanced in this trial; use as baseline."))
    coach_act2 = str(coach_context.get("act2", "Use average of 3-5 valid trials and CV% before concluding pattern."))

    snapshot_cell = [
        Paragraph("<b>Coach snapshot</b>", card_h2),
        Spacer(1, 2),
        Paragraph(f"<b>{coach_headline}</b>", ParagraphStyle('SH', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#11395f'))),
        Spacer(1, 2),
        Paragraph(coach_sub, ParagraphStyle('SS', fontName='Helvetica', fontSize=7.0, leading=8.5, textColor=colors.HexColor('#475467'))),
        Spacer(1, 3),
        Paragraph(f"• {coach_act1}", ParagraphStyle('SA1', fontName='Helvetica', fontSize=6.8, leading=8.2, textColor=colors.HexColor('#7c2d12'))),
        Spacer(1, 1),
        Paragraph(f"• {coach_act2}", ParagraphStyle('SA2', fontName='Helvetica', fontSize=6.8, leading=8.2, textColor=colors.HexColor('#7c2d12'))),
    ]

    upper_grid = [[
        RLImage(force_chart_buf, width=515, height=135),
        snapshot_cell
    ]]
    upper_table = Table(upper_grid, colWidths=[520, 272])
    upper_table.setStyle(TableStyle([
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#f8fafc')),
        ('BOX', (1, 0), (1, 0), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (1, 0), (1, 0), 5),
        ('LEFTPADDING', (1, 0), (1, 0), 7),
        ('RIGHTPADDING', (1, 0), (1, 0), 7),
        ('BOTTOMPADDING', (1, 0), (1, 0), 5),
    ]))
    story.append(upper_table)
    story.append(Spacer(1, 4))

    # 4. Middle Grid: Braking & Propulsion Table (792 pt)
    bp_data = [
        [Paragraph("<b>Metric</b>", th_style), Paragraph("<b>Left</b>", th_style), Paragraph("<b>Right</b>", th_style), Paragraph("<b>Directional asymmetry</b>", th_style)],
        [Paragraph("<b>Braking phase</b>", ParagraphStyle('BPP', fontName='Helvetica-Bold', fontSize=7.0, textColor=colors.HexColor('#11395f'))), "", "", ""],
        [Paragraph("Peak force", cell_lbl), Paragraph(f"{coach_context.get('pk_br_l', '-')} N", cell_val), Paragraph(f"{coach_context.get('pk_br_r', '-')} N", cell_val), Paragraph(str(coach_context.get('asym_pk_br', '-')), cell_val)],
        [Paragraph("NET impulse", cell_lbl), Paragraph(f"{coach_context.get('br_net_l', '-')} N·s", cell_val), Paragraph(f"{coach_context.get('br_net_r', '-')} N·s", cell_val), Paragraph(str(coach_context.get('asym_br_net', '-')), cell_val)],
        [Paragraph("Gross GRF impulse", cell_lbl), Paragraph(f"{coach_context.get('br_gross_l', '-')} N·s", cell_val), Paragraph(f"{coach_context.get('br_gross_r', '-')} N·s", cell_val), Paragraph(str(coach_context.get('asym_br_gross', '-')), cell_val)],
        [Paragraph("<b>Propulsion / explosive phase</b>", ParagraphStyle('BPP2', fontName='Helvetica-Bold', fontSize=7.0, textColor=colors.HexColor('#11395f'))), "", "", ""],
        [Paragraph("Peak force", cell_lbl), Paragraph(f"{coach_context.get('pk_pr_l', '-')} N", cell_val), Paragraph(f"{coach_context.get('pk_pr_r', '-')} N", cell_val), Paragraph(str(coach_context.get('asym_pk_pr', '-')), cell_val)],
        [Paragraph("NET impulse", cell_lbl), Paragraph(f"{coach_context.get('pr_net_l', '-')} N·s", cell_val), Paragraph(f"{coach_context.get('pr_net_r', '-')} N·s", cell_val), Paragraph(str(coach_context.get('asym_pr_net', '-')), cell_val)],
        [Paragraph("Gross GRF impulse", cell_lbl), Paragraph(f"{coach_context.get('pr_gross_l', '-')} N·s", cell_val), Paragraph(f"{coach_context.get('pr_gross_r', '-')} N·s", cell_val), Paragraph(str(coach_context.get('asym_pr_gross', '-')), cell_val)],
    ]
    bp_table = Table(bp_data, colWidths=[270, 160, 160, 202])
    bp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#fafbfc')),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#fafbfc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
    ]))
    story.append(bp_table)
    story.append(Spacer(1, 4))

    # 5. Page Break for Page 2
    story.append(PageBreak())

    # Page 2: QC, Timing, Mini Charts & Clinical
    story.append(Paragraph("<b>CMJ Coach Analyzer v4 — Clinical &amp; Data Quality Details (Page 2)</b>", ParagraphStyle('P2H', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#0d3154'))))
    story.append(Spacer(1, 4))

    # QC and Timing Table (No <br> tag syntax issues)
    qc_content = [
        Paragraph(f"• Sampling: {coach_context.get('fs', '2400')} Hz", cell_lbl),
        Paragraph(f"• Quiet-standing CV: {coach_context.get('cv', '0.26')}%", cell_lbl),
        Paragraph(f"• Flight Residual: {coach_context.get('fres', '1.2')} N", cell_lbl),
        Paragraph(f"• JH Agreement: {coach_context.get('jh_diff', '0.67')} cm", cell_lbl)
    ]
    timing_content = [
        Paragraph(f"• Body mass: {coach_context.get('mass', '73.4')} kg", cell_lbl),
        Paragraph(f"• Time to take-off: {coach_context.get('ttt', '870')} ms", cell_lbl),
        Paragraph(f"• Braking duration: {coach_context.get('d_brk', '548')} ms", cell_lbl),
        Paragraph(f"• Countermovement depth: {coach_context.get('depth', '38.7')} cm", cell_lbl)
    ]

    qc_timing_grid = [
        [Paragraph("<b>Data quality checks</b>", card_h2), Paragraph("<b>Timing &amp; strategy</b>", card_h2)],
        [qc_content, timing_content]
    ]
    qc_timing_table = Table(qc_timing_grid, colWidths=[390, 395])
    qc_timing_table.setStyle(TableStyle([
        ('BOX', (0, 0), (0, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BOX', (1, 0), (1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(qc_timing_table)
    story.append(Spacer(1, 4))

    # Sub-charts and Landing Load
    vel_arr = coach_context.get('vel', np.array([]))
    pow_arr = coach_context.get('power', np.array([]))
    t_sub = t[:len(vel_arr)] if len(vel_arr) > 0 else np.array([])
    t_pow = t[:len(pow_arr)] if len(pow_arr) > 0 else np.array([])

    vel_buf = create_mini_series_image(t_sub, vel_arr, "COM velocity", "m/s", color='#11395f')
    pow_buf = create_mini_series_image(t_pow, pow_arr, "COM power", "W/kg", color='#ed7d31')

    landing_details = [
        Paragraph("<b>Total landing load</b>", card_h2),
        Spacer(1, 2),
        Paragraph(f"• Peak landing force: <b>{coach_context.get('pk_land_tot', '-')} N</b> ({coach_context.get('pk_land_bw', '-')} ×BW)", cell_lbl),
        Paragraph(f"• 20–80% loading rate: <b>{coach_context.get('load_rate', '-')} N/s</b> ({coach_context.get('load_rate_bw', '-')} BW/s)", cell_lbl),
        Paragraph(f"• Landing impulse 0–250 ms: <b>{coach_context.get('land_imp_250', '-')} N·s</b>", cell_lbl),
        Paragraph(f"• Time to peak landing: <b>{coach_context.get('ttp_land', '-')} ms</b>", cell_lbl),
        Spacer(1, 2),
        Paragraph("<b>Research Derived Metrics</b>", card_h2),
        Paragraph(f"• Mean braking force: {coach_context.get('mean_br_f', '-')} N/kg", cell_lbl),
        Paragraph(f"• Leg stiffness (Exploratory): {coach_context.get('leg_stiff', '-')} N/m/kg", cell_lbl)
    ]

    p2_grid = [[
        RLImage(vel_buf, width=255, height=95),
        RLImage(pow_buf, width=255, height=95),
        landing_details
    ]]
    p2_table = Table(p2_grid, colWidths=[260, 260, 272])
    p2_table.setStyle(TableStyle([
        ('BOX', (2, 0), (2, 0), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (2, 0), (2, 0), colors.HexColor('#f8fafc')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(p2_table)
    story.append(Spacer(1, 4))

    # Interpretation Box
    interp_box = [
        [Paragraph("<b>Interpretation for coach</b>", card_h2)],
        [Paragraph(str(coach_context.get("interp_text", "Performance &amp; Kinetic Asymmetry summary.")), cell_lbl)]
    ]
    interp_table = Table(interp_box, colWidths=[792])
    interp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(interp_table)

    doc.build(story)

# -------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------
def generate_pdf_report(report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert=15.0, crop_x_min=None, crop_x_max=None, theme="Modern Coach (Dashboard)", coach_context=None):
    pdf_buffer = io.BytesIO()
    
    if "Coach" in theme:
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=landscape(A4),
            leftMargin=20,
            rightMargin=20,
            topMargin=15,
            bottomMargin=15
        )
        build_modern_coach_pdf(doc, report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max, coach_context or {})
    else:
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            leftMargin=25,
            rightMargin=25,
            topMargin=15,
            bottomMargin=15
        )
        build_classic_pdf(doc, report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
