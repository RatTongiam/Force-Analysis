import io
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image as PILImage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -------------------------------------------------------------
# Thai Font Registration Engine (Sarabun Font)
# -------------------------------------------------------------
THAI_FONT_REGISTERED = False
THAI_FONT_NAME = 'Helvetica'
THAI_FONT_BOLD = 'Helvetica-Bold'

def register_thai_font():
    global THAI_FONT_REGISTERED, THAI_FONT_NAME, THAI_FONT_BOLD
    if THAI_FONT_REGISTERED:
        return
    try:
        url_reg = "https://raw.githubusercontent.com/google/fonts/main/ofl/sarabun/Sarabun-Regular.ttf"
        url_bold = "https://raw.githubusercontent.com/google/fonts/main/ofl/sarabun/Sarabun-Bold.ttf"
        
        req_reg = urllib.request.urlopen(url_reg, timeout=3)
        req_bold = urllib.request.urlopen(url_bold, timeout=3)
        
        pdfmetrics.registerFont(TTFont('Sarabun', io.BytesIO(req_reg.read())))
        pdfmetrics.registerFont(TTFont('Sarabun-Bold', io.BytesIO(req_bold.read())))
        
        THAI_FONT_NAME = 'Sarabun'
        THAI_FONT_BOLD = 'Sarabun-Bold'
        THAI_FONT_REGISTERED = True
    except Exception:
        THAI_FONT_NAME = 'Helvetica'
        THAI_FONT_BOLD = 'Helvetica-Bold'

register_thai_font()

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
    fig, ax = plt.subplots(figsize=(8.2, 2.8 if is_coach else 3.4), dpi=300)
    
    col_l = '#2f6fed' if is_coach else '#818cf8'
    col_r = '#ed7d31' if is_coach else '#f87171'
    col_tot = '#11395f' if is_coach else '#4d2994'

    ax.plot(t, sl, color=col_l, linewidth=0.75, label='Left')
    ax.plot(t, sr, color=col_r, linewidth=0.75, label='Right')
    ax.plot(t, sf, color=col_tot, linewidth=1.2, label='Total')
    
    ax.axvspan(t_start, t_braking, color='#eab308', alpha=0.12)
    ax.axvspan(t_braking, t_split, color='#ef4444', alpha=0.12)
    ax.axvspan(t_split, t_takeoff, color='#22c55e', alpha=0.12)
    ax.axvspan(t_takeoff, t_landing, color='#94a3b8', alpha=0.12)
    
    ax.axvline(x=t_start, color='#ca8a04', linestyle='--', linewidth=0.7)
    ax.axvline(x=t_braking, color='#ef4444', linestyle='--', linewidth=0.7)
    ax.axvline(x=t_split, color='#22c55e', linestyle='--', linewidth=0.7)
    ax.axvline(x=t_takeoff, color='#dc2626', linestyle='--', linewidth=0.7)
    ax.axvline(x=t_landing, color='#0284c7', linestyle='--', linewidth=0.7)
    
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
    ax.set_ylabel('Force (N)', fontsize=8.5)
    ax.set_title('FORCE-TIME ANALYSIS & SUB-PHASES' if not is_coach else 'Interactive Vertical Force–Time Curve', fontsize=9.5, fontweight='bold', color=col_tot, pad=4)
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
            imagebox = OffsetImage(img, zoom=0.11 if is_coach else 0.13)
            ab = AnnotationBbox(imagebox, (pic["x"], max_y * 0.50), frameon=False, box_alignment=(0.5, 0.0))
            ax.add_artist(ab)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

def create_mini_series_image(t, data, title, ylabel, color='#11395f'):
    fig, ax = plt.subplots(figsize=(4.0, 1.4), dpi=300)
    if len(t) == len(data) and len(t) > 0:
        ax.plot(t, data, color=color, linewidth=1.0)
    else:
        ax.plot([0, 1], [0, 0], color=color, linewidth=1.0)
    ax.axhline(y=0, color='#94a3b8', linestyle='--', linewidth=0.6)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_title(title, fontsize=8.5, fontweight='bold', color='#1e293b', pad=3)
    ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

def create_deficit_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert=15.0, crop_x_min=None, crop_x_max=None, is_coach=True):
    fig, ax = plt.subplots(figsize=(8.2, 2.0), dpi=300)
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
    ax.set_xlabel('Time (s)', fontsize=8.0)
    ax.set_ylabel('Deficit %', fontsize=8.0)
    ax.set_title('L/R ASYMMETRY % PROFILE', fontsize=9.0, fontweight='bold', color=col_tot, pad=4)
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
    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=13, leading=15, textColor=colors.HexColor('#1e1b4b'))
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#64748b'))
    header_style = ParagraphStyle('HeaderStyle', fontName='Helvetica-Bold', fontSize=7.5, leading=9, textColor=colors.whitesmoke)
    cell_style = ParagraphStyle('CellStyle', fontName='Helvetica', fontSize=6.8, leading=8.5, textColor=colors.HexColor('#1e293b'))
    cat_style = ParagraphStyle('CatStyle', fontName='Helvetica-Bold', fontSize=7.0, leading=8.5, textColor=colors.HexColor('#4d2994'))
    
    story = []
    story.append(Paragraph("Free JumpAnz Team - Biomechanics Analysis", title_style))
    story.append(Paragraph("PRIMA MOTION TECHNOLOGY — Technology that unlocks scientific insight", subtitle_style))
    story.append(Spacer(1, 3))
    
    force_chart_buf = create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, crop_x_min, crop_x_max, is_coach=False)
    story.append(RLImage(force_chart_buf, width=545, height=140))
    story.append(Spacer(1, 3))
    
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
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.0),
        ('TOPPADDING', (0, 0), (-1, -1), 1.0),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ])
    
    table = Table(table_data, colWidths=[315, 55, 55, 55, 65])
    table.setStyle(t_style)
    story.append(table)
    story.append(Spacer(1, 3))

    deficit_chart_buf = create_deficit_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max, is_coach=False)
    story.append(RLImage(deficit_chart_buf, width=545, height=90))

    doc.build(story)

# -------------------------------------------------------------
# Template 2: Modern Coach Analyzer v4 (A4 Portrait 2-Page Fit)
# -------------------------------------------------------------
def build_modern_coach_pdf(doc, report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max, ctx):
    register_thai_font()
    f_reg = THAI_FONT_NAME
    f_bld = THAI_FONT_BOLD
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('HeadTitle', parent=styles['Normal'], fontName=f_bld, fontSize=12.5, leading=14.5, textColor=colors.HexColor('#ffffff'))
    sub_style = ParagraphStyle('HeadSub', parent=styles['Normal'], fontName=f_reg, fontSize=7.5, leading=9.5, textColor=colors.HexColor('#dce8f5'))
    
    kpi_sub_style = ParagraphStyle('KPISub', fontName=f_reg, fontSize=6.5, leading=8.0, textColor=colors.HexColor('#64748b'), alignment=1)
    kpi_val_style = ParagraphStyle('KPIVal', fontName=f_bld, fontSize=13.0, leading=15.0, textColor=colors.HexColor('#11395f'), alignment=1)
    
    card_h2 = ParagraphStyle('CardH2', fontName=f_bld, fontSize=8.5, leading=10.5, textColor=colors.HexColor('#11395f'))
    cell_lbl = ParagraphStyle('CellLbl', fontName=f_reg, fontSize=7.0, leading=8.5, textColor=colors.HexColor('#334155'))
    cell_val = ParagraphStyle('CellVal', fontName=f_bld, fontSize=7.0, leading=8.5, textColor=colors.HexColor('#0f172a'), alignment=2)
    th_style = ParagraphStyle('THStyle', fontName=f_bld, fontSize=7.0, leading=8.5, textColor=colors.HexColor('#475467'))

    story = []

    # =========================================================
    # PAGE 1 (A4 Portrait: 545 pt Printable Width)
    # =========================================================
    
    # 1. Header Banner
    header_data = [
        [Paragraph("<b>CMJ Coach Analyzer v4 — Research &amp; Clinical Report</b>", title_style)],
        [Paragraph("PRIMA MOTION TECHNOLOGY • Bilateral Asymmetry &amp; Phase Dynamics Engine", sub_style)]
    ]
    head_table = Table(header_data, colWidths=[545])
    head_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0d3154')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(head_table)
    story.append(Spacer(1, 3))

    # 2. 4 Top KPI Scorecards
    kpi_data = [
        [Paragraph("Jump height — impulse", kpi_sub_style), Paragraph("Jump height — flight", kpi_sub_style), Paragraph("RSImod", kpi_sub_style), Paragraph("Peak concentric power", kpi_sub_style)],
        [Paragraph(f"{ctx.get('jh_imp', '-')} cm", kpi_val_style), Paragraph(f"{ctx.get('jh_flt', '-')} cm", kpi_val_style), Paragraph(f"{ctx.get('rsi', '-')} m/s", kpi_val_style), Paragraph(f"{ctx.get('ppk', '-')} W/kg", kpi_val_style)],
        [Paragraph("preferred kinetics estimate", kpi_sub_style), Paragraph("quality-check estimate", kpi_sub_style), Paragraph("JH (m) / TTT (s)", kpi_sub_style), Paragraph("relative power", kpi_sub_style)]
    ]
    kpi_table = Table(kpi_data, colWidths=[136, 136, 136, 137])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 3))

    # 3. Coach Snapshot Box (Thai Supported)
    snapshot_box = [
        Paragraph("<b>Coach snapshot</b>", card_h2),
        Spacer(1, 1),
        Paragraph(f"<b>{ctx.get('headline', '-')}</b>", ParagraphStyle('SH', fontName=f_bld, fontSize=7.5, leading=9.5, textColor=colors.HexColor('#11395f'))),
        Paragraph(ctx.get('sub', '-'), ParagraphStyle('SS', fontName=f_reg, fontSize=7.0, leading=8.5, textColor=colors.HexColor('#475467'))),
        Spacer(1, 1),
        Paragraph(f"• {ctx.get('act1', '-')}", ParagraphStyle('SA1', fontName=f_reg, fontSize=7.0, leading=8.5, textColor=colors.HexColor('#7c2d12'))),
        Paragraph(f"• {ctx.get('act2', '-')}", ParagraphStyle('SA2', fontName=f_reg, fontSize=7.0, leading=8.5, textColor=colors.HexColor('#7c2d12'))),
    ]
    snap_table = Table([[snapshot_box]], colWidths=[545])
    snap_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(snap_table)
    story.append(Spacer(1, 3))

    # 4. Interactive Force-Time Graph
    force_chart_buf = create_force_chart_image(t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, crop_x_min, crop_x_max, is_coach=True)
    story.append(RLImage(force_chart_buf, width=545, height=135))
    story.append(Spacer(1, 3))

    # 5. Complete Braking & Propulsion Table (with All RFD Rows)
    bp_data = [
        [Paragraph("<b>Metric</b>", th_style), Paragraph("<b>Left</b>", th_style), Paragraph("<b>Right</b>", th_style), Paragraph("<b>Directional asymmetry</b>", th_style)],
        [Paragraph("<b>Braking phase</b>", ParagraphStyle('BPP', fontName=f_bld, fontSize=6.8, textColor=colors.HexColor('#11395f'))), "", "", ""],
        [Paragraph("Peak force", cell_lbl), Paragraph(f"{ctx.get('pk_br_l', '-')} N", cell_val), Paragraph(f"{ctx.get('pk_br_r', '-')} N", cell_val), Paragraph(str(ctx.get('asym_pk_br', '-')), cell_val)],
        [Paragraph("NET impulse", cell_lbl), Paragraph(f"{ctx.get('br_net_l', '-')} N·s", cell_val), Paragraph(f"{ctx.get('br_net_r', '-')} N·s", cell_val), Paragraph(str(ctx.get('asym_br_net', '-')), cell_val)],
        [Paragraph("Gross GRF impulse", cell_lbl), Paragraph(f"{ctx.get('br_gross_l', '-')} N·s", cell_val), Paragraph(f"{ctx.get('br_gross_r', '-')} N·s", cell_val), Paragraph(str(ctx.get('asym_br_gross', '-')), cell_val)],
        [Paragraph("Average RFD to peak", cell_lbl), Paragraph(f"{ctx.get('avg_rfd_br_l', '-')} N/s", cell_val), Paragraph(f"{ctx.get('avg_rfd_br_r', '-')} N/s", cell_val), Paragraph(str(ctx.get('asym_avg_rfd_br', '-')), cell_val)],
        [Paragraph("Max 20 ms RFD", cell_lbl), Paragraph(f"{ctx.get('win_rfd_br_l', '-')} N/s", cell_val), Paragraph(f"{ctx.get('win_rfd_br_r', '-')} N/s", cell_val), Paragraph(str(ctx.get('asym_win_rfd_br', '-')), cell_val)],
        [Paragraph("RFD 0–50 ms", cell_lbl), Paragraph(f"{ctx.get('rfd50_br_l', '-')} N/s", cell_val), Paragraph(f"{ctx.get('rfd50_br_r', '-')} N/s", cell_val), Paragraph(str(ctx.get('asym_rfd50_br', '-')), cell_val)],
        [Paragraph("RFD 0–100 ms", cell_lbl), Paragraph(f"{ctx.get('rfd100_br_l', '-')} N/s", cell_val), Paragraph(f"{ctx.get('rfd100_br_r', '-')} N/s", cell_val), Paragraph(str(ctx.get('asym_rfd100_br', '-')), cell_val)],
        [Paragraph("RFD 0–200 ms", cell_lbl), Paragraph(f"{ctx.get('rfd200_br_l', '-')} N/s", cell_val), Paragraph(f"{ctx.get('rfd200_br_r', '-')} N/s", cell_val), Paragraph(str(ctx.get('asym_rfd200_br', '-')), cell_val)],
        [Paragraph("<b>Propulsion / explosive phase</b>", ParagraphStyle('BPP2', fontName=f_bld, fontSize=6.8, textColor=colors.HexColor('#11395f'))), "", "", ""],
        [Paragraph("Peak force", cell_lbl), Paragraph(f"{ctx.get('pk_pr_l', '-')} N", cell_val), Paragraph(f"{ctx.get('pk_pr_r', '-')} N", cell_val), Paragraph(str(ctx.get('asym_pk_pr', '-')), cell_val)],
        [Paragraph("NET impulse", cell_lbl), Paragraph(f"{ctx.get('pr_net_l', '-')} N·s", cell_val), Paragraph(f"{ctx.get('pr_net_r', '-')} N·s", cell_val), Paragraph(str(ctx.get('asym_pr_net', '-')), cell_val)],
        [Paragraph("Gross GRF impulse", cell_lbl), Paragraph(f"{ctx.get('pr_gross_l', '-')} N·s", cell_val), Paragraph(f"{ctx.get('pr_gross_r', '-')} N·s", cell_val), Paragraph(str(ctx.get('asym_pr_gross', '-')), cell_val)],
        [Paragraph("Average RFD to peak", cell_lbl), Paragraph(f"{ctx.get('avg_rfd_pr_l', '-')} N/s", cell_val), Paragraph(f"{ctx.get('avg_rfd_pr_r', '-')} N/s", cell_val), Paragraph(str(ctx.get('asym_avg_rfd_pr', '-')), cell_val)],
        [Paragraph("Max 20 ms RFD", cell_lbl), Paragraph(f"{ctx.get('win_rfd_pr_l', '-')} N/s", cell_val), Paragraph(f"{ctx.get('win_rfd_pr_r', '-')} N/s", cell_val), Paragraph(str(ctx.get('asym_win_rfd_pr', '-')), cell_val)],
        [Paragraph("RFD 0–50 ms", cell_lbl), Paragraph(f"{ctx.get('rfd50_pr_l', '-')} N/s", cell_val), Paragraph(f"{ctx.get('rfd50_pr_r', '-')} N/s", cell_val), Paragraph(str(ctx.get('asym_rfd50_pr', '-')), cell_val)],
        [Paragraph("RFD 0–100 ms", cell_lbl), Paragraph(f"{ctx.get('rfd100_pr_l', '-')} N/s", cell_val), Paragraph(f"{ctx.get('rfd100_pr_r', '-')} N/s", cell_val), Paragraph(str(ctx.get('asym_rfd100_pr', '-')), cell_val)],
        [Paragraph("RFD 0–200 ms", cell_lbl), Paragraph(f"{ctx.get('rfd200_pr_l', '-')} N/s", cell_val), Paragraph(f"{ctx.get('rfd200_pr_r', '-')} N/s", cell_val), Paragraph(str(ctx.get('asym_rfd200_pr', '-')), cell_val)],
    ]
    bp_table = Table(bp_data, colWidths=[185, 110, 110, 140])
    bp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#fafbfc')),
        ('BACKGROUND', (0, 10), (-1, 10), colors.HexColor('#fafbfc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 0.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0.8),
    ]))
    story.append(bp_table)
    story.append(Spacer(1, 2))
    story.append(Paragraph("Positive directional asymmetry = Left higher; negative = Right higher. Net impulse subtracts each leg’s quiet-standing force; gross impulse does not.", ParagraphStyle('Note', fontName=f_reg, fontSize=6.0, textColor=colors.HexColor('#64748b'))))

    # =========================================================
    # PAGE 2 (A4 Portrait: Clinical, Kinematics & Sensitivity)
    # =========================================================
    story.append(PageBreak())

    story.append(Paragraph("<b>CMJ Coach Analyzer v4 — Clinical, Kinematics &amp; Sensitivity Details (Page 2)</b>", ParagraphStyle('P2H', fontName=f_bld, fontSize=10.0, textColor=colors.HexColor('#0d3154'))))
    story.append(Spacer(1, 3))

    # Row 1 (Page 2): QC Checks (270 pt) + Timing & Strategy (270 pt)
    qc_box = [
        Paragraph("<b>Data quality checks</b>", card_h2),
        Spacer(1, 1),
        Paragraph(f"• Sampling: {ctx.get('fs', '2400')} Hz (Derived from force samples)", cell_lbl),
        Paragraph(f"• Flight zero-offset: {ctx.get('zero_off', '0.0 / 0.0 N')}", cell_lbl),
        Paragraph(f"• Quiet-standing CV: {ctx.get('cv', '0.0')}% (BW {ctx.get('bw', '-')} N)", cell_lbl),
        Paragraph(f"• Flight residual force: {ctx.get('fres', '0.0')} N", cell_lbl),
        Paragraph(f"• JH agreement: {ctx.get('jh_diff', '0.0')} cm ({ctx.get('jh_diff_pct', '0.0')}%)", cell_lbl),
        Paragraph(f"• Impulse closure: {ctx.get('closure', '0.0')} N·s", cell_lbl)
    ]
    timing_box = [
        Paragraph("<b>Timing &amp; strategy</b>", card_h2),
        Spacer(1, 1),
        Paragraph(f"• Body mass: {ctx.get('mass', '-')} kg", cell_lbl),
        Paragraph(f"• Time to take-off: {ctx.get('ttt', '-')} ms", cell_lbl),
        Paragraph(f"• Unweighting duration: {ctx.get('d_unw', '-')} ms", cell_lbl),
        Paragraph(f"• Braking duration: {ctx.get('d_brk', '-')} ms", cell_lbl),
        Paragraph(f"• Propulsion duration: {ctx.get('d_pro', '-')} ms", cell_lbl),
        Paragraph(f"• Flight time: {ctx.get('d_fly', '-')} ms", cell_lbl),
        Paragraph(f"• Countermovement depth: {ctx.get('depth', '-')} cm", cell_lbl)
    ]

    r1_grid = [[qc_box, timing_box]]
    r1_table = Table(r1_grid, colWidths=[270, 275])
    r1_table.setStyle(TableStyle([
        ('BOX', (0, 0), (0, 0), 0.5, colors.HexColor('#cbd5e1')),
        ('BOX', (1, 0), (1, 0), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(r1_table)
    story.append(Spacer(1, 3))

    # Row 2 (Page 2): Velocity Chart (270 pt) + Power Chart (270 pt)
    vel_arr = ctx.get('vel', np.array([]))
    pow_arr = ctx.get('power', np.array([]))
    t_sub = t[:len(vel_arr)] if len(vel_arr) > 0 else np.array([])
    t_pow = t[:len(pow_arr)] if len(pow_arr) > 0 else np.array([])

    vel_buf = create_mini_series_image(t_sub, vel_arr, "COM velocity", "m/s", color='#11395f')
    pow_buf = create_mini_series_image(t_pow, pow_arr, "COM power", "W/kg", color='#ed7d31')

    r2_grid = [[RLImage(vel_buf, width=265, height=75), RLImage(pow_buf, width=265, height=75)]]
    r2_table = Table(r2_grid, colWidths=[270, 275])
    r2_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    story.append(r2_table)
    story.append(Spacer(1, 3))

    # Row 3 (Page 2): Landing Load (270 pt) + Research Metrics (270 pt)
    landing_box = [
        Paragraph("<b>Landing &amp; Total Load</b>", card_h2),
        Spacer(1, 1),
        Paragraph(f"• Peak landing: L {ctx.get('pk_land_l', '-')} N | R {ctx.get('pk_land_r', '-')} N", cell_lbl),
        Paragraph(f"• Total peak force: <b>{ctx.get('pk_land_tot', '-')} N</b> ({ctx.get('pk_land_bw', '-')} ×BW)", cell_lbl),
        Paragraph(f"• 20–80% loading rate: <b>{ctx.get('load_rate', '-')} N/s</b> ({ctx.get('load_rate_bw', '-')} BW/s)", cell_lbl),
        Paragraph(f"• Landing impulse 0–250 ms: <b>{ctx.get('land_imp_250', '-')} N·s</b>", cell_lbl),
        Paragraph(f"• Time to peak landing: <b>{ctx.get('ttp_land', '-')} ms</b>", cell_lbl)
    ]
    res_box = [
        Paragraph("<b>Research Derived Metrics</b>", card_h2),
        Spacer(1, 1),
        Paragraph(f"• Mean braking force: {ctx.get('mean_br_f', '-')} N/kg ({ctx.get('mean_br_f_tot', '-')} N)", cell_lbl),
        Paragraph(f"• Mean propulsive force: {ctx.get('mean_pr_f', '-')} N/kg ({ctx.get('mean_pr_f_tot', '-')} N)", cell_lbl),
        Paragraph(f"• Mean braking power: {ctx.get('mean_br_p', '-')} W/kg", cell_lbl),
        Paragraph(f"• Mean propulsive power: {ctx.get('mean_pr_p', '-')} W/kg", cell_lbl),
        Paragraph(f"• Positive net impulse: {ctx.get('pos_net_imp_kg', '-')} N·s/kg", cell_lbl),
        Paragraph(f"• Leg stiffness (Exploratory): {ctx.get('leg_stiff', '-')} N/m/kg", cell_lbl)
    ]

    r3_grid = [[landing_box, res_box]]
    r3_table = Table(r3_grid, colWidths=[270, 275])
    r3_table.setStyle(TableStyle([
        ('BOX', (0, 0), (0, 0), 0.5, colors.HexColor('#cbd5e1')),
        ('BOX', (1, 0), (1, 0), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(r3_table)
    story.append(Spacer(1, 3))

    # Row 4 (Page 2): Filter Sensitivity Analysis Table (Full Width 545 pt)
    sens_data = [
        [Paragraph("<b>Filter sensitivity analysis</b>", th_style), Paragraph("<b>Raw</b>", th_style), Paragraph("<b>20 Hz</b>", th_style), Paragraph("<b>30 Hz</b>", th_style), Paragraph("<b>50 Hz</b>", th_style), Paragraph("<b>Max spread</b>", th_style)],
        [Paragraph("Jump height (cm)", cell_lbl), Paragraph(ctx.get('sens_jh_raw', '-'), cell_val), Paragraph(ctx.get('sens_jh_20', '-'), cell_val), Paragraph(ctx.get('sens_jh_30', '-'), cell_val), Paragraph(ctx.get('sens_jh_50', '-'), cell_val), Paragraph(ctx.get('sens_jh_sp', '-'), cell_val)],
        [Paragraph("Propulsive net impulse (N·s)", cell_lbl), Paragraph(ctx.get('sens_imp_raw', '-'), cell_val), Paragraph(ctx.get('sens_imp_20', '-'), cell_val), Paragraph(ctx.get('sens_imp_30', '-'), cell_val), Paragraph(ctx.get('sens_imp_50', '-'), cell_val), Paragraph(ctx.get('sens_imp_sp', '-'), cell_val)],
        [Paragraph("Peak propulsive force (N)", cell_lbl), Paragraph(ctx.get('sens_pk_raw', '-'), cell_val), Paragraph(ctx.get('sens_pk_20', '-'), cell_val), Paragraph(ctx.get('sens_pk_30', '-'), cell_val), Paragraph(ctx.get('sens_pk_50', '-'), cell_val), Paragraph(ctx.get('sens_pk_sp', '-'), cell_val)],
        [Paragraph("Peak landing force (N)", cell_lbl), Paragraph(ctx.get('sens_land_raw', '-'), cell_val), Paragraph(ctx.get('sens_land_20', '-'), cell_val), Paragraph(ctx.get('sens_land_30', '-'), cell_val), Paragraph(ctx.get('sens_land_50', '-'), cell_val), Paragraph(ctx.get('sens_land_sp', '-'), cell_val)],
        [Paragraph("Max-window propulsive RFD (N/s)", cell_lbl), Paragraph(ctx.get('sens_rfd_raw', '-'), cell_val), Paragraph(ctx.get('sens_rfd_20', '-'), cell_val), Paragraph(ctx.get('sens_rfd_30', '-'), cell_val), Paragraph(ctx.get('sens_rfd_50', '-'), cell_val), Paragraph(ctx.get('sens_rfd_sp', '-'), cell_val)]
    ]
    sens_table = Table(sens_data, colWidths=[175, 74, 74, 74, 74, 74])
    sens_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.0),
    ]))
    story.append(sens_table)
    story.append(Spacer(1, 3))

    # Row 5 (Page 2): Interpretation for Coach (Full Thai Text)
    interp_cell = [
        Paragraph("<b>Interpretation for coach</b>", card_h2),
        Spacer(1, 1),
        Paragraph(f"• <b>Performance:</b> Jump height (impulse) = {ctx.get('jh_imp', '-')} cm, Flight-time = {ctx.get('jh_flt', '-')} cm, Time-to-take-off = {ctx.get('ttt', '-')} ms, RSImod = {ctx.get('rsi', '-')} m/s.", cell_lbl),
        Paragraph(f"• <b>Braking strategy:</b> Net impulse asymmetry = {ctx.get('asym_br_net', '-')}. Net impulse indicates momentum change.", cell_lbl),
        Paragraph(f"• <b>Propulsion strategy:</b> Net impulse asymmetry = {ctx.get('asym_pr_net', '-')}. Evaluate alongside peak force and RFD.", cell_lbl),
        Paragraph(f"• <b>Landing strategy:</b> Peak force asymmetry = {ctx.get('asym_pk_land', '-')}; 20–80% loading rate = {ctx.get('load_rate_bw', '-')} BW/s.", cell_lbl)
    ]
    interp_table = Table([[interp_cell]], colWidths=[545])
    interp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(interp_table)

    doc.build(story)

# -------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------
def generate_pdf_report(report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert=15.0, crop_x_min=None, crop_x_max=None, theme="Modern Coach (Dashboard)", coach_context=None):
    pdf_buffer = io.BytesIO()
    
    # กำหนด A4 แนวตั้ง (Portrait) เสมอทั้งสอง Theme
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        leftMargin=25,
        rightMargin=25,
        topMargin=15,
        bottomMargin=15
    )
    
    if "Coach" in theme:
        build_modern_coach_pdf(doc, report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max, coach_context or {})
    else:
        build_classic_pdf(doc, report, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff, t_landing, threshold_alert, crop_x_min, crop_x_max)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
