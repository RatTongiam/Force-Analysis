import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(report_data, t, sf, sl, sr, t_start, t_braking, t_split, t_takeoff):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor('#4d2994'))
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=colors.HexColor('#1a0f30'))
    cell_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor('#1a0f30'))
    cell_bold = ParagraphStyle('TableCellBold', parent=cell_style, fontName='Helvetica-Bold')
    
    story = [
        Paragraph("BIOMECHANICAL ANALYSIS REPORT (CMJ)", title_style),
        Paragraph("FREE JUMPANZ TEAM — PRIMA MOTION TECHNOLOGY", subtitle_style),
        Spacer(1, 10)
    ]

    fig_plt, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
    ax.plot(t, sl, label='Left Limb', color='#818cf8', linewidth=1.5)
    ax.plot(t, sr, label='Right Limb', color='#f87171', linewidth=1.5)
    ax.plot(t, sf, label='Total Force', color='#4d2994', linewidth=2.5)

    ax.axvspan(t_start, t_braking, color='yellow', alpha=0.15)
    ax.axvspan(t_braking, t_split, color='red', alpha=0.15)
    ax.axvspan(t_split, t_takeoff, color='green', alpha=0.15)

    ax.axvline(t_start, color='#ca8a04', linestyle='--', linewidth=1)
    ax.axvline(t_braking, color='#ef4444', linestyle='--', linewidth=1)
    ax.axvline(t_split, color='#22c55e', linestyle='--', linewidth=1)
    ax.axvline(t_takeoff, color='#dc2626', linestyle='--', linewidth=1)

    ax.set_title("FORCE-TIME ANALYSIS & SUB-PHASES", fontsize=9, fontweight='bold', color='#1a0f30')
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Force (N)", fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=200)
    plt.close(fig_plt)
    img_buffer.seek(0)

    story.append(Image(img_buffer, width=520, height=208))
    story.append(Spacer(1, 10))
    
    table_data = [[
        Paragraph("<b>Biomechanical Metric</b>", cell_bold), 
        Paragraph("<b>Left</b>", cell_bold), 
        Paragraph("<b>Right</b>", cell_bold), 
        Paragraph("<b>TOTAL</b>", cell_bold),
        Paragraph("<b>Deficit %</b>", cell_bold)
    ]]
    
    for phase_name, metrics in report_data.items():
        table_data.append([
            Paragraph(f"<b>{phase_name.upper()}</b>", ParagraphStyle('PhaseHeader', parent=cell_bold, textColor=colors.HexColor('#4d2994'))),
            "", "", "", ""
        ])
        for m_name, vals in metrics.items():
            l_val = str(vals.get("Left", "-"))
            r_val = str(vals.get("Right", "-"))
            tot_val = str(vals.get("Total", "-"))
            def_val = str(vals.get("Deficit", "-"))
            
            table_data.append([
                Paragraph(m_name, cell_style),
                Paragraph(l_val, cell_style),
                Paragraph(r_val, cell_style),
                Paragraph(f"<b>{tot_val}</b>", cell_bold),
                Paragraph(def_val, cell_style)
            ])
            
    t_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f6fb')),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 2),
        ('TOPPADDING', (0, 0), (-1, 0), 2),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ('SPAN', (0, 1), (4, 1)),
        ('SPAN', (0, 14), (4, 14)),
        ('SPAN', (0, 20), (4, 20)),
        ('SPAN', (0, 24), (4, 24)),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f1ebf9')),
        ('BACKGROUND', (0, 14), (-1, 14), colors.HexColor('#f1ebf9')),
        ('BACKGROUND', (0, 20), (-1, 20), colors.HexColor('#f1ebf9')),
        ('BACKGROUND', (0, 24), (-1, 24), colors.HexColor('#f1ebf9')),
    ])
    
    doc_table = Table(table_data, colWidths=[190, 75, 75, 90, 90])
    doc_table.setStyle(t_style)
    story.append(doc_table)
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer
