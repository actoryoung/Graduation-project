#!/usr/bin/env python3
"""Convert Markdown to PDF with academic formatting"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import markdown2
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Install with: pip install markdown2 reportlab")
    sys.exit(1)

import re


def register_fonts():
    """Register Chinese fonts for PDF generation"""
    try:
        font_path = "C:/Windows/Fonts/simsun.ttc"
        if Path(font_path).exists():
            pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
            return True
        return False
    except Exception as e:
        print(f"Warning: Could not register Chinese fonts: {e}")
        return False


def convert_markdown_to_pdf(md_path, pdf_path=None, title="多模态情感分析系统设计与实现文献综述"):
    """Convert Markdown file to PDF with academic styling."""
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    if pdf_path is None:
        pdf_path = md_path.with_suffix('.pdf')
    else:
        pdf_path = Path(pdf_path)

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Register Chinese font if available
    has_chinese_font = register_fonts()
    font_name = 'ChineseFont' if has_chinese_font else 'Helvetica'

    # Create styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor='#000000'
    )

    heading2_style = ParagraphStyle(
        'Heading2Style',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        spaceAfter=12,
        textColor='#333333'
    )

    heading3_style = ParagraphStyle(
        'Heading3Style',
        parent=styles['Heading3'],
        fontName=font_name,
        fontSize=12,
        spaceAfter=8,
        textColor='#444444'
    )

    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['BodyText'],
        fontName=font_name,
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        textColor='#222222'
    )

    # Parse markdown content
    elements = []
    lines = md_content.split('\n')
    current_buffer = []

    # Add title
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(0.5*cm))

    for line in lines:
        # Headers
        if line.startswith('## '):
            if current_buffer:
                text = ' '.join(current_buffer)
                elements.append(Paragraph(text, body_style))
                current_buffer = []
            elements.append(Paragraph(line[3:], heading2_style))
            elements.append(Spacer(0.2*cm))
        elif line.startswith('### '):
            if current_buffer:
                text = ' '.join(current_buffer)
                elements.append(Paragraph(text, body_style))
                current_buffer = []
            elements.append(Paragraph(line[4:], heading3_style))
            elements.append(Spacer(0.1*cm))
        # Horizontal rule
        elif line.strip() == '---':
            elements.append(Spacer(0.3*cm))
        # Regular text
        elif line.strip():
            # Clean markdown syntax
            clean_line = line.strip()
            # Bold: **text** -> <b>text</b>
            clean_line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', clean_line)
            # Italic: *text* -> <i>text</i>
            clean_line = re.sub(r'\*(.+?)\*', r'<i>\1</i>', clean_line)
            # Code: `text` -> <font name="Courier">text</font>
            clean_line = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', clean_line)
            # Links: [text](url) -> <link href="url">text</link>
            clean_line = re.sub(r'\[(.+?)\]\((.+?)\)', r'<link href="\2">\1</link>', clean_line)
            current_buffer.append(clean_line)
        # Empty line - flush buffer
        elif current_buffer:
            text = ' '.join(current_buffer)
            elements.append(Paragraph(text, body_style))
            current_buffer = []

    # Add remaining content
    if current_buffer:
        text = ' '.join(current_buffer)
        elements.append(Paragraph(text, body_style))

    # Create PDF document
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                          leftMargin=2*cm, rightMargin=2*cm,
                          topMargin=2.5*cm, bottomMargin=2*cm)

    # Build PDF
    print(f"Converting {md_path} -> {pdf_path}")
    doc.build(elements)
    print(f"PDF generated successfully: {pdf_path}")

    return pdf_path


if __name__ == "__main__":
    md_file = Path(__file__).parent.parent / "docs/thesis/文献综述_本科版.md"
    if len(sys.argv) > 1:
        md_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_markdown_to_pdf(md_file, pdf_file)
