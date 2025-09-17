#!/usr/bin/env python3
"""
Standalone Markdown to PDF Converter
====================================

Simple utility to convert markdown content to PDF using installed libraries.
No path manipulation needed - uses standard Python package imports.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union

def check_dependencies() -> Dict[str, bool]:
    """Check which PDF generation libraries are available."""
    available = {}

    # Check ReportLab
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        available['reportlab'] = True
    except ImportError:
        available['reportlab'] = False

    # Check markdown
    try:
        import markdown
        available['markdown'] = True
    except ImportError:
        available['markdown'] = False

    return available

def convert_markdown_to_pdf(markdown_content: str, output_path: Union[str, Path],
                           title: str = "Document") -> Dict[str, Any]:
    """
    Convert markdown content to PDF using ReportLab.

    Args:
        markdown_content: Markdown text to convert
        output_path: Path where PDF should be saved
        title: Document title for PDF metadata

    Returns:
        Dict with conversion results and metadata
    """

    # Check dependencies
    deps = check_dependencies()

    if not deps['reportlab']:
        return {
            "success": False,
            "error": "ReportLab library not available",
            "install_command": "pip install reportlab",
            "dependencies": deps
        }

    try:
        # Import libraries (we know they're available)
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
            title=title
        )

        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=16,
            spaceAfter=20
        )

        # Build PDF content
        story = []

        # Add title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))

        # Process markdown content
        if deps['markdown']:
            # Use markdown library if available
            import markdown
            html_content = markdown.markdown(markdown_content)

            # Simple HTML to ReportLab conversion
            lines = html_content.split('\n')
        else:
            # Fallback: treat as plain text with basic markdown parsing
            lines = markdown_content.split('\n')

        current_paragraph = ""

        for line in lines:
            line = line.strip()

            if not line:
                if current_paragraph:
                    story.append(Paragraph(current_paragraph, styles['Normal']))
                    story.append(Spacer(1, 12))
                    current_paragraph = ""
            elif line.startswith('# '):
                if current_paragraph:
                    story.append(Paragraph(current_paragraph, styles['Normal']))
                    current_paragraph = ""
                header = line[2:].strip()
                story.append(Paragraph(header, styles['Heading1']))
                story.append(Spacer(1, 12))
            elif line.startswith('## '):
                if current_paragraph:
                    story.append(Paragraph(current_paragraph, styles['Normal']))
                    current_paragraph = ""
                header = line[3:].strip()
                story.append(Paragraph(header, styles['Heading2']))
                story.append(Spacer(1, 12))
            elif line.startswith('### '):
                if current_paragraph:
                    story.append(Paragraph(current_paragraph, styles['Normal']))
                    current_paragraph = ""
                header = line[4:].strip()
                story.append(Paragraph(header, styles['Heading3']))
                story.append(Spacer(1, 12))
            elif line.startswith('**') and line.endswith('**'):
                # Bold text
                bold_text = line[2:-2]
                story.append(Paragraph(f"<b>{bold_text}</b>", styles['Normal']))
                story.append(Spacer(1, 6))
            elif line.startswith('- '):
                # Bullet point
                bullet_text = line[2:].strip()
                story.append(Paragraph(f"• {bullet_text}", styles['Normal']))
                story.append(Spacer(1, 6))
            else:
                # Regular text - accumulate for paragraph
                current_paragraph += line + " "

        # Add final paragraph if exists
        if current_paragraph:
            story.append(Paragraph(current_paragraph.strip(), styles['Normal']))

        # Build PDF
        doc.build(story)

        return {
            "success": True,
            "output_path": str(output_path),
            "file_size": output_path.stat().st_size,
            "pages_estimated": len(story) // 10 + 1,
            "conversion_method": "ReportLab with basic markdown parsing",
            "dependencies": deps
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"PDF conversion failed: {str(e)}",
            "output_path": str(output_path) if 'output_path' in locals() else None,
            "dependencies": deps
        }

def main():
    """Command line interface for md2pdf conversion."""
    if len(sys.argv) < 3:
        print("Usage: python md2pdf.py <input.md> <output.pdf> [title]")
        print("   or: python md2pdf.py --test")
        sys.exit(1)

    if sys.argv[1] == "--test":
        # Run test
        test_markdown = """# Test Report

## Section 1
This is a **test document** to verify PDF generation.

### Subsection 1.1
- First bullet point
- Second bullet point

### Subsection 1.2
Regular paragraph text here.

## Section 2
Another section with more content.

**Bold text example**

Normal text continues here.
"""

        result = convert_markdown_to_pdf(test_markdown, "test_output.pdf", "Test Document")
        print(f"Test result: {result}")
        if result["success"]:
            print(f"[OK] Test PDF generated: {result['output_path']}")
        else:
            print(f"[X] Test failed: {result['error']}")
        return

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else input_file.stem

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Read markdown content
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Convert to PDF
    result = convert_markdown_to_pdf(markdown_content, output_file, title)

    if result["success"]:
        print(f"[OK] PDF generated successfully!")
        print(f"   Input: {input_file}")
        print(f"   Output: {result['output_path']}")
        print(f"   Size: {result['file_size']} bytes")
        print(f"   Method: {result['conversion_method']}")
    else:
        print(f"[X] PDF generation failed!")
        print(f"   Error: {result['error']}")
        if 'install_command' in result:
            print(f"   Solution: {result['install_command']}")

if __name__ == "__main__":
    main()