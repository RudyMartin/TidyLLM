#!/usr/bin/env python3
"""
Generate Infrastructure Files PDF
Creates a comprehensive PDF document with all sorted infrastructure files
"""

import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime

def read_file_content(filepath):
    """Read file content with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def generate_infra_pdf():
    """Generate PDF with all infrastructure files"""
    
    # Define the infrastructure files in order
    infra_files = [
        {
            'filename': '01_extensions.sql',
            'title': 'PostgreSQL Extensions Setup',
            'description': 'Enables required PostgreSQL extensions (pgvector, uuid-ossp)'
        },
        {
            'filename': '02_review_system.sql', 
            'title': 'Review System Tables',
            'description': 'Creates review system tables for QA validation with comprehensive metrics'
        },
        {
            'filename': '03_embeddings_system.sql',
            'title': 'Embeddings and Vector Storage',
            'description': 'Creates embeddings system with pgvector support and "Walmart-style" categorical filters'
        },
        {
            'filename': '04_event_tracking.sql',
            'title': 'Event Tracking and Analytics', 
            'description': 'Creates event tracking tables for analytics and monitoring'
        },
        {
            'filename': '05_mlflow_integration.sql',
            'title': 'MLflow Integration Database Schema',
            'description': 'Creates database schema for MLflow integration with Unified LLM Gateway system'
        }
    ]
    
    # Create PDF document
    output_filename = f"infrastructure_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(output_filename, pagesize=A4)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle', 
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=20,
        textColor=colors.darkgreen
    )
    
    file_title_style = ParagraphStyle(
        'FileTitle',
        parent=styles['Heading3'],
        fontSize=16,
        spaceAfter=10,
        textColor=colors.darkred
    )
    
    description_style = ParagraphStyle(
        'Description',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=15,
        leftIndent=20,
        textColor=colors.darkgray
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        fontName='Courier',
        leftIndent=20,
        rightIndent=20,
        spaceAfter=20
    )
    
    # Build the story (content)
    story = []
    
    # Title page
    story.append(Paragraph("🏗️ INFRASTRUCTURE FILES", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Database Infrastructure Setup Scripts", subtitle_style))
    story.append(Spacer(1, 30))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Location: database/infra/", styles['Normal']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Total Files: 5", styles['Normal']))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("📋 TABLE OF CONTENTS", subtitle_style))
    story.append(Spacer(1, 20))
    
    for i, file_info in enumerate(infra_files, 1):
        toc_entry = f"{i}. {file_info['title']} ({file_info['filename']})"
        story.append(Paragraph(toc_entry, styles['Normal']))
        story.append(Spacer(1, 5))
    
    story.append(PageBreak())
    
    # File contents
    for file_info in infra_files:
        filepath = os.path.join('database', 'infra', file_info['filename'])
        
        # File header
        story.append(Paragraph(f"📄 {file_info['title']}", file_title_style))
        story.append(Paragraph(f"File: {file_info['filename']}", styles['Normal']))
        story.append(Paragraph(f"Description: {file_info['description']}", description_style))
        story.append(Spacer(1, 10))
        
        # File content
        content = read_file_content(filepath)
        if content:
            # Split content into manageable chunks for PDF
            lines = content.split('\n')
            chunk_size = 50  # Lines per chunk
            
            for i in range(0, len(lines), chunk_size):
                chunk = '\n'.join(lines[i:i+chunk_size])
                story.append(Preformatted(chunk, code_style))
                
                # Add page break if not the last chunk
                if i + chunk_size < len(lines):
                    story.append(PageBreak())
                    story.append(Paragraph(f"📄 {file_info['title']} (continued)", file_title_style))
                    story.append(Spacer(1, 10))
        
        story.append(PageBreak())
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF generated: {output_filename}")
    return output_filename

if __name__ == "__main__":
    generate_infra_pdf()
