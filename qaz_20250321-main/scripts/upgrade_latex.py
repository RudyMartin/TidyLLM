#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaTeX Utility Script for QA Criteria Reports

This utility script provides LaTeX processing capabilities for QA HealthCheck reports.
It can be included in upgrade_qa_criteria.py for enhanced report generation.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaTeXProcessor:
    """Utility class for LaTeX processing and PDF generation"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = None
        
    def __enter__(self):
        """Context manager entry"""
        self.temp_dir = tempfile.mkdtemp()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temp directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def check_latex_installation(self) -> bool:
        """Check if LaTeX is installed and available"""
        try:
            result = subprocess.run(['pdflatex', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ LaTeX (pdflatex) is available")
                return True
            else:
                logger.warning("⚠️ LaTeX (pdflatex) not found or not working")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("⚠️ LaTeX (pdflatex) not installed")
            return False
    
    def compile_latex_to_pdf(self, latex_file: str, output_name: str = None) -> Optional[str]:
        """Compile LaTeX file to PDF"""
        
        if not self.check_latex_installation():
            logger.error("❌ LaTeX not available - cannot compile PDF")
            return None
        
        latex_path = Path(latex_file)
        if not latex_path.exists():
            logger.error(f"❌ LaTeX file not found: {latex_file}")
            return None
        
        # Determine output name
        if output_name is None:
            output_name = latex_path.stem + ".pdf"
        
        output_path = self.output_dir / output_name
        
        try:
            logger.info(f"🔄 Compiling LaTeX to PDF: {latex_path.name}")
            
            # Run pdflatex compilation
            cmd = [
                'pdflatex',
                '-interaction=nonstopmode',
                '-output-directory=' + str(self.temp_dir),
                str(latex_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Find the generated PDF
                temp_pdf = Path(self.temp_dir) / f"{latex_path.stem}.pdf"
                if temp_pdf.exists():
                    # Copy to output directory
                    shutil.copy2(temp_pdf, output_path)
                    logger.info(f"✅ PDF generated successfully: {output_path}")
                    return str(output_path)
                else:
                    logger.error("❌ PDF file not found after compilation")
                    return None
            else:
                logger.error(f"❌ LaTeX compilation failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("❌ LaTeX compilation timed out")
            return None
        except Exception as e:
            logger.error(f"❌ Error during LaTeX compilation: {e}")
            return None
    
    def create_latex_template(self, template_name: str = "qa_report_template.tex") -> str:
        """Create a basic LaTeX template for QA reports"""
        
        template_content = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{parskip}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{framed}

% Page setup
\geometry{margin=1in}
\pagestyle{fancy}
\fancyhf{}
\rhead{QA HealthCheck Report}
\lhead{Page \thepage\ of \pageref{LastPage}}

% Color definitions
\definecolor{excellent}{RGB}{16, 185, 129}
\definecolor{good}{RGB}{59, 130, 246}
\definecolor{satisfactory}{RGB}{245, 158, 11}
\definecolor{needsimprovement}{RGB}{239, 68, 68}
\definecolor{codebg}{RGB}{248, 249, 250}

% Custom commands
\newcommand{\statusbox}[2]{%
    \fboxsep=0pt\fboxrule=0.4pt%
    \colorbox{#1}{\textcolor{white}{\textbf{#2}}}%
}

\newcommand{\criteriaTable}[1]{%
    \begin{longtable}{p{0.35\textwidth}p{0.15\textwidth}p{0.15\textwidth}p{0.35\textwidth}}
    \toprule
    \textbf{Criterion} & \textbf{Score} & \textbf{Status} & \textbf{Explanation} \\
    \midrule
    #1
    \bottomrule
    \end{longtable}%
}

\newcommand{\metadataTable}[1]{%
    \begin{table}[h]
    \centering
    \begin{tabular}{ll}
    \toprule
    \textbf{Field} & \textbf{Value} \\
    \midrule
    #1
    \bottomrule
    \end{tabular}
    \end{table}%
}

% Code listing style
\lstset{
    backgroundcolor=\color{codebg},
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    captionpos=b,
    commentstyle=\color{gray},
    keywordstyle=\color{blue},
    stringstyle=\color{red},
    numbers=left,
    numberstyle=\tiny\color{gray},
    numbersep=5pt,
    frame=single,
    rulecolor=\color{black},
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2
}

\begin{document}

% Title page
\begin{titlepage}
\centering
\vspace*{2cm}
{\Huge\bfseries QA HealthCheck Report\par}
\vspace{1cm}
{\Large Review ID: \textbf{REVIEW_ID_PLACEHOLDER}\par}
\vspace{0.5cm}
{\large Model: \textbf{MODEL_NAME_PLACEHOLDER}\par}
\vspace{0.5cm}
{\large Risk Tier: \textbf{RISK_TIER_PLACEHOLDER}\par}
\vspace{1cm}
{\large Generated: \textbf{DATE_PLACEHOLDER}\par}
\vspace{2cm}
{\large Overall Status: \textbf{OVERALL_STATUS_PLACEHOLDER}\par}
\vspace{0.5cm}
{\large Overall Score: \textbf{OVERALL_SCORE_PLACEHOLDER}\%}\par
\vfill
{\large QA Team\par}
\vspace{0.5cm}
{\large \textbf{Model Risk Management}\par}
\end{titlepage}

\tableofcontents
\newpage

% Executive Summary
\section{Executive Summary}

This QA HealthCheck report provides a comprehensive assessment of the model validation documentation and processes. The review was conducted using established QA criteria and evidence-based evaluation.

\textbf{Key Findings:}
\begin{itemize}
    \item Total Criteria Evaluated: \textbf{TOTAL_CRITERIA_PLACEHOLDER}
    \item Passed Criteria: \textbf{PASSED_CRITERIA_PLACEHOLDER}
    \item Failed Criteria: \textbf{FAILED_CRITERIA_PLACEHOLDER}
    \item Conditional Criteria: \textbf{CONDITIONAL_CRITERIA_PLACEHOLDER}
\end{itemize}

\textbf{Quality Metrics:}
\begin{itemize}
    \item Evidence Coverage: \textbf{EVIDENCE_COVERAGE_PLACEHOLDER}\%
    \item Average Evidence per Criterion: \textbf{AVG_EVIDENCE_PLACEHOLDER}
    \item Categories Above Threshold: \textbf{CATEGORIES_ABOVE_PLACEHOLDER}
    \item Critical Issues: \textbf{CRITICAL_ISSUES_PLACEHOLDER}
\end{itemize}

% Review Metadata
\section{Review Metadata}

\metadataTable{METADATA_TABLE_PLACEHOLDER}

% Criteria Analysis
\section{Criteria Analysis}

CRITERIA_SECTIONS_PLACEHOLDER

% Recommendations
\section{Recommendations}

RECOMMENDATIONS_PLACEHOLDER

% Next Steps
\section{Next Steps}

NEXT_STEPS_PLACEHOLDER

% Appendix: Evidence Details
\section{Appendix: Evidence Details}

EVIDENCE_DETAILS_PLACEHOLDER

\end{document}
"""
        
        template_path = self.output_dir / template_name
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        logger.info(f"✅ LaTeX template created: {template_path}")
        return str(template_path)
    
    def validate_latex_syntax(self, latex_file: str) -> Tuple[bool, List[str]]:
        """Validate LaTeX syntax and return errors if any"""
        
        latex_path = Path(latex_file)
        if not latex_path.exists():
            return False, [f"LaTeX file not found: {latex_file}"]
        
        errors = []
        
        try:
            with open(latex_path, 'r') as f:
                content = f.read()
            
            # Basic syntax checks
            if '\\documentclass' not in content:
                errors.append("Missing \\documentclass declaration")
            
            if '\\begin{document}' not in content:
                errors.append("Missing \\begin{document}")
            
            if '\\end{document}' not in content:
                errors.append("Missing \\end{document}")
            
            # Check for common LaTeX errors
            if '\\' in content and '\\' not in content.replace('\\\\', ''):
                errors.append("Potential unescaped backslash")
            
            # Check for placeholder values
            placeholders = ['PLACEHOLDER', 'REVIEW_ID_PLACEHOLDER', 'MODEL_NAME_PLACEHOLDER']
            for placeholder in placeholders:
                if placeholder in content:
                    errors.append(f"Found placeholder: {placeholder}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Error reading LaTeX file: {e}"]
    
    def batch_compile_latex(self, latex_files: List[str]) -> Dict[str, str]:
        """Compile multiple LaTeX files to PDF"""
        
        results = {}
        
        for latex_file in latex_files:
            logger.info(f"🔄 Processing: {latex_file}")
            
            # Validate syntax first
            is_valid, errors = self.validate_latex_syntax(latex_file)
            if not is_valid:
                logger.warning(f"⚠️ Syntax errors in {latex_file}: {errors}")
                results[latex_file] = None
                continue
            
            # Compile to PDF
            pdf_path = self.compile_latex_to_pdf(latex_file)
            results[latex_file] = pdf_path
        
        return results
    
    def cleanup_temp_files(self):
        """Clean up temporary LaTeX files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("✅ Temporary files cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Could not clean up temp files: {e}")


def create_enhanced_latex_report(qa_report_data: Dict[str, Any], 
                                template_path: str = None) -> str:
    """Create an enhanced LaTeX report with additional features"""
    
    with LaTeXProcessor() as processor:
        # Use provided template or create default
        if template_path is None:
            template_path = processor.create_latex_template()
        
        # Read template
        with open(template_path, 'r') as f:
            latex_content = f.read()
        
        # Replace placeholders with actual data
        content = latex_content
        
        # Basic metadata replacements
        metadata = qa_report_data.get('metadata', {})
        content = content.replace("REVIEW_ID_PLACEHOLDER", metadata.get('review_id', 'REV00000'))
        content = content.replace("MODEL_NAME_PLACEHOLDER", metadata.get('model_name', 'Unknown Model'))
        content = content.replace("RISK_TIER_PLACEHOLDER", metadata.get('risk_tier', 'Unknown'))
        content = content.replace("DATE_PLACEHOLDER", metadata.get('report_generated_at', 'Unknown')[:10])
        content = content.replace("OVERALL_STATUS_PLACEHOLDER", qa_report_data.get('overall_status', 'Unknown').title())
        content = content.replace("OVERALL_SCORE_PLACEHOLDER", f"{qa_report_data.get('overall_score', 0):.1f}")
        
        # Summary statistics
        content = content.replace("TOTAL_CRITERIA_PLACEHOLDER", str(qa_report_data.get('total_criteria', 0)))
        content = content.replace("PASSED_CRITERIA_PLACEHOLDER", str(qa_report_data.get('passed_criteria', 0)))
        content = content.replace("FAILED_CRITERIA_PLACEHOLDER", str(qa_report_data.get('failed_criteria', 0)))
        content = content.replace("CONDITIONAL_CRITERIA_PLACEHOLDER", str(qa_report_data.get('conditional_criteria', 0)))
        
        # Quality metrics
        quality_metrics = qa_report_data.get('quality_metrics', {})
        content = content.replace("EVIDENCE_COVERAGE_PLACEHOLDER", f"{quality_metrics.get('evidence_coverage', 0):.1f}")
        content = content.replace("AVG_EVIDENCE_PLACEHOLDER", f"{quality_metrics.get('average_evidence_per_criterion', 0):.1f}")
        content = content.replace("CATEGORIES_ABOVE_PLACEHOLDER", str(quality_metrics.get('categories_above_threshold', 0)))
        content = content.replace("CRITICAL_ISSUES_PLACEHOLDER", str(quality_metrics.get('critical_issues', 0)))
        
        # Generate metadata table
        metadata_table_rows = ""
        for key, value in metadata.items():
            if isinstance(value, list):
                value = ", ".join(value)
            metadata_table_rows += f"{key.replace('_', ' ').title()} & {value} \\\\\n"
        
        content = content.replace("METADATA_TABLE_PLACEHOLDER", metadata_table_rows)
        
        # Generate criteria sections
        criteria_sections = ""
        categories = qa_report_data.get('categories', [])
        
        for category in categories:
            criteria_sections += f"\\subsection{{{category.get('category_name', 'Unknown')}}}\n"
            criteria_sections += f"\\textbf{{Category Score: {category.get('percentage_score', 0):.1f}\\%}} \\\\\n"
            criteria_sections += f"\\textbf{{Status: {category.get('status', 'Unknown').title()}}} \\\\\n\n"
            
            # Create criteria table
            table_rows = ""
            for criterion in category.get('criteria_results', []):
                # Determine status color
                status = criterion.get('status', 'unknown')
                if status == "pass":
                    status_color = "excellent"
                elif status == "conditional":
                    status_color = "satisfactory"
                else:
                    status_color = "needsimprovement"
                
                table_rows += f"{criterion.get('criterion_text', 'Unknown')} & {criterion.get('score', 0):.0f} & \\statusbox{{{status_color}}}{{{status.title()}}} & {criterion.get('explanation', 'No explanation')} \\\\\n"
            
            criteria_sections += f"\\criteriaTable{{{table_rows}}}\n\n"
        
        content = content.replace("CRITERIA_SECTIONS_PLACEHOLDER", criteria_sections)
        
        # Generate recommendations
        recommendations_text = ""
        for rec in qa_report_data.get('recommendations', []):
            recommendations_text += f"\\item {rec}\n"
        
        content = content.replace("RECOMMENDATIONS_PLACEHOLDER", recommendations_text)
        
        # Generate next steps
        next_steps_text = ""
        for step in qa_report_data.get('next_steps', []):
            next_steps_text += f"\\item {step}\n"
        
        content = content.replace("NEXT_STEPS_PLACEHOLDER", next_steps_text)
        
        # Generate evidence details
        evidence_details = ""
        for category in categories:
            evidence_details += f"\\subsection{{{category.get('category_name', 'Unknown')}}}\n"
            for criterion in category.get('criteria_results', []):
                evidence_details += f"\\subsubsection{{{criterion.get('criterion_text', 'Unknown')}}}\n"
                evidence_details += f"Evidence Count: {criterion.get('evidence_count', 0)} \\\\\n"
                evidence_details += "Evidence Found:\\\\\n"
                for evidence in criterion.get('evidence_found', [])[:5]:  # Show first 5
                    evidence_details += f"\\begin{{itemize}}\n\\item {evidence}\n\\end{{itemize}}\n"
                evidence_details += "\\newpage\n"
        
        content = content.replace("EVIDENCE_DETAILS_PLACEHOLDER", evidence_details)
        
        # Save enhanced LaTeX file
        timestamp = qa_report_data.get('metadata', {}).get('report_generated_at', '').replace(':', '').replace('-', '')
        output_path = processor.output_dir / f"enhanced_qa_report_{timestamp}.tex"
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Enhanced LaTeX report created: {output_path}")
        return str(output_path)


def main():
    """Test the LaTeX utility functions"""
    
    print("🚀 Testing LaTeX Utility Functions")
    print("=" * 50)
    
    with LaTeXProcessor() as processor:
        # Test LaTeX installation
        print("\n1. Checking LaTeX installation...")
        if processor.check_latex_installation():
            print("✅ LaTeX is available")
        else:
            print("❌ LaTeX not available - PDF compilation will be skipped")
        
        # Create template
        print("\n2. Creating LaTeX template...")
        template_path = processor.create_latex_template()
        print(f"✅ Template created: {template_path}")
        
        # Validate template
        print("\n3. Validating LaTeX syntax...")
        is_valid, errors = processor.validate_latex_syntax(template_path)
        if is_valid:
            print("✅ LaTeX syntax is valid")
        else:
            print(f"⚠️ LaTeX syntax issues: {errors}")
        
        # Test compilation (if LaTeX is available)
        if processor.check_latex_installation():
            print("\n4. Testing PDF compilation...")
            pdf_path = processor.compile_latex_to_pdf(template_path, "test_template.pdf")
            if pdf_path:
                print(f"✅ PDF compiled: {pdf_path}")
            else:
                print("❌ PDF compilation failed")
        else:
            print("\n4. Skipping PDF compilation (LaTeX not available)")
        
        print("\n🎉 LaTeX utility test complete!")


if __name__ == "__main__":
    main()
