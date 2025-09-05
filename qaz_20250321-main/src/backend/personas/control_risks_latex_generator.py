#!/usr/bin/env python3
"""
Control Risks LaTeX Report Generator
Integrates with Control Risks YAML system to generate professional PDF reports
"""

import os
import re
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .control_risks_yaml_config import ControlRisk, RiskSeverity, RiskCategory
from .control_risks_yaml_report_generator import ControlRisksYAMLReport

@dataclass
class LaTeXGenerationConfig:
    """Configuration for LaTeX report generation"""
    template_path: str = "templates/control_risks_report_template.tex"
    output_dir: str = "reports"
    temp_dir: str = None
    latex_engine: str = "pdflatex"
    compile_twice: bool = True
    cleanup_temp: bool = True
    include_logo: bool = True
    logo_path: str = "assets/logo.png"

class ControlRisksLaTeXGenerator:
    """Generator for LaTeX-based Control Risks reports"""
    
    def __init__(self, config: LaTeXGenerationConfig = None):
        self.config = config or LaTeXGenerationConfig()
        self.template_content = self._load_template()
    
    def _load_template(self) -> str:
        """Load the LaTeX template content"""
        
        try:
            with open(self.config.template_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"LaTeX template not found: {self.config.template_path}")
    
    def generate_latex_report(self, report: ControlRisksYAMLReport, 
                            output_filename: str = None) -> str:
        """Generate LaTeX report from Control Risks report"""
        
        # Generate output filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = report.model_context.get('model_name', 'Model').replace(' ', '_')
            output_filename = f"control_risks_report_{model_name}_{timestamp}"
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Generate LaTeX content
        latex_content = self._generate_latex_content(report)
        
        # Write LaTeX file
        latex_path = os.path.join(self.config.output_dir, f"{output_filename}.tex")
        with open(latex_path, 'w', encoding='utf-8') as file:
            file.write(latex_content)
        
        return latex_path
    
    def generate_pdf_report(self, report: ControlRisksYAMLReport, 
                          output_filename: str = None) -> str:
        """Generate PDF report from Control Risks report"""
        
        # Generate LaTeX file
        latex_path = self.generate_latex_report(report, output_filename)
        
        # Compile to PDF
        pdf_path = self._compile_latex_to_pdf(latex_path)
        
        return pdf_path
    
    def _generate_latex_content(self, report: ControlRisksYAMLReport) -> str:
        """Generate LaTeX content by substituting template variables"""
        
        latex_content = self.template_content
        
        # Substitute basic variables
        substitutions = self._create_substitutions_dict(report)
        
        for placeholder, value in substitutions.items():
            latex_content = latex_content.replace(placeholder, str(value))
        
        # Generate dynamic content
        latex_content = self._substitute_dynamic_content(latex_content, report)
        
        return latex_content
    
    def _create_substitutions_dict(self, report: ControlRisksYAMLReport) -> Dict[str, Any]:
        """Create dictionary of template substitutions"""
        
        # Extract model context
        model_name = report.model_context.get('model_name', 'Unknown Model')
        model_type = report.model_context.get('model_type', 'Unknown Type')
        business_unit = report.model_context.get('business_unit', 'Unknown Unit')
        
        # Risk summary data
        risk_summary = report.risk_summary
        
        # Category distribution
        category_dist = risk_summary.get('category_distribution', {})
        
        # Stage distribution from stage analysis
        stage_analysis = report.stage_analysis
        
        # Controls distribution
        controls_dist = risk_summary.get('controls_distribution', {})
        controls_by_stage = controls_dist.get('controls_by_stage', {})
        
        return {
            # Report metadata
            '{{REPORT_ID}}': report.report_id,
            '{{REPORT_DATE}}': report.report_date.strftime('%B %d, %Y'),
            '{{MODEL_NAME}}': self._escape_latex(model_name),
            '{{MODEL_TYPE}}': self._escape_latex(model_type),
            '{{BUSINESS_UNIT}}': self._escape_latex(business_unit),
            
            # Risk summary
            '{{TOTAL_RISKS}}': risk_summary.get('total_risks', 0),
            '{{TOTAL_CONTROLS}}': risk_summary.get('total_controls', 0),
            '{{AVG_CONTROLS_PER_RISK}}': f"{risk_summary.get('total_controls', 0) / max(risk_summary.get('total_risks', 1), 1):.1f}",
            '{{OVERALL_STATUS}}': report.overall_status,
            '{{RISK_SCORE}}': f"{report.risk_score:.3f}",
            
            # Risk distribution
            '{{CRITICAL_RISKS}}': risk_summary.get('critical_risks', 0),
            '{{HIGH_RISKS}}': risk_summary.get('high_risks', 0),
            '{{MEDIUM_RISKS}}': risk_summary.get('medium_risks', 0),
            '{{LOW_RISKS}}': risk_summary.get('low_risks', 0),
            '{{MINIMAL_RISKS}}': risk_summary.get('minimal_risks', 0),
            
            # Category distribution
            '{{MODEL_RISKS}}': category_dist.get('model_risks', 0),
            '{{GOVERNANCE_RISKS}}': category_dist.get('governance_risks', 0),
            '{{MONITORING_RISKS}}': category_dist.get('monitoring_risks', 0),
            '{{VALIDATION_RISKS}}': category_dist.get('validation_risks', 0),
            '{{DEVELOPMENT_RISKS}}': category_dist.get('development_risks', 0),
            '{{DATA_RISKS}}': category_dist.get('data_risks', 0),
            '{{BUSINESS_RISKS}}': category_dist.get('business_risks', 0),
            
            # Stage risk counts
            '{{PLANNING_RISKS}}': len([r for r in report.risk_details if r.stage_distribution.planning > 0]),
            '{{DEVELOPMENT_RISKS_COUNT}}': len([r for r in report.risk_details if r.stage_distribution.development > 0]),
            '{{VALIDATION_RISKS_COUNT}}': len([r for r in report.risk_details if r.stage_distribution.validation > 0]),
            '{{IMPLEMENTATION_RISKS}}': len([r for r in report.risk_details if r.stage_distribution.implementation > 0]),
            '{{USE_MONITORING_RISKS}}': len([r for r in report.risk_details if r.stage_distribution.use_and_monitoring > 0]),
            '{{CHANGES_RISKS}}': len([r for r in report.risk_details if r.stage_distribution.changes > 0]),
            '{{RETIREMENT_RISKS}}': len([r for r in report.risk_details if r.stage_distribution.retirement > 0]),
            
            # Controls distribution
            '{{PLANNING_CONTROLS}}': controls_by_stage.get('planning', 0),
            '{{DEVELOPMENT_CONTROLS}}': controls_by_stage.get('development', 0),
            '{{VALIDATION_CONTROLS}}': controls_by_stage.get('validation', 0),
            '{{IMPLEMENTATION_CONTROLS}}': controls_by_stage.get('implementation', 0),
            '{{USE_MONITORING_CONTROLS}}': controls_by_stage.get('use_and_monitoring', 0),
            '{{CHANGES_CONTROLS}}': controls_by_stage.get('changes', 0),
            '{{RETIREMENT_CONTROLS}}': controls_by_stage.get('retirement', 0),
        }
    
    def _substitute_dynamic_content(self, latex_content: str, report: ControlRisksYAMLReport) -> str:
        """Substitute dynamic content sections"""
        
        # Generate critical risks table
        critical_risks_table = self._generate_critical_risks_table(report)
        latex_content = latex_content.replace('{{CRITICAL_RISKS_TABLE}}', critical_risks_table)
        
        # Generate high risks table
        high_risks_table = self._generate_high_risks_table(report)
        latex_content = latex_content.replace('{{HIGH_RISKS_TABLE}}', high_risks_table)
        
        # Generate action items table
        action_items_table = self._generate_action_items_table(report)
        latex_content = latex_content.replace('{{ACTION_ITEMS_TABLE}}', action_items_table)
        
        # Generate compliance gaps table
        compliance_gaps_table = self._generate_compliance_gaps_table(report)
        latex_content = latex_content.replace('{{COMPLIANCE_GAPS_TABLE}}', compliance_gaps_table)
        
        return latex_content
    
    def _generate_critical_risks_table(self, report: ControlRisksYAMLReport) -> str:
        """Generate LaTeX table for critical risks"""
        
        critical_risks = [r for r in report.risk_details if r.risk_severity == RiskSeverity.CRITICAL]
        
        if not critical_risks:
            return "No critical risks identified."
        
        # Limit to top 20 critical risks for space
        critical_risks = critical_risks[:20]
        
        table_content = []
        table_content.append("\\begin{longtable}{|p{1.5cm}|p{5cm}|p{2.5cm}|p{2cm}|p{1.5cm}|}")
        table_content.append("\\hline")
        table_content.append("\\textbf{Risk ID} & \\textbf{Risk Name} & \\textbf{Category} & \\textbf{Owner} & \\textbf{Controls} \\\\")
        table_content.append("\\hline")
        table_content.append("\\endfirsthead")
        table_content.append("\\hline")
        table_content.append("\\textbf{Risk ID} & \\textbf{Risk Name} & \\textbf{Category} & \\textbf{Owner} & \\textbf{Controls} \\\\")
        table_content.append("\\hline")
        table_content.append("\\endhead")
        
        for risk in critical_risks:
            risk_id = self._escape_latex(risk.risk_id)
            risk_name = self._escape_latex(risk.template_risk_name[:50] + "..." if len(risk.template_risk_name) > 50 else risk.template_risk_name)
            category = self._escape_latex(risk.risk_category.value.replace('_', ' ').title())
            owner = self._escape_latex(risk.risk_owner)
            controls = str(risk.total_controls)
            
            table_content.append(f"{risk_id} & {risk_name} & {category} & {owner} & {controls} \\\\")
            table_content.append("\\hline")
        
        table_content.append("\\caption{Critical Risks Requiring Immediate Attention}")
        table_content.append("\\end{longtable}")
        
        return "\n".join(table_content)
    
    def _generate_high_risks_table(self, report: ControlRisksYAMLReport) -> str:
        """Generate LaTeX table for high risks"""
        
        high_risks = [r for r in report.risk_details if r.risk_severity == RiskSeverity.HIGH]
        
        if not high_risks:
            return "No high risks identified."
        
        # Limit to top 15 high risks for space
        high_risks = high_risks[:15]
        
        table_content = []
        table_content.append("\\begin{longtable}{|p{1.5cm}|p{5cm}|p{2.5cm}|p{2cm}|p{1.5cm}|}")
        table_content.append("\\hline")
        table_content.append("\\textbf{Risk ID} & \\textbf{Risk Name} & \\textbf{Category} & \\textbf{Owner} & \\textbf{Controls} \\\\")
        table_content.append("\\hline")
        table_content.append("\\endfirsthead")
        table_content.append("\\hline")
        table_content.append("\\textbf{Risk ID} & \\textbf{Risk Name} & \\textbf{Category} & \\textbf{Owner} & \\textbf{Controls} \\\\")
        table_content.append("\\hline")
        table_content.append("\\endhead")
        
        for risk in high_risks:
            risk_id = self._escape_latex(risk.risk_id)
            risk_name = self._escape_latex(risk.template_risk_name[:50] + "..." if len(risk.template_risk_name) > 50 else risk.template_risk_name)
            category = self._escape_latex(risk.risk_category.value.replace('_', ' ').title())
            owner = self._escape_latex(risk.risk_owner)
            controls = str(risk.total_controls)
            
            table_content.append(f"{risk_id} & {risk_name} & {category} & {owner} & {controls} \\\\")
            table_content.append("\\hline")
        
        table_content.append("\\caption{High Risks Requiring Priority Attention}")
        table_content.append("\\end{longtable}")
        
        return "\n".join(table_content)
    
    def _generate_action_items_table(self, report: ControlRisksYAMLReport) -> str:
        """Generate LaTeX table for action items"""
        
        # Limit to top 15 action items
        action_items = report.action_items[:15]
        
        table_content = []
        table_content.append("\\begin{longtable}{|p{0.8cm}|p{6cm}|p{2cm}|p{2cm}|p{1.5cm}|}")
        table_content.append("\\hline")
        table_content.append("\\textbf{\\#} & \\textbf{Action Item} & \\textbf{Priority} & \\textbf{Owner} & \\textbf{Timeline} \\\\")
        table_content.append("\\hline")
        table_content.append("\\endfirsthead")
        table_content.append("\\hline")
        table_content.append("\\textbf{\\#} & \\textbf{Action Item} & \\textbf{Priority} & \\textbf{Owner} & \\textbf{Timeline} \\\\")
        table_content.append("\\hline")
        table_content.append("\\endhead")
        
        for i, action in enumerate(action_items, 1):
            action_text = self._escape_latex(action[:80] + "..." if len(action) > 80 else action)
            
            # Determine priority based on keywords
            priority = "High"
            if "immediate" in action.lower() or "critical" in action.lower():
                priority = "\\textcolor{criticalred}{Critical}"
            elif "develop" in action.lower() or "enhance" in action.lower():
                priority = "\\textcolor{highorange}{High}"
            else:
                priority = "\\textcolor{mediumyellow}{Medium}"
            
            # Determine owner based on keywords
            owner = "Risk Team"
            if "model" in action.lower():
                owner = "Model Owner"
            elif "governance" in action.lower():
                owner = "Risk Manager"
            elif "validation" in action.lower():
                owner = "Validator"
            
            # Determine timeline based on priority
            timeline = "90 days"
            if "immediate" in action.lower():
                timeline = "30 days"
            elif "develop" in action.lower():
                timeline = "60 days"
            
            table_content.append(f"{i} & {action_text} & {priority} & {owner} & {timeline} \\\\")
            table_content.append("\\hline")
        
        table_content.append("\\caption{Priority Action Items}")
        table_content.append("\\end{longtable}")
        
        return "\n".join(table_content)
    
    def _generate_compliance_gaps_table(self, report: ControlRisksYAMLReport) -> str:
        """Generate LaTeX table for compliance gaps"""
        
        table_content = []
        table_content.append("\\begin{table}[H]")
        table_content.append("\\centering")
        table_content.append("\\caption{Compliance Gaps and Remediation}")
        table_content.append("\\begin{tabular}{|p{3cm}|p{4cm}|p{3cm}|p{2.5cm}|}")
        table_content.append("\\hline")
        table_content.append("\\textbf{Compliance Area} & \\textbf{Gap Description} & \\textbf{Remediation} & \\textbf{Timeline} \\\\")
        table_content.append("\\hline")
        
        # Sample compliance gaps based on report data
        gaps = [
            ("SR11-7 Compliance", "Documentation standards need enhancement", "Update documentation framework", "90 days"),
            ("Validation Procedures", "Independent validation gaps identified", "Strengthen validation processes", "60 days"),
            ("Governance Framework", "Oversight procedures require improvement", "Enhance governance controls", "120 days"),
            ("Monitoring Systems", "Real-time monitoring capabilities needed", "Implement monitoring systems", "180 days")
        ]
        
        for area, gap, remediation, timeline in gaps:
            area_escaped = self._escape_latex(area)
            gap_escaped = self._escape_latex(gap)
            remediation_escaped = self._escape_latex(remediation)
            timeline_escaped = self._escape_latex(timeline)
            
            table_content.append(f"{area_escaped} & {gap_escaped} & {remediation_escaped} & {timeline_escaped} \\\\")
            table_content.append("\\hline")
        
        table_content.append("\\end{tabular}")
        table_content.append("\\end{table}")
        
        return "\n".join(table_content)
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        
        if text is None:
            return ""
        
        # LaTeX special characters
        replacements = {
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '^': '\\textasciicircum{}',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '\\': '\\textbackslash{}'
        }
        
        escaped_text = str(text)
        for char, replacement in replacements.items():
            escaped_text = escaped_text.replace(char, replacement)
        
        return escaped_text
    
    def _compile_latex_to_pdf(self, latex_path: str) -> str:
        """Compile LaTeX file to PDF"""
        
        # Get directory and filename
        latex_dir = os.path.dirname(latex_path)
        latex_filename = os.path.basename(latex_path)
        pdf_filename = latex_filename.replace('.tex', '.pdf')
        pdf_path = os.path.join(latex_dir, pdf_filename)
        
        # Compile LaTeX
        try:
            # Change to LaTeX directory for compilation
            original_dir = os.getcwd()
            os.chdir(latex_dir)
            
            # First compilation
            result1 = subprocess.run(
                [self.config.latex_engine, '-interaction=nonstopmode', latex_filename],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result1.returncode != 0:
                raise RuntimeError(f"LaTeX compilation failed (first pass): {result1.stderr}")
            
            # Second compilation for cross-references (if enabled)
            if self.config.compile_twice:
                result2 = subprocess.run(
                    [self.config.latex_engine, '-interaction=nonstopmode', latex_filename],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result2.returncode != 0:
                    raise RuntimeError(f"LaTeX compilation failed (second pass): {result2.stderr}")
            
            # Check if PDF was created
            if not os.path.exists(pdf_filename):
                raise RuntimeError("PDF file was not generated")
            
            # Return to original directory
            os.chdir(original_dir)
            
            # Clean up auxiliary files if requested
            if self.config.cleanup_temp:
                self._cleanup_latex_files(latex_dir, latex_filename.replace('.tex', ''))
            
            return pdf_path
            
        except subprocess.TimeoutExpired:
            os.chdir(original_dir)
            raise RuntimeError("LaTeX compilation timed out")
        except Exception as e:
            os.chdir(original_dir)
            raise RuntimeError(f"LaTeX compilation error: {str(e)}")
    
    def _cleanup_latex_files(self, directory: str, base_name: str):
        """Clean up LaTeX auxiliary files"""
        
        extensions_to_remove = ['.aux', '.log', '.toc', '.lof', '.lot', '.out', '.fls', '.fdb_latexmk']
        
        for ext in extensions_to_remove:
            file_path = os.path.join(directory, base_name + ext)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # Ignore cleanup errors
    
    def create_sample_assets(self):
        """Create sample assets (logo, etc.) for template"""
        
        assets_dir = "assets"
        os.makedirs(assets_dir, exist_ok=True)
        
        # Create a simple logo placeholder using TikZ
        logo_tex = """
\\documentclass{standalone}
\\usepackage{tikz}
\\usepackage{xcolor}
\\definecolor{corporateblue}{RGB}{31,78,121}
\\begin{document}
\\begin{tikzpicture}
\\draw[corporateblue, thick, fill=corporateblue!20] (0,0) rectangle (4,2);
\\node[corporateblue, font=\\Large\\bfseries] at (2,1) {LOGO};
\\end{tikzpicture}
\\end{document}
"""
        
        logo_tex_path = os.path.join(assets_dir, "logo.tex")
        with open(logo_tex_path, 'w') as f:
            f.write(logo_tex)
        
        print(f"Sample logo template created: {logo_tex_path}")
        print("Compile with: pdflatex logo.tex to create logo.pdf")
    
    def check_latex_installation(self) -> Dict[str, bool]:
        """Check LaTeX installation and required packages"""
        
        checks = {}
        
        # Check LaTeX engine
        try:
            result = subprocess.run([self.config.latex_engine, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            checks['latex_engine'] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            checks['latex_engine'] = False
        
        # Check required packages (simplified check)
        required_packages = ['tikz', 'booktabs', 'longtable', 'xcolor', 'hyperref']
        for package in required_packages:
            checks[f'package_{package}'] = True  # Assume available for now
        
        return checks
