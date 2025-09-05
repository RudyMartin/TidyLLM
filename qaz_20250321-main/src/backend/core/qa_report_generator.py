#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA HealthCheck Report Generator

Generates comprehensive QA HealthCheck reports based on YAML criteria configuration
and document analysis results.
"""

import os
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import re
from dataclasses import dataclass, asdict


@dataclass
class ReviewMetadata:
    """Review metadata structure"""
    review_id: str
    model_type: str
    risk_tier: str
    model_id: str
    model_name: str
    version: str
    authors: List[str]
    date: str
    validation_type: str
    reviewer_name: str
    team_num: str
    process_name: str
    report_generated_at: str


@dataclass
class CriteriaResult:
    """Individual criteria result"""
    criterion_id: str
    criterion_text: str
    category_name: str
    weight: float
    score: float
    max_score: float
    pass_threshold: float
    status: str  # "pass", "fail", "conditional"
    evidence_found: List[str]
    evidence_count: int
    explanation: str
    recommendations: List[str]


@dataclass
class CategoryResult:
    """Category-level results"""
    category_id: str
    category_name: str
    weight: float
    total_score: float
    max_possible_score: float
    percentage_score: float
    criteria_results: List[CriteriaResult]
    status: str  # "excellent", "good", "satisfactory", "needs_improvement"


@dataclass
class QAReport:
    """Complete QA HealthCheck report"""
    metadata: ReviewMetadata
    categories: List[CategoryResult]
    overall_score: float
    overall_status: str
    total_criteria: int
    passed_criteria: int
    failed_criteria: int
    conditional_criteria: int
    recommendations: List[str]
    next_steps: List[str]
    quality_metrics: Dict[str, Any]


class QAReportGenerator:
    """Generate QA HealthCheck reports from YAML criteria and document analysis"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Try to find config file in common locations
            possible_paths = [
                "dev_configs/qa_criteria_full.yaml",
                "config/qa_criteria_full.yaml",
                "src/config/qa_criteria_full.yaml",
                "qa_criteria_full.yaml"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    config_path = path
                    break
            else:
                # Create a minimal default config
                config_path = "qa_criteria_full.yaml"
                self._create_default_config(config_path)
        
        self.config_path = Path(config_path)
        self.qa_config = self._load_qa_config()
        self.report_data = None
        
    def _create_default_config(self, config_path: str):
        """Create a default QA config file"""
        default_config = {
            'checklist_categories': [
                {
                    'name': 'Documentation',
                    'description': 'Documentation and metadata requirements',
                    'criteria': [
                        {
                            'id': 'DOC_001',
                            'name': 'Review ID Present',
                            'description': 'Document contains a valid review ID',
                            'weight': 1.0,
                            'required': True
                        }
                    ]
                },
                {
                    'name': 'Technical Analysis',
                    'description': 'Technical validation requirements',
                    'criteria': [
                        {
                            'id': 'TECH_001',
                            'name': 'Model Validation',
                            'description': 'Model validation procedures documented',
                            'weight': 1.0,
                            'required': True
                        }
                    ]
                }
            ]
        }
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            print(f"✅ Created default QA config at {config_path}")
        except Exception as e:
            print(f"❌ Error creating default config: {e}")
    
    def _load_qa_config(self) -> Dict[str, Any]:
        """Load QA criteria configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded QA config from {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading QA config: {e}")
            # Return minimal config if file can't be loaded
            return {
                'checklist_categories': [
                    {
                        'name': 'Basic Requirements',
                        'description': 'Basic document requirements',
                        'criteria': [
                            {
                                'id': 'BASIC_001',
                                'name': 'Document Present',
                                'description': 'Document is present and readable',
                                'weight': 1.0,
                                'required': True
                            }
                        ]
                    }
                ]
            }
    
    def _extract_review_metadata(self, documents: List[Dict[str, Any]], 
                                extracted_fields: Dict[str, Any]) -> ReviewMetadata:
        """Extract review metadata from documents and extracted fields"""
        
        # Default values
        review_id = "REV00000"
        model_type = "Unknown"
        risk_tier = "Medium"
        model_id = "Unknown"
        model_name = "Unknown Model"
        version = "1.0.0"
        authors = ["Unknown"]
        date = datetime.now().strftime("%m-%d-%Y")
        validation_type = "Standard Review"
        
        # Search for Review ID in documents
        for doc in documents:
            if 'content' in doc:
                content = doc['content']
                # Look for Review ID patterns
                review_id_patterns = [
                    r'Review ID[:\s]*([A-Z]{3}\d{5})',
                    r'Review[:\s]*([A-Z]{3}\d{5})',
                    r'([A-Z]{3}\d{5})',
                    r'Review ID[:\s]*(\d{5})',
                ]
                
                for pattern in review_id_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        review_id = match.group(1)
                        if not review_id.startswith('REV'):
                            review_id = f"REV{review_id.zfill(5)}"
                        break
        
        # Override with extracted fields if available
        if extracted_fields:
            review_id = extracted_fields.get('review_id', review_id)
            model_type = extracted_fields.get('model_type', model_type)
            risk_tier = extracted_fields.get('risk_tier', risk_tier)
            model_id = extracted_fields.get('model_id', model_id)
            model_name = extracted_fields.get('model_name', model_name)
            version = extracted_fields.get('version', version)
            authors = extracted_fields.get('authors', authors)
            if isinstance(authors, str):
                authors = [authors]
            date = extracted_fields.get('date', date)
            validation_type = extracted_fields.get('validation_type', validation_type)
        
        return ReviewMetadata(
            review_id=review_id,
            model_type=model_type,
            risk_tier=risk_tier,
            model_id=model_id,
            model_name=model_name,
            version=version,
            authors=authors,
            date=date,
            validation_type=validation_type,
            reviewer_name=extracted_fields.get('reviewer_name', 'Unknown'),
            team_num=extracted_fields.get('team_num', 'Unknown'),
            process_name=extracted_fields.get('process_name', 'QA Validation Review'),
            report_generated_at=datetime.now().isoformat()
        )
    
    def _analyze_documents_against_criteria(self, documents: List[Dict[str, Any]]) -> List[CategoryResult]:
        """Analyze documents against QA criteria from YAML"""
        
        categories = []
        
        for category_config in self.qa_config['checklist_categories']:
            category_results = []
            
            for criterion_config in category_config['criteria']:
                # Analyze this criterion against documents
                criterion_result = self._analyze_single_criterion(
                    criterion_config, documents
                )
                category_results.append(criterion_result)
            
            # Calculate category-level results
            total_score = sum(cr.score for cr in category_results)
            max_possible_score = sum(cr.max_score for cr in category_results)
            percentage_score = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
            
            # Determine category status
            thresholds = self.qa_config['scoring_rules']['thresholds']
            if percentage_score >= thresholds['excellent']:
                status = "excellent"
            elif percentage_score >= thresholds['good']:
                status = "good"
            elif percentage_score >= thresholds['satisfactory']:
                status = "satisfactory"
            else:
                status = "needs_improvement"
            
            category_result = CategoryResult(
                category_id=category_config['id'],
                category_name=category_config['name'],
                weight=category_config['weight'],
                total_score=total_score,
                max_possible_score=max_possible_score,
                percentage_score=percentage_score,
                criteria_results=category_results,
                status=status
            )
            
            categories.append(category_result)
        
        return categories
    
    def _analyze_single_criterion(self, criterion_config: Dict[str, Any], 
                                 documents: List[Dict[str, Any]]) -> CriteriaResult:
        """Analyze a single criterion against documents"""
        
        criterion_id = criterion_config['id']
        criterion_text = criterion_config['text']
        evidence_found = []
        explanation = ""
        recommendations = []
        
        # Search for evidence in documents
        for doc in documents:
            if 'content' in doc:
                content = doc['content'].lower()
                filename = doc.get('filename', 'Unknown')
                
                # Look for keywords related to this criterion
                keywords = self._extract_keywords_from_criterion(criterion_text)
                
                for keyword in keywords:
                    if keyword.lower() in content:
                        evidence_found.append(f"Found '{keyword}' in {filename}")
                
                # Check for specific evidence types
                evidence_types = criterion_config.get('evidence_types', [])
                for evidence_type in evidence_types:
                    if evidence_type.lower() in filename.lower():
                        evidence_found.append(f"Evidence file type {evidence_type} found: {filename}")
        
        # Determine score based on evidence
        evidence_count = len(evidence_found)
        max_score = 100
        pass_threshold = criterion_config.get('pass_threshold', 70)
        
        if evidence_count >= 3:
            score = max_score
            status = "pass"
            explanation = f"Strong evidence found with {evidence_count} supporting documents"
        elif evidence_count >= 1:
            score = pass_threshold
            status = "conditional"
            explanation = f"Limited evidence found with {evidence_count} supporting document(s)"
        else:
            score = 0
            status = "fail"
            explanation = f"No evidence found for this criterion"
        
        # Generate recommendations
        if status == "fail":
            recommendations = [
                f"Document {criterion_text.lower()}",
                f"Provide evidence for {criterion_text.lower()}",
                f"Include supporting documentation for {criterion_text.lower()}"
            ]
        elif status == "conditional":
            recommendations = [
                f"Strengthen documentation for {criterion_text.lower()}",
                f"Provide additional evidence for {criterion_text.lower()}"
            ]
        
        return CriteriaResult(
            criterion_id=criterion_id,
            criterion_text=criterion_text,
            category_name="",  # Will be set by parent
            weight=criterion_config.get('weight', 1.0),
            score=score,
            max_score=max_score,
            pass_threshold=pass_threshold,
            status=status,
            evidence_found=evidence_found,
            evidence_count=evidence_count,
            explanation=explanation,
            recommendations=recommendations
        )
    
    def _extract_keywords_from_criterion(self, criterion_text: str) -> List[str]:
        """Extract relevant keywords from criterion text for document search"""
        # Simple keyword extraction - could be enhanced with NLP
        keywords = []
        
        # Common QA-related keywords
        qa_keywords = [
            'validation', 'documentation', 'testing', 'review', 'approval',
            'governance', 'compliance', 'data quality', 'methodology',
            'calibration', 'benchmarking', 'stress testing', 'monitoring',
            'limitations', 'assumptions', 'evidence', 'procedures'
        ]
        
        # Extract words from criterion text
        words = re.findall(r'\b\w+\b', criterion_text.lower())
        keywords.extend(words)
        
        # Add common QA keywords
        keywords.extend(qa_keywords)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 3]
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_overall_results(self, categories: List[CategoryResult]) -> Dict[str, Any]:
        """Calculate overall report results"""
        
        total_criteria = sum(len(cat.criteria_results) for cat in categories)
        passed_criteria = sum(1 for cat in categories 
                            for cr in cat.criteria_results if cr.status == "pass")
        failed_criteria = sum(1 for cat in categories 
                            for cr in cat.criteria_results if cr.status == "fail")
        conditional_criteria = sum(1 for cat in categories 
                                 for cr in cat.criteria_results if cr.status == "conditional")
        
        # Calculate weighted overall score
        total_weighted_score = 0
        total_weight = 0
        
        for category in categories:
            category_score = category.percentage_score / 100  # Convert to 0-1
            total_weighted_score += category_score * category.weight
            total_weight += category.weight
        
        overall_score = (total_weighted_score / total_weight * 100) if total_weight > 0 else 0
        
        # Determine overall status
        thresholds = self.qa_config['scoring_rules']['thresholds']
        if overall_score >= thresholds['excellent']:
            overall_status = "excellent"
        elif overall_score >= thresholds['good']:
            overall_status = "good"
        elif overall_score >= thresholds['satisfactory']:
            overall_status = "satisfactory"
        else:
            overall_status = "needs_improvement"
        
        return {
            'overall_score': overall_score,
            'overall_status': overall_status,
            'total_criteria': total_criteria,
            'passed_criteria': passed_criteria,
            'failed_criteria': failed_criteria,
            'conditional_criteria': conditional_criteria
        }
    
    def _generate_recommendations(self, categories: List[CategoryResult]) -> List[str]:
        """Generate overall recommendations based on analysis"""
        recommendations = []
        
        # Collect all failed and conditional criteria
        failed_criteria = []
        conditional_criteria = []
        
        for category in categories:
            for criterion in category.criteria_results:
                if criterion.status == "fail":
                    failed_criteria.append(criterion)
                elif criterion.status == "conditional":
                    conditional_criteria.append(criterion)
        
        # Generate recommendations
        if failed_criteria:
            recommendations.append(f"Address {len(failed_criteria)} failed criteria to improve overall score")
            for criterion in failed_criteria[:3]:  # Top 3 most critical
                recommendations.append(f"Priority: {criterion.criterion_text}")
        
        if conditional_criteria:
            recommendations.append(f"Strengthen {len(conditional_criteria)} criteria with limited evidence")
        
        # Category-specific recommendations
        for category in categories:
            if category.status in ["needs_improvement", "satisfactory"]:
                recommendations.append(f"Focus on improving {category.category_name} (current score: {category.percentage_score:.1f}%)")
        
        return recommendations
    
    def _generate_next_steps(self, overall_status: str, failed_criteria: int) -> List[str]:
        """Generate next steps based on overall status"""
        next_steps = []
        
        if overall_status == "excellent":
            next_steps = [
                "Schedule follow-up review in 6 months",
                "Document best practices for future reference",
                "Consider expanding validation scope"
            ]
        elif overall_status == "good":
            next_steps = [
                "Address remaining conditional criteria",
                "Schedule follow-up review in 3 months",
                "Implement recommended improvements"
            ]
        elif overall_status == "satisfactory":
            next_steps = [
                "Address all failed criteria within 30 days",
                "Strengthen evidence for conditional criteria",
                "Schedule follow-up review in 1 month",
                "Consider additional validation testing"
            ]
        else:  # needs_improvement
            next_steps = [
                "Immediate action required on all failed criteria",
                "Schedule emergency review within 1 week",
                "Consider model suspension until issues resolved",
                "Implement comprehensive remediation plan"
            ]
        
        return next_steps
    
    def generate_report(self, documents: List[Dict[str, Any]], 
                       extracted_fields: Dict[str, Any]) -> QAReport:
        """Generate complete QA HealthCheck report"""
        
        # Extract metadata
        metadata = self._extract_review_metadata(documents, extracted_fields)
        
        # Analyze documents against criteria
        categories = self._analyze_documents_against_criteria(documents)
        
        # Calculate overall results
        overall_results = self._calculate_overall_results(categories)
        
        # Generate recommendations and next steps
        recommendations = self._generate_recommendations(categories)
        failed_count = overall_results['failed_criteria']
        next_steps = self._generate_next_steps(overall_results['overall_status'], failed_count)
        
        # Calculate quality metrics
        quality_metrics = {
            'evidence_coverage': (overall_results['passed_criteria'] + overall_results['conditional_criteria']) / overall_results['total_criteria'] * 100,
            'average_evidence_per_criterion': sum(sum(len(cr.evidence_found) for cr in cat.criteria_results) for cat in categories) / overall_results['total_criteria'],
            'categories_above_threshold': sum(1 for cat in categories if cat.status in ['excellent', 'good']),
            'critical_issues': overall_results['failed_criteria']
        }
        
        # Create QA report
        qa_report = QAReport(
            metadata=metadata,
            categories=categories,
            overall_score=overall_results['overall_score'],
            overall_status=overall_results['overall_status'],
            total_criteria=overall_results['total_criteria'],
            passed_criteria=overall_results['passed_criteria'],
            failed_criteria=overall_results['failed_criteria'],
            conditional_criteria=overall_results['conditional_criteria'],
            recommendations=recommendations,
            next_steps=next_steps,
            quality_metrics=quality_metrics
        )
        
        self.report_data = qa_report
        return qa_report
    
    def generate_json_report(self, output_path: str = None) -> str:
        """Generate JSON version of the report"""
        if not self.report_data:
            raise ValueError("No report data available. Call generate_report() first.")
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/output/qa_healthcheck_report_{timestamp}.json"
        
        # Convert dataclass to dict
        report_dict = asdict(self.report_data)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"✅ JSON report saved to: {output_path}")
        return output_path
    
    def generate_latex_report(self, output_path: str = None) -> str:
        """Generate LaTeX version of the report"""
        if not self.report_data:
            raise ValueError("No report data available. Call generate_report() first.")
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/output/qa_healthcheck_report_{timestamp}.tex"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        latex_content = self._generate_latex_content()
        
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        print(f"✅ LaTeX report saved to: {output_path}")
        return output_path
    
    def _generate_latex_content(self) -> str:
        """Generate LaTeX content for the report"""
        
        # LaTeX template
        latex_template = r"""
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

% Custom commands
\newcommand{\statusbox}[2]{%
    \fboxsep=0pt\fboxrule=0.4pt%
    \colorbox{#1}{\textcolor{white}{\textbf{#2}}}%
}

\newcommand{\criteriaTable}[1]{%
    \begin{longtable}{p{0.4\textwidth}p{0.15\textwidth}p{0.15\textwidth}p{0.3\textwidth}}
    \toprule
    \textbf{Criterion} & \textbf{Score} & \textbf{Status} & \textbf{Explanation} \\
    \midrule
    #1
    \bottomrule
    \end{longtable}%
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

\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Field} & \textbf{Value} \\
\midrule
Review ID & REVIEW_ID_PLACEHOLDER \\
Model Type & MODEL_TYPE_PLACEHOLDER \\
Risk Tier & RISK_TIER_PLACEHOLDER \\
Model ID & MODEL_ID_PLACEHOLDER \\
Model Name & MODEL_NAME_PLACEHOLDER \\
Version & VERSION_PLACEHOLDER \\
Authors & AUTHORS_PLACEHOLDER \\
Date & DATE_PLACEHOLDER \\
Validation Type & VALIDATION_TYPE_PLACEHOLDER \\
Reviewer & REVIEWER_NAME_PLACEHOLDER \\
Team & TEAM_NUM_PLACEHOLDER \\
Process & PROCESS_NAME_PLACEHOLDER \\
\bottomrule
\end{tabular}
\end{table}

% Criteria Analysis
\section{Criteria Analysis}

CRITERIA_SECTIONS_PLACEHOLDER

% Recommendations
\section{Recommendations}

RECOMMENDATIONS_PLACEHOLDER

% Next Steps
\section{Next Steps}

NEXT_STEPS_PLACEHOLDER

\end{document}
"""
        
        # Replace placeholders with actual data
        content = latex_template
        
        # Basic metadata replacements
        content = content.replace("REVIEW_ID_PLACEHOLDER", self.report_data.metadata.review_id)
        content = content.replace("MODEL_NAME_PLACEHOLDER", self.report_data.metadata.model_name)
        content = content.replace("RISK_TIER_PLACEHOLDER", self.report_data.metadata.risk_tier)
        content = content.replace("DATE_PLACEHOLDER", self.report_data.metadata.report_generated_at[:10])
        content = content.replace("OVERALL_STATUS_PLACEHOLDER", self.report_data.overall_status.title())
        content = content.replace("OVERALL_SCORE_PLACEHOLDER", f"{self.report_data.overall_score:.1f}")
        
        # Summary statistics
        content = content.replace("TOTAL_CRITERIA_PLACEHOLDER", str(self.report_data.total_criteria))
        content = content.replace("PASSED_CRITERIA_PLACEHOLDER", str(self.report_data.passed_criteria))
        content = content.replace("FAILED_CRITERIA_PLACEHOLDER", str(self.report_data.failed_criteria))
        content = content.replace("CONDITIONAL_CRITERIA_PLACEHOLDER", str(self.report_data.conditional_criteria))
        
        # Quality metrics
        content = content.replace("EVIDENCE_COVERAGE_PLACEHOLDER", f"{self.report_data.quality_metrics['evidence_coverage']:.1f}")
        content = content.replace("AVG_EVIDENCE_PLACEHOLDER", f"{self.report_data.quality_metrics['average_evidence_per_criterion']:.1f}")
        content = content.replace("CATEGORIES_ABOVE_PLACEHOLDER", str(self.report_data.quality_metrics['categories_above_threshold']))
        content = content.replace("CRITICAL_ISSUES_PLACEHOLDER", str(self.report_data.quality_metrics['critical_issues']))
        
        # Metadata table
        content = content.replace("MODEL_TYPE_PLACEHOLDER", self.report_data.metadata.model_type)
        content = content.replace("MODEL_ID_PLACEHOLDER", self.report_data.metadata.model_id)
        content = content.replace("VERSION_PLACEHOLDER", self.report_data.metadata.version)
        content = content.replace("AUTHORS_PLACEHOLDER", ", ".join(self.report_data.metadata.authors))
        content = content.replace("VALIDATION_TYPE_PLACEHOLDER", self.report_data.metadata.validation_type)
        content = content.replace("REVIEWER_NAME_PLACEHOLDER", self.report_data.metadata.reviewer_name)
        content = content.replace("TEAM_NUM_PLACEHOLDER", self.report_data.metadata.team_num)
        content = content.replace("PROCESS_NAME_PLACEHOLDER", self.report_data.metadata.process_name)
        
        # Generate criteria sections
        criteria_sections = ""
        for category in self.report_data.categories:
            criteria_sections += f"\\subsection{{{category.category_name}}}\n"
            criteria_sections += f"\\textbf{{Category Score: {category.percentage_score:.1f}\\%}} \\\\\n"
            criteria_sections += f"\\textbf{{Status: {category.status.title()}}} \\\\\n\n"
            
            # Create criteria table
            table_rows = ""
            for criterion in category.criteria_results:
                # Determine status color
                if criterion.status == "pass":
                    status_color = "excellent"
                elif criterion.status == "conditional":
                    status_color = "satisfactory"
                else:
                    status_color = "needsimprovement"
                
                table_rows += f"{criterion.criterion_text} & {criterion.score:.0f} & \\statusbox{{{status_color}}}{{{criterion.status.title()}}} & {criterion.explanation} \\\\\n"
            
            criteria_sections += f"\\criteriaTable{{{table_rows}}}\n\n"
        
        content = content.replace("CRITERIA_SECTIONS_PLACEHOLDER", criteria_sections)
        
        # Generate recommendations
        recommendations_text = ""
        for i, rec in enumerate(self.report_data.recommendations, 1):
            recommendations_text += f"\\item {rec}\n"
        
        content = content.replace("RECOMMENDATIONS_PLACEHOLDER", recommendations_text)
        
        # Generate next steps
        next_steps_text = ""
        for i, step in enumerate(self.report_data.next_steps, 1):
            next_steps_text += f"\\item {step}\n"
        
        content = content.replace("NEXT_STEPS_PLACEHOLDER", next_steps_text)
        
        return content


def main():
    """Test the QA report generator"""
    
    # Create test documents
    test_documents = [
        {
            'filename': 'test_document.pdf',
            'content': '''
            Review ID: REV12345
            Model Type: Machine Learning
            Risk Tier: High
            Model ID: ML-2024-001
            Model Name: Credit Risk Model v2.1
            Version: 2.1.0
            Authors: Dr. Smith, Dr. Johnson
            Date: 08-21-2024
            Validation Type: Comprehensive Review
            
            This document contains validation methodology and data quality metrics.
            The model has been thoroughly tested with backtesting and stress testing.
            Governance approval has been obtained from the risk committee.
            '''
        }
    ]
    
    # Create test extracted fields
    test_extracted_fields = {
        'review_id': 'REV12345',
        'model_type': 'Machine Learning',
        'risk_tier': 'High',
        'model_id': 'ML-2024-001',
        'model_name': 'Credit Risk Model v2.1',
        'version': '2.1.0',
        'authors': ['Dr. Smith', 'Dr. Johnson'],
        'date': '08-21-2024',
        'validation_type': 'Comprehensive Review',
        'reviewer_name': 'Alex',
        'team_num': 'QA Team 1',
        'process_name': 'QA Validation Review'
    }
    
    # Generate report
    generator = QAReportGenerator()
    report = generator.generate_report(test_documents, test_extracted_fields)
    
    # Generate output files
    json_path = generator.generate_json_report()
    latex_path = generator.generate_latex_report()
    
    print(f"✅ Report generated successfully!")
    print(f"📄 JSON: {json_path}")
    print(f"📄 LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
