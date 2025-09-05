#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input Folder DataMart EnhancedQAOrchestrator Test - Non-Standard PDFs

This script tests the EnhancedQAOrchestrator against non-standard PDFs from the
input folder, using DataMart to track processing and show metrics.

Features:
- Focused testing on input folder PDFs
- DataMart integration for processing tracking
- Non-standard PDF handling capability testing
- Real-time DataMart metrics and values display
- Processing history and analytics
"""

import sys
import os
import json
import time
import uuid
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Import the standalone orchestrator
import importlib.util
spec = importlib.util.spec_from_file_location("enhanced_demo", "03_enhanced_qa_orchestrator_demo.py")
enhanced_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_demo)
StandaloneEnhancedQAOrchestrator = enhanced_demo.StandaloneEnhancedQAOrchestrator


class DataMart:
    """DataMart for tracking processing history and metrics with Enhanced QA integration"""
    
    def __init__(self):
        self.datamart_id = str(uuid.uuid4())
        self.processing_history = []
        self.knowledge_base_documents = self._load_knowledge_base_index()
        self.previously_processed = self._load_processing_history()
        
        self.metrics = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'pdf_processed': 0,
            'non_standard_pdfs': 0,
            'new_embeddings': 0,
            'existing_embeddings': 0,
            'in_knowledge_base': 0,
            'not_in_knowledge_base': 0,
            'processing_times': [],
            'quality_scores': [],
            'file_sizes': [],
            'error_types': defaultdict(int),
            'content_analysis': {
                'toc_found': 0,
                'links_found': 0,
                'references_found': 0,
                'options_found': 0,
                'captions_found': 0
            },
            'mvr_analysis': {
                'mvr_documents_found': 0,
                'peer_review_ready': 0,
                'mvs_compliant': 0,
                'vst_scope_identified': 0,
                'peer_review_challenges_generated': 0,
                'high_confidence_assessments': 0
            },
            'qa_criteria_compliance': {
                'full_criteria_analyzed': 0,
                'simplified_criteria_analyzed': 0,
                'regulatory_compliance_checked': 0,
                'workflow_readiness_assessed': 0,
                'express_mode_ready': 0,
                'high_compliance_documents': 0
            }
        }
        self.current_session = {
            'session_id': str(uuid.uuid4()),
            'start_time': datetime.now(),
            'documents_processed': []
        }
    
    def _load_knowledge_base_index(self) -> set:
        """Load index of documents in knowledge base"""
        kb_docs = set()
        try:
            kb_path = Path(__file__).parent.parent.parent / "knowledge_base"
            if kb_path.exists():
                for pdf_file in kb_path.rglob("*.pdf"):
                    kb_docs.add(pdf_file.name.lower())
                print(f"📚 Loaded {len(kb_docs)} documents from knowledge base index")
        except Exception as e:
            print(f"⚠️ Error loading knowledge base index: {e}")
        return kb_docs
    
    def _load_processing_history(self) -> set:
        """Load previously processed documents (simulated)"""
        # In a real system, this would load from a database or cache
        # For now, we'll simulate with some common patterns
        processed_docs = set()
        try:
            # Check for existing embedding files or processing logs
            cache_path = Path(__file__).parent.parent.parent / "data" / "cache"
            if cache_path.exists():
                for cache_file in cache_path.rglob("*.json"):
                    if "embedding" in cache_file.name or "processed" in cache_file.name:
                        # Extract document name from cache file
                        doc_name = cache_file.stem.replace("_embedding", "").replace("_processed", "")
                        processed_docs.add(doc_name.lower())
            print(f"🔄 Found {len(processed_docs)} previously processed documents")
        except Exception as e:
            print(f"⚠️ Error loading processing history: {e}")
        return processed_docs
        
    def add_processing_record(self, file_path: str, file_type: str, file_size: int, 
                            processing_time: float, success: bool, quality_score: float = 0.0, 
                            error: str = None, is_non_standard: bool = False, 
                            enhanced_result: Dict[str, Any] = None, mvr_analysis: Dict[str, Any] = None,
                            qa_compliance_analysis: Dict[str, Any] = None):
        """Add a processing record to DataMart with detailed Enhanced QA analysis"""
        
        # Check knowledge base and processing history
        filename = Path(file_path).name.lower()
        in_knowledge_base = filename in self.knowledge_base_documents
        previously_processed = filename in self.previously_processed
        
        # Extract detailed content analysis from Enhanced QA result
        content_analysis = self._extract_content_analysis(enhanced_result)
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'file_path': file_path,
            'file_type': file_type,
            'file_size': file_size,
            'processing_time': processing_time,
            'success': success,
            'quality_score': quality_score,
            'error': error,
            'is_non_standard': is_non_standard,
            'in_knowledge_base': in_knowledge_base,
            'previously_processed': previously_processed,
            'new_embedding': not previously_processed,
            'content_analysis': content_analysis,
            'enhanced_qa_components': self._extract_enhanced_qa_components(enhanced_result),
            'mvr_analysis': mvr_analysis,
            'qa_compliance_analysis': qa_compliance_analysis
        }
        
        self.processing_history.append(record)
        self.current_session['documents_processed'].append(record)
        
        # Update metrics
        self.metrics['total_processed'] += 1
        if success:
            self.metrics['successful'] += 1
            self.metrics['processing_times'].append(processing_time)
            self.metrics['quality_scores'].append(quality_score)
        else:
            self.metrics['failed'] += 1
            if error:
                self.metrics['error_types'][error] += 1
        
        self.metrics['file_sizes'].append(file_size)
        
        if file_type.lower() == 'pdf':
            self.metrics['pdf_processed'] += 1
            if is_non_standard:
                self.metrics['non_standard_pdfs'] += 1
        
        # Update knowledge base and processing metrics
        if in_knowledge_base:
            self.metrics['in_knowledge_base'] += 1
        else:
            self.metrics['not_in_knowledge_base'] += 1
            
        if previously_processed:
            self.metrics['existing_embeddings'] += 1
        else:
            self.metrics['new_embeddings'] += 1
        
        # Update content analysis metrics
        if content_analysis['toc_found']:
            self.metrics['content_analysis']['toc_found'] += 1
        if content_analysis['links_found']:
            self.metrics['content_analysis']['links_found'] += 1
        if content_analysis['references_found']:
            self.metrics['content_analysis']['references_found'] += 1
        if content_analysis['options_found']:
            self.metrics['content_analysis']['options_found'] += 1
        if content_analysis['captions_found']:
            self.metrics['content_analysis']['captions_found'] += 1
    
    def _extract_content_analysis(self, enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed content analysis from Enhanced QA result"""
        if not enhanced_result or enhanced_result.get('status') != 'success':
            return {
                'toc_found': False,
                'links_found': False,
                'references_found': False,
                'options_found': False,
                'captions_found': False,
                'toc_sample': None,
                'link_count': 0,
                'reference_count': 0,
                'caption_count': 0,
                'sections_analyzed': 0
            }
        
        try:
            enhanced_report = enhanced_result.get('enhanced_report', {})
            document_inspection = enhanced_result.get('document_inspection', {})
            caption_analysis = enhanced_result.get('caption_analysis', {})
            
            # Extract TOC information
            toc_info = document_inspection.get('toc_analysis', {})
            toc_found = toc_info.get('score', 0) > 0.5
            toc_sample = toc_info.get('headings', [])[:3] if toc_found else None
            
            # Extract link information
            link_info = document_inspection.get('link_analysis', {})
            links_found = link_info.get('score', 0) > 0.1
            link_count = link_info.get('total_links', 0)
            
            # Extract reference information
            ref_info = document_inspection.get('bibliography_analysis', {})
            references_found = ref_info.get('score', 0) > 0.5
            reference_count = ref_info.get('references_found', 0)
            
            # Extract caption information
            caption_found = caption_analysis.get('total_captions', 0) > 0
            caption_count = caption_analysis.get('total_captions', 0)
            
            # Look for options/choices in content
            content = enhanced_result.get('content', '')
            options_found = any(keyword in content.lower() for keyword in ['option', 'choice', 'select', 'choose'])
            
            # Count sections analyzed
            structure_info = document_inspection.get('structure_analysis', {})
            sections_analyzed = structure_info.get('headings', 0)
            
            return {
                'toc_found': toc_found,
                'links_found': links_found,
                'references_found': references_found,
                'options_found': options_found,
                'captions_found': caption_found,
                'toc_sample': toc_sample,
                'link_count': link_count,
                'reference_count': reference_count,
                'caption_count': caption_count,
                'sections_analyzed': sections_analyzed
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting content analysis: {e}")
            return {
                'toc_found': False,
                'links_found': False,
                'references_found': False,
                'options_found': False,
                'captions_found': False,
                'toc_sample': None,
                'link_count': 0,
                'reference_count': 0,
                'caption_count': 0,
                'sections_analyzed': 0
            }
    
    def _extract_enhanced_qa_components(self, enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Enhanced QA component scores and details"""
        if not enhanced_result or enhanced_result.get('status') != 'success':
            return {
                'simple_qa_score': 0.0,
                'inspection_score': 0.0,
                'caption_score': 0.0,
                'prediction_score': 0.0,
                'enhanced_quality_score': 0.0,
                'component_breakdown': {}
            }
        
        try:
            enhanced_report = enhanced_result.get('enhanced_report', {})
            
            return {
                'simple_qa_score': enhanced_report.get('simple_qa_score', 0.0),
                'inspection_score': enhanced_report.get('inspection_score', 0.0),
                'caption_score': enhanced_report.get('caption_score', 0.0),
                'prediction_score': enhanced_report.get('prediction_score', 0.0),
                'enhanced_quality_score': enhanced_report.get('enhanced_quality_score', 0.0),
                'component_breakdown': {
                    'document_inspection': enhanced_result.get('document_inspection', {}),
                    'caption_analysis': enhanced_result.get('caption_analysis', {}),
                    'quality_prediction': enhanced_result.get('quality_prediction', {})
                }
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting Enhanced QA components: {e}")
            return {
                'simple_qa_score': 0.0,
                'inspection_score': 0.0,
                'caption_score': 0.0,
                'prediction_score': 0.0,
                'enhanced_quality_score': 0.0,
                'component_breakdown': {}
            }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current DataMart metrics"""
        metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics['processing_times']:
            metrics['avg_processing_time'] = sum(metrics['processing_times']) / len(metrics['processing_times'])
            metrics['min_processing_time'] = min(metrics['processing_times'])
            metrics['max_processing_time'] = max(metrics['processing_times'])
        else:
            metrics['avg_processing_time'] = 0
            metrics['min_processing_time'] = 0
            metrics['max_processing_time'] = 0
            
        if metrics['quality_scores']:
            metrics['avg_quality_score'] = sum(metrics['quality_scores']) / len(metrics['quality_scores'])
            metrics['min_quality_score'] = min(metrics['quality_scores'])
            metrics['max_quality_score'] = max(metrics['quality_scores'])
        else:
            metrics['avg_quality_score'] = 0
            metrics['min_quality_score'] = 0
            metrics['max_quality_score'] = 0
            
        if metrics['file_sizes']:
            metrics['avg_file_size'] = sum(metrics['file_sizes']) / len(metrics['file_sizes'])
            metrics['total_size_processed'] = sum(metrics['file_sizes'])
        else:
            metrics['avg_file_size'] = 0
            metrics['total_size_processed'] = 0
            
        metrics['success_rate'] = metrics['successful'] / metrics['total_processed'] if metrics['total_processed'] > 0 else 0
        metrics['error_rate'] = metrics['failed'] / metrics['total_processed'] if metrics['total_processed'] > 0 else 0
        
        return metrics
    
    def print_current_metrics(self):
        """Print current DataMart metrics with detailed Enhanced QA analysis"""
        metrics = self.get_current_metrics()
        
        print(f"\n📊 DATAMART METRICS (Session: {self.current_session['session_id'][:8]}...)")
        print("=" * 60)
        print(f"🆔 DataMart ID: {self.datamart_id[:8]}...")
        print(f"⏰ Session Start: {self.current_session['start_time'].strftime('%H:%M:%S')}")
        
        # Basic Processing Stats
        print(f"\n📈 PROCESSING STATISTICS:")
        print(f"   Total Processed: {metrics['total_processed']}")
        print(f"   ✅ Successful: {metrics['successful']} ({metrics['success_rate']:.1%})")
        print(f"   ❌ Failed: {metrics['failed']} ({metrics['error_rate']:.1%})")
        print(f"   📄 PDFs Processed: {metrics['pdf_processed']}")
        print(f"   🔧 Non-Standard PDFs: {metrics['non_standard_pdfs']}")
        
        # Knowledge Base & Embedding Status
        print(f"\n📚 KNOWLEDGE BASE & EMBEDDING STATUS:")
        print(f"   📚 In Knowledge Base: {metrics['in_knowledge_base']}")
        print(f"   📚 Not in Knowledge Base: {metrics['not_in_knowledge_base']}")
        print(f"   🔄 New Embeddings: {metrics['new_embeddings']}")
        print(f"   🔄 Existing Embeddings: {metrics['existing_embeddings']}")
        
        # Performance Metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Avg Processing Time: {metrics['avg_processing_time']:.2f}ms")
        print(f"   Processing Range: {metrics['min_processing_time']:.2f}ms - {metrics['max_processing_time']:.2f}ms")
        print(f"   Avg Quality Score: {metrics['avg_quality_score']:.3f}")
        print(f"   Quality Range: {metrics['min_quality_score']:.3f} - {metrics['max_quality_score']:.3f}")
        print(f"   Avg File Size: {metrics['avg_file_size']:.0f} bytes")
        print(f"   Total Size Processed: {metrics['total_size_processed']:,} bytes")
        
        # Content Analysis (Granular Section-by-Section)
        print(f"\n🔍 CONTENT ANALYSIS (Section-by-Section):")
        content_analysis = metrics.get('content_analysis', {})
        print(f"   📑 TOC Found: {content_analysis.get('toc_found', 0)}")
        print(f"   🔗 Links Found: {content_analysis.get('links_found', 0)}")
        print(f"   📖 References Found: {content_analysis.get('references_found', 0)}")
        print(f"   ⚙️ Options Found: {content_analysis.get('options_found', 0)}")
        print(f"   🖼️ Captions Found: {content_analysis.get('captions_found', 0)}")
        
        # Enhanced QA Component Scores
        print(f"\n🎯 ENHANCED QA COMPONENT ANALYSIS:")
        if metrics['successful'] > 0:
            avg_simple = sum([r.get('enhanced_qa_components', {}).get('simple_qa_score', 0) for r in self.processing_history if r.get('success')]) / metrics['successful']
            avg_inspection = sum([r.get('enhanced_qa_components', {}).get('inspection_score', 0) for r in self.processing_history if r.get('success')]) / metrics['successful']
            avg_caption = sum([r.get('enhanced_qa_components', {}).get('caption_score', 0) for r in self.processing_history if r.get('success')]) / metrics['successful']
            avg_prediction = sum([r.get('enhanced_qa_components', {}).get('prediction_score', 0) for r in self.processing_history if r.get('success')]) / metrics['successful']
            
            print(f"   🔧 Simple QA Score: {avg_simple:.3f}")
            print(f"   🔍 Inspection Score: {avg_inspection:.3f}")
            print(f"   🖼️ Caption Score: {avg_caption:.3f}")
            print(f"   🔮 Prediction Score: {avg_prediction:.3f}")
        
        # MVR Peer Review Analysis
        print(f"\n📑 MVR PEER REVIEW ANALYSIS:")
        mvr_analysis = metrics.get('mvr_analysis', {})
        print(f"   📄 MVR Documents Found: {mvr_analysis.get('mvr_documents_found', 0)}")
        print(f"   ✅ Peer Review Ready: {mvr_analysis.get('peer_review_ready', 0)}")
        print(f"   🛡️ MVS Compliant: {mvr_analysis.get('mvs_compliant', 0)}")
        print(f"   📋 VST Scope Identified: {mvr_analysis.get('vst_scope_identified', 0)}")
        print(f"   🔍 Peer Review Challenges: {mvr_analysis.get('peer_review_challenges_generated', 0)}")
        print(f"   🎯 High Confidence Assessments: {mvr_analysis.get('high_confidence_assessments', 0)}")
        
        # QA Criteria Compliance Analysis
        print(f"\n📋 QA CRITERIA COMPLIANCE ANALYSIS:")
        qa_compliance = metrics.get('qa_criteria_compliance', {})
        print(f"   📊 Full Criteria Analyzed: {qa_compliance.get('full_criteria_analyzed', 0)}")
        print(f"   ⚡ Simplified Criteria Analyzed: {qa_compliance.get('simplified_criteria_analyzed', 0)}")
        print(f"   🛡️ Regulatory Compliance Checked: {qa_compliance.get('regulatory_compliance_checked', 0)}")
        print(f"   🔄 Workflow Readiness Assessed: {qa_compliance.get('workflow_readiness_assessed', 0)}")
        print(f"   🚀 Express Mode Ready: {qa_compliance.get('express_mode_ready', 0)}")
        print(f"   🎯 High Compliance Documents: {qa_compliance.get('high_compliance_documents', 0)}")
        
        # Error Analysis
        if metrics['error_types']:
            print(f"\n❌ ERROR BREAKDOWN:")
            for error_type, count in metrics['error_types'].items():
                print(f"   {error_type}: {count}")
        
        print("=" * 60)
    
    def save_datamart_report(self, output_file: str):
        """Save DataMart report to file"""
        report = {
            'datamart_info': {
                'datamart_id': self.datamart_id,
                'session_id': self.current_session['session_id'],
                'start_time': self.current_session['start_time'].isoformat(),
                'end_time': datetime.now().isoformat()
            },
            'metrics': self.get_current_metrics(),
            'processing_history': self.processing_history,
            'session_data': self.current_session
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


class InputFolderPDFLoader:
    """Load PDFs specifically from input folder"""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent.parent
        self.input_folder = self.repo_root / "data" / "input"
        self.pdf_files = []
        
    def load_input_folder_pdfs(self) -> List[Dict[str, Any]]:
        """Load PDF files from input folder"""
        print(f"🔍 Scanning input folder: {self.input_folder}")
        
        if not self.input_folder.exists():
            print(f"❌ Input folder not found: {self.input_folder}")
            return []
        
        # Find all PDF files in input folder and subdirectories
        pdf_files = []
        for pdf_file in self.input_folder.rglob("*.pdf"):
            try:
                file_size = pdf_file.stat().st_size
                relative_path = pdf_file.relative_to(self.repo_root)
                
                # Determine if this is a non-standard PDF
                is_non_standard = self._is_non_standard_pdf(pdf_file)
                
                # Create content representation
                pdf_content = self._create_pdf_content_representation(pdf_file)
                
                if len(pdf_content.strip()) > 50:  # Only include substantial files
                    pdf_files.append({
                        'content': pdf_content,
                        'metadata': {
                            'title': pdf_file.stem.replace('_', ' ').title(),
                            'author': 'Input Document',
                            'type': 'pdf',
                            'file_path': str(relative_path),
                            'file_size': file_size,
                            'source': 'input_folder',
                            'is_non_standard': is_non_standard
                        },
                        'test_id': f"input_pdf_{pdf_file.stem}",
                        'is_non_standard': is_non_standard
                    })
                    
                    print(f"📄 Found PDF: {relative_path} ({file_size:,} bytes) {'[NON-STANDARD]' if is_non_standard else ''}")
                    
            except Exception as e:
                print(f"❌ Error loading {pdf_file}: {e}")
        
        print(f"📊 Total PDFs found: {len(pdf_files)}")
        return pdf_files
    
    def _is_non_standard_pdf(self, pdf_file: Path) -> bool:
        """Determine if PDF is non-standard based on various criteria"""
        try:
            # Check file size (very large or very small might be non-standard)
            file_size = pdf_file.stat().st_size
            if file_size < 1000 or file_size > 50000000:  # < 1KB or > 50MB
                return True
            
            # Check filename patterns that might indicate non-standard content
            filename = pdf_file.name.lower()
            non_standard_patterns = [
                'test', 'sample', 'draft', 'temp', 'backup', 'old', 'archive',
                'corrupted', 'broken', 'invalid', 'scanned', 'image', 'photo'
            ]
            
            for pattern in non_standard_patterns:
                if pattern in filename:
                    return True
            
            # Check if it's in certain subdirectories that might contain non-standard content
            path_str = str(pdf_file).lower()
            non_standard_dirs = ['test', 'sample', 'temp', 'backup', 'archive', 'old']
            for dir_name in non_standard_dirs:
                if f'/{dir_name}/' in path_str:
                    return True
            
            return False
            
        except Exception:
            return True  # Assume non-standard if we can't determine
    
    def _create_pdf_content_representation(self, pdf_file: Path) -> str:
        """Create a content representation for PDF files"""
        try:
            # Try to extract text from PDF if possible
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_file)
            text_content = ""
            
            # Try to extract text from first few pages
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                text_content += page.get_text()
            
            doc.close()
            
            if len(text_content.strip()) > 100:
                return text_content[:3000]  # Limit to first 3000 chars
            else:
                # Fallback to file metadata
                return f"# {pdf_file.stem}\n\nPDF Document: {pdf_file.name}\n\nFile Size: {pdf_file.stat().st_size} bytes\n\nThis is a PDF document from the input folder. Content extraction may be limited due to document format or structure."
        except ImportError:
            # Fallback if PyMuPDF not available
            return f"# {pdf_file.stem}\n\nPDF Document: {pdf_file.name}\n\nFile Size: {pdf_file.stat().st_size} bytes\n\nThis is a PDF document from the input folder. PyMuPDF not available for text extraction."
        except Exception as e:
            # Fallback for any PDF reading errors
            return f"# {pdf_file.stem}\n\nPDF Document: {pdf_file.name}\n\nFile Size: {pdf_file.stat().st_size} bytes\n\nThis is a PDF document from the input folder. Error reading content: {str(e)}"


class InputFolderDataMartEnhancedQATester:
    """Test EnhancedQAOrchestrator with input folder PDFs and DataMart tracking"""
    
    def __init__(self):
        self.orchestrator = StandaloneEnhancedQAOrchestrator()
        self.doc_loader = InputFolderPDFLoader()
        self.datamart = DataMart()
        self.test_results = []
        self.mvr_peer_review_prompt = self._load_mvr_peer_review_prompt()
        self.qa_criteria_config = self._load_qa_criteria_config()
    
    def _load_qa_criteria_config(self) -> Dict[str, Any]:
        """Load QA criteria configuration from YAML files"""
        try:
            import yaml
            
            # Load both full and simplified configurations
            configs = {}
            
            # Full configuration
            full_config_path = Path(__file__).parent.parent.parent / 'dev_configs' / 'qa_criteria_full.yaml'
            if full_config_path.exists():
                with open(full_config_path, 'r') as f:
                    configs['full'] = yaml.safe_load(f)
            
            # Simplified configuration
            simplified_config_path = Path(__file__).parent.parent.parent / 'dev_configs' / 'qa_criteria_simplified.yaml'
            if simplified_config_path.exists():
                with open(simplified_config_path, 'r') as f:
                    configs['simplified'] = yaml.safe_load(f)
            
            print(f"📋 Loaded QA Criteria Configs: {list(configs.keys())}")
            return configs
            
        except Exception as e:
            print(f"⚠️ Error loading QA criteria config: {e}")
            return {}
    
    def _load_mvr_peer_review_prompt(self) -> str:
        """Load the MVR Peer Review Prompt"""
        return """📑 Automated MVR Peer Review Prompt

(Logic- and Evidence-Focused, Execution-Only)

🛡️ System Role

You are an automated peer reviewer trained on the Model Validation Standard (MVS) and Validation Scoping Template (VST).
Your task is to critically evaluate whether the attached Model Validation Report (MVR) demonstrates sufficient, logical, and compliant execution of required review procedures — not to assess model quality, risk, or findings themselves.

⚠️ Important Rules:

Do not assess model performance, risk, or recommendations.

Only flag gaps in the validator's execution of required MVS or VST procedures, and the sufficiency/logic of MRM's analysis and conclusions.

Do not treat model risk findings, observations, or recommendations as compliance gaps unless the MRM's logic, evidence, or rationale is insufficient or unsupported.

Focus strictly on review quality, logic, and documentation.

⚙️ Step 0: Initialization

Initialize output_rows[] as a persistent list.

Parse the MVR Table of Contents (TOC) to heading level 1.1.1.1.

Create section_ids[] as an ordered list of all section identifiers.

Set last_completed_section = None unless a resume point is provided.

If resume_from_section is specified, begin from that section; otherwise, start from the first in TOC.

Set current_section = section_ids[0].

📑 Section Traversal Rules (Critical)

Always begin at the first section listed in TOC.

Do not skip executive summaries, introductions, or scope sections.

Process sections strictly in TOC order, including all subsections.

Skip a section only if explicitly instructed.

Before processing, print the first 5 section IDs. If the first is not the first in TOC, halt and prompt for correction.

🧠 Step 1: Model Context Extraction

Extract model type and risk tier from the first 10 pages or executive summary.

Retrieve and cache:

MVS requirements filtered by model type + risk tier.

VST sections marked in-scope and their test descriptions.

Determine if the validation is targeted or full scope.

📂 Step 2: Section Indexing

Parse TOC to heading level 1.1, 1.1.1, 1.1.1.1, etc.

Record start/end page for each section.

Do not load full document — process one section at a time.

Treat each section as atomic.

Maintain:

section_ids = [0, 0.1, 0.2, 1, 1.1, ..., 5.2.3]

✅ Step 3: Peer Review (Logic- and Evidence-Focused)
3.1 Recursive Evaluation

Evaluate each section and all nested subsections.

Log each as a separate row.

Evaluate parent sections even if they only contain subheaders.

3.2 Peer Review Workflow

For each section:

Extract section text.

Map each MRM conclusion, finding, or assertion to the specific evidence cited.

Trace the Logic: Step-by-step reasoning — are there leaps, unsupported assertions, or gaps?

Effective Challenge: Would a peer reviewer agree with the logic from evidence to conclusion?

Contradiction Search (MANDATORY):

Internal: Look for inconsistencies, omissions, or self-contradictions.

External: Search for regulatory updates, enforcement actions, or industry criticism that may contradict.

Peer Reviewer's Challenge: State the strongest challenge against sufficiency.

Adjust Confidence: Based on evidence strength, gaps, or contradictions.

🎯 Confidence Score Calibration

Certain → direct, unambiguous, independently corroborated evidence.

Highly Confident → strong evidence but self-referential/ambiguous.

Moderately Confident → contradiction or challenge exists.

Speculative/Unknown → weak, indirect, or missing evidence.

📊 Compliance Status

✅ Compliant

⚠️ Partially Compliant

❌ Non-Compliant

❓ Inconclusive

📋 Step 4: Output Format

Append each row to output_rows[]:

MVR Section	MVS Requirement(s) + VST Section(s)	Review Narrative	Contradiction / Challenge Summary	Peer Review Challenge	Conclusion	Confidence Score	Defect Type
[SectionID]	[RequirementIDs]	[Did MRM's logic + evidence meet requirements?]	[Summary of contradictions, or "None"]	[Strongest challenge]	✅ / ⚠️ / ❌ / ❓	Certain / Highly / Moderate / Speculative / Unknown	[If applicable]

⚙️ Step 5: Performance Optimization

Process section-by-section.

Cache TOC + requirements.

Extract only necessary text.

Limit quotes (~5 per section).

Retry incomplete outputs.

Verify all fields in each row.

📑 Step 6: Automation Instructions

Track progress via TOC.

Resume from last_completed_section.

Use next_section = TOC[index(last_completed_section) + 1].

Batch sections dynamically (5–10).

After each batch, export output_rows[] to CSV.

Pause + prompt user if nearing system limit.

Break large sections into sub-chunks.

Never aggregate multiple sections.

Retry skipped/partial sections.

✅ Step 7: Final Output

Return full table in one or more parts.

Prompt user if split.

Concatenate all output_rows[] for final.

Ensure all TOC sections included.

Export to CSV/Excel if conversation length reached.

Key Emphases

Do not critique model's technical quality.

Always trace evidence for each conclusion.

Focus on logic, sufficiency, and documentation.

Include Peer Review Challenge column.

📊 Example Output (Section 4: Conceptual Soundness)
MVR Section	MVS Requirement(s) + VST Section(s)	Review Narrative	Contradiction / Challenge Summary	Peer Review Challenge	Conclusion	Confidence Score	Defect Type
4	MVS 5.4.3, 5.4.3.1–3, 5.12.1; VST Conceptual Soundness	Section covers methodology, segmentation, variable selection, assumptions, retraining. Acknowledges SHAP feature selection limits + lack of uncertainty quantification.	No contradictions. Devil's advocate: reliance on SHAP is not theoretically robust.	Rationale for SHAP-based feature selection is not fully supported; suggest stronger statistical justification.	✅	Highly Confident	N/A"""
        
    def run_input_folder_datamart_test(self):
        """Run comprehensive testing with input folder PDFs and DataMart tracking"""
        print(f"Input Folder DataMart EnhancedQAOrchestrator Test")
        print("=" * 60)
        print(f"Session ID: {self.orchestrator.session_id}")
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Load input folder PDFs
        print("Loading PDFs from input folder...")
        input_pdfs = self.doc_loader.load_input_folder_pdfs()
        
        if not input_pdfs:
            print("No PDF files found in input folder!")
            return
        
        print(f"Loaded {len(input_pdfs)} PDFs from input folder")
        print()
        
        # Show initial DataMart state
        self.datamart.print_current_metrics()
        
        # Run tests
        print("Running tests with input folder PDFs...")
        self._run_tests(input_pdfs)
        
        # Show final DataMart state
        self.datamart.print_current_metrics()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        # Save DataMart report
        self._save_datamart_report()
        
        print(f"\nInput folder DataMart test completed!")
        print(f"DataMart report saved to: {Path(__file__).parent / 'input_folder_datamart_report.json'}")
    
    def _run_tests(self, documents: List[Dict[str, Any]]):
        """Run tests on all input folder PDFs with DataMart tracking"""
        total_docs = len(documents)
        
        for i, doc in enumerate(documents, 1):
            print(f"\nProcessing document {i}/{total_docs}: {doc['test_id']}")
            print(f"  File: {doc['metadata']['file_path']}")
            print(f"  Size: {doc['metadata']['file_size']:,} bytes")
            print(f"  Non-Standard: {'Yes' if doc['is_non_standard'] else 'No'}")
            
            # Process document
            start_time = time.time()
            result = self.orchestrator.process_document(doc)
            processing_time = (time.time() - start_time) * 1000
            
            # Store result
            test_result = {
                'test_id': doc['test_id'],
                'document_type': doc['metadata']['type'],
                'file_path': doc['metadata']['file_path'],
                'file_size': doc['metadata']['file_size'],
                'is_non_standard': doc['is_non_standard'],
                'result': result,
                'processing_time': processing_time,
                'success': result['status'] == 'success'
            }
            self.test_results.append(test_result)
            
            # Add to DataMart
            quality_score = result['enhanced_report']['enhanced_quality_score'] if result['status'] == 'success' else 0.0
            error = result.get('error') if result['status'] != 'success' else None
            
            # Check if this is an MVR document and apply peer review
            mvr_analysis = None
            if self._is_mvr_document(doc):
                mvr_analysis = self._apply_mvr_peer_review(doc, result)
                print(f"  📑 MVR Peer Review: Applied to {doc['metadata']['file_path']}")
            
            # Apply QA Criteria Compliance analysis
            qa_compliance_analysis = self._apply_qa_criteria_compliance(doc, result)
            if qa_compliance_analysis:
                print(f"  📋 QA Compliance: Analyzed {doc['metadata']['file_path']}")
            
            self.datamart.add_processing_record(
                file_path=doc['metadata']['file_path'],
                file_type=doc['metadata']['type'],
                file_size=doc['metadata']['file_size'],
                processing_time=processing_time,
                success=result['status'] == 'success',
                quality_score=quality_score,
                error=error,
                is_non_standard=doc['is_non_standard'],
                enhanced_result=result,
                mvr_analysis=mvr_analysis,
                qa_compliance_analysis=qa_compliance_analysis
            )
            
            # Show quick result
            if result['status'] == 'success':
                enhanced_score = result['enhanced_report']['enhanced_quality_score']
                print(f"  Result: ✅ Success (Score: {enhanced_score:.3f}, Time: {processing_time:.2f}ms)")
            else:
                print(f"  Result: ❌ Failed ({result.get('error', 'Unknown error')})")
            
            # Show DataMart metrics every 5 documents
            if i % 5 == 0:
                self.datamart.print_current_metrics()
            
            # Progress indicator
            if i % 10 == 0:
                print(f"\nProgress: {i}/{total_docs} ({i/total_docs*100:.1f}%)")
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("INPUT FOLDER DATAMART ENHANCED QA TEST REPORT")
        print("=" * 60)
        
        # Get final metrics
        metrics = self.datamart.get_current_metrics()
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   Total PDFs Tested: {metrics['total_processed']}")
        print(f"   Successful: {metrics['successful']}")
        print(f"   Failed: {metrics['failed']}")
        print(f"   Success Rate: {metrics['success_rate']:.1%}")
        print(f"   Error Rate: {metrics['error_rate']:.1%}")
        
        # PDF-specific statistics
        print(f"\n📄 PDF-SPECIFIC STATISTICS")
        print(f"   Total PDFs: {metrics['pdf_processed']}")
        print(f"   Non-Standard PDFs: {metrics['non_standard_pdfs']}")
        print(f"   Standard PDFs: {metrics['pdf_processed'] - metrics['non_standard_pdfs']}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS")
        print(f"   Average Processing Time: {metrics['avg_processing_time']:.2f}ms")
        print(f"   Min Processing Time: {metrics['min_processing_time']:.2f}ms")
        print(f"   Max Processing Time: {metrics['max_processing_time']:.2f}ms")
        
        # Quality metrics
        print(f"\n🎯 QUALITY METRICS")
        print(f"   Average Quality Score: {metrics['avg_quality_score']:.3f}")
        print(f"   Min Quality Score: {metrics['min_quality_score']:.3f}")
        print(f"   Max Quality Score: {metrics['max_quality_score']:.3f}")
        
        # File size metrics
        print(f"\n📏 FILE SIZE METRICS")
        print(f"   Average File Size: {metrics['avg_file_size']:.0f} bytes")
        print(f"   Total Size Processed: {metrics['total_size_processed']:,} bytes")
        
        # Error analysis
        if metrics['error_types']:
            print(f"\n❌ ERROR ANALYSIS")
            for error_type, count in metrics['error_types'].items():
                print(f"   {error_type}: {count}")
        
        # DataMart summary
        print(f"\n🎯 DATAMART SUMMARY")
        print(f"   DataMart ID: {self.datamart.datamart_id[:8]}...")
        print(f"   Session ID: {self.datamart.current_session['session_id'][:8]}...")
        print(f"   Processing Records: {len(self.datamart.processing_history)}")
        print(f"   Session Duration: {datetime.now() - self.datamart.current_session['start_time']}")
    
    def _is_mvr_document(self, doc: Dict[str, Any]) -> bool:
        """Check if document is a Model Validation Report (MVR)"""
        try:
            # Check filename patterns
            filename = doc['metadata']['file_path'].lower()
            mvr_patterns = [
                'mvr', 'model validation', 'validation report', 'model risk',
                'validation_', 'model_validation', 'risk_model'
            ]
            
            for pattern in mvr_patterns:
                if pattern in filename:
                    return True
            
            # Check content for MVR indicators
            content = doc['content'].lower()
            mvr_content_indicators = [
                'model validation', 'validation report', 'model risk management',
                'mvs', 'validation standard', 'peer review', 'compliance',
                'regulatory', 'model governance', 'validation scope'
            ]
            
            indicator_count = sum(1 for indicator in mvr_content_indicators if indicator in content)
            return indicator_count >= 3  # At least 3 indicators suggest MVR
            
        except Exception as e:
            print(f"⚠️ Error checking MVR status: {e}")
            return False
    
    def _apply_mvr_peer_review(self, doc: Dict[str, Any], enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply MVR Peer Review analysis to document"""
        try:
            print(f"  🔍 Applying MVR Peer Review to: {doc['metadata']['file_path']}")
            
            # Extract TOC and section information
            toc_analysis = enhanced_result.get('document_inspection', {}).get('toc_analysis', {})
            structure_analysis = enhanced_result.get('document_inspection', {}).get('structure_analysis', {})
            
            # Create MVR analysis structure
            mvr_analysis = {
                'is_mvr_document': True,
                'toc_sections': toc_analysis.get('headings', []),
                'total_sections': structure_analysis.get('headings', 0),
                'peer_review_ready': len(toc_analysis.get('headings', [])) > 0,
                'mvs_compliance_check': self._check_mvs_compliance(doc, enhanced_result),
                'vst_scope_analysis': self._analyze_vst_scope(doc, enhanced_result),
                'peer_review_challenges': self._generate_peer_review_challenges(doc, enhanced_result),
                'confidence_assessment': self._assess_confidence(doc, enhanced_result)
            }
            
            print(f"  📊 MVR Analysis: {mvr_analysis['total_sections']} sections, {len(mvr_analysis['toc_sections'])} TOC entries")
            return mvr_analysis
            
        except Exception as e:
            print(f"⚠️ Error applying MVR peer review: {e}")
            return {
                'is_mvr_document': True,
                'error': str(e),
                'peer_review_ready': False
            }
    
    def _check_mvs_compliance(self, doc: Dict[str, Any], enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Check MVS (Model Validation Standard) compliance"""
        try:
            content = doc['content'].lower()
            
            # MVS requirement patterns
            mvs_requirements = {
                'conceptual_soundness': ['conceptual soundness', 'model theory', 'methodology'],
                'ongoing_monitoring': ['ongoing monitoring', 'performance monitoring', 'backtesting'],
                'outcome_analysis': ['outcome analysis', 'results analysis', 'validation results'],
                'change_management': ['change management', 'model changes', 'modifications'],
                'governance': ['governance', 'policies', 'procedures', 'controls']
            }
            
            compliance_status = {}
            for req_name, keywords in mvs_requirements.items():
                compliance_status[req_name] = {
                    'found': any(keyword in content for keyword in keywords),
                    'keywords_found': [kw for kw in keywords if kw in content],
                    'score': sum(1 for kw in keywords if kw in content) / len(keywords)
                }
            
            return compliance_status
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_vst_scope(self, doc: Dict[str, Any], enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze VST (Validation Scoping Template) scope"""
        try:
            content = doc['content'].lower()
            
            # VST scope indicators
            vst_indicators = {
                'full_scope': ['comprehensive', 'full scope', 'complete validation'],
                'targeted_scope': ['targeted', 'limited scope', 'focused validation'],
                'risk_tier': ['high risk', 'medium risk', 'low risk', 'risk tier'],
                'model_type': ['credit risk', 'market risk', 'operational risk', 'liquidity risk']
            }
            
            scope_analysis = {}
            for indicator_name, keywords in vst_indicators.items():
                scope_analysis[indicator_name] = {
                    'found': any(keyword in content for keyword in keywords),
                    'keywords_found': [kw for kw in keywords if kw in content]
                }
            
            return scope_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_peer_review_challenges(self, doc: Dict[str, Any], enhanced_result: Dict[str, Any]) -> List[str]:
        """Generate peer review challenges based on content analysis"""
        try:
            challenges = []
            content = doc['content'].lower()
            
            # Common peer review challenge patterns
            challenge_patterns = [
                ('insufficient_evidence', 'insufficient evidence', 'lack of evidence', 'no evidence'),
                ('methodology_gaps', 'methodology gap', 'approach limitations', 'method limitations'),
                ('data_quality', 'data quality', 'data limitations', 'data issues'),
                ('assumption_validation', 'assumption', 'assumptions not validated', 'unvalidated'),
                ('documentation_gaps', 'documentation gap', 'incomplete documentation', 'missing documentation')
            ]
            
            for challenge_type, keywords in challenge_patterns:
                if any(keyword in content for keyword in keywords):
                    challenges.append(f"{challenge_type.replace('_', ' ').title()}: Found in content")
            
            return challenges
            
        except Exception as e:
            return [f"Error generating challenges: {e}"]
    
    def _assess_confidence(self, doc: Dict[str, Any], enhanced_result: Dict[str, Any]) -> str:
        """Assess confidence level based on evidence and analysis quality"""
        try:
            # Use Enhanced QA scores to assess confidence
            enhanced_report = enhanced_result.get('enhanced_report', {})
            inspection_score = enhanced_report.get('inspection_score', 0)
            prediction_score = enhanced_report.get('prediction_score', 0)
            
            if inspection_score > 0.8 and prediction_score > 0.8:
                return "Highly Confident"
            elif inspection_score > 0.6 and prediction_score > 0.6:
                return "Moderately Confident"
            elif inspection_score > 0.4 and prediction_score > 0.4:
                return "Speculative"
            else:
                return "Unknown"
                
        except Exception as e:
            return "Unknown"
    
    def _apply_qa_criteria_compliance(self, doc: Dict[str, Any], enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply QA Criteria Compliance analysis to document"""
        try:
            if not self.qa_criteria_config:
                return None
            
            print(f"  🔍 Applying QA Criteria Compliance to: {doc['metadata']['file_path']}")
            
            # Analyze against both full and simplified criteria
            compliance_analysis = {
                'full_criteria_analysis': self._analyze_full_criteria(doc, enhanced_result),
                'simplified_criteria_analysis': self._analyze_simplified_criteria(doc, enhanced_result),
                'regulatory_compliance': self._check_regulatory_compliance(doc, enhanced_result),
                'workflow_readiness': self._assess_workflow_readiness(doc, enhanced_result)
            }
            
            return compliance_analysis
            
        except Exception as e:
            print(f"⚠️ Error applying QA criteria compliance: {e}")
            return None
    
    def _analyze_full_criteria(self, doc: Dict[str, Any], enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document against full QA criteria"""
        try:
            full_config = self.qa_criteria_config.get('full', {})
            categories = full_config.get('checklist_categories', [])
            
            analysis = {
                'categories_analyzed': len(categories),
                'criteria_covered': 0,
                'compliance_scores': {},
                'missing_criteria': [],
                'evidence_gaps': []
            }
            
            content = doc['content'].lower()
            
            for category in categories:
                category_id = category.get('id', 'unknown')
                category_name = category.get('name', 'Unknown')
                criteria_list = category.get('criteria', [])
                
                category_score = 0
                criteria_covered = 0
                
                for criterion in criteria_list:
                    criterion_id = criterion.get('id', 'unknown')
                    criterion_text = criterion.get('text', '').lower()
                    keywords = criterion_text.split()
                    
                    # Check if criterion is mentioned in content
                    if any(keyword in content for keyword in keywords if len(keyword) > 3):
                        criteria_covered += 1
                        category_score += 1
                    else:
                        analysis['missing_criteria'].append({
                            'category': category_name,
                            'criterion_id': criterion_id,
                            'criterion_text': criterion.get('text', ''),
                            'regulatory_refs': criterion.get('regulatory_references', [])
                        })
                
                if criteria_list:
                    category_score = (criteria_covered / len(criteria_list)) * 100
                
                analysis['compliance_scores'][category_id] = {
                    'name': category_name,
                    'score': category_score,
                    'criteria_covered': criteria_covered,
                    'total_criteria': len(criteria_list)
                }
                
                analysis['criteria_covered'] += criteria_covered
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_simplified_criteria(self, doc: Dict[str, Any], enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document against simplified QA criteria"""
        try:
            simplified_config = self.qa_criteria_config.get('simplified', {})
            categories = simplified_config.get('checklist_categories', [])
            
            analysis = {
                'categories_analyzed': len(categories),
                'criteria_covered': 0,
                'compliance_scores': {},
                'express_mode_ready': False,
                'missing_criteria': []
            }
            
            content = doc['content'].lower()
            total_criteria = 0
            covered_criteria = 0
            
            for category in categories:
                category_id = category.get('id', 'unknown')
                category_name = category.get('name', 'Unknown')
                criteria_list = category.get('criteria', [])
                
                category_score = 0
                criteria_covered = 0
                
                for criterion in criteria_list:
                    total_criteria += 1
                    criterion_id = criterion.get('id', 'unknown')
                    criterion_text = criterion.get('text', '').lower()
                    keywords = criterion_text.split()
                    
                    # Check if criterion is mentioned in content
                    if any(keyword in content for keyword in keywords if len(keyword) > 3):
                        criteria_covered += 1
                        covered_criteria += 1
                        category_score += 1
                    else:
                        analysis['missing_criteria'].append({
                            'category': category_name,
                            'criterion_id': criterion_id,
                            'criterion_text': criterion.get('text', ''),
                            'combined_criteria': criterion.get('combined_criteria', [])
                        })
                
                if criteria_list:
                    category_score = (criteria_covered / len(criteria_list)) * 100
                
                analysis['compliance_scores'][category_id] = {
                    'name': category_name,
                    'score': category_score,
                    'criteria_covered': criteria_covered,
                    'total_criteria': len(criteria_list)
                }
            
            # Determine if document is ready for express mode
            if total_criteria > 0:
                coverage_ratio = covered_criteria / total_criteria
                analysis['express_mode_ready'] = coverage_ratio >= 0.7  # 70% coverage threshold
            
            analysis['criteria_covered'] = covered_criteria
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _check_regulatory_compliance(self, doc: Dict[str, Any], enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Check regulatory compliance based on QA criteria"""
        try:
            content = doc['content'].lower()
            
            # Regulatory references from both configs
            regulatory_refs = set()
            
            for config_type, config in self.qa_criteria_config.items():
                categories = config.get('checklist_categories', [])
                for category in categories:
                    criteria_list = category.get('criteria', [])
                    for criterion in criteria_list:
                        refs = criterion.get('regulatory_references', [])
                        regulatory_refs.update(refs)
            
            compliance_status = {}
            for ref in regulatory_refs:
                # Check if regulatory reference is mentioned
                ref_lower = ref.lower()
                compliance_status[ref] = {
                    'mentioned': ref_lower in content,
                    'context': self._find_regulatory_context(content, ref_lower)
                }
            
            return {
                'regulatory_references_found': list(regulatory_refs),
                'compliance_status': compliance_status,
                'total_regulatory_refs': len(regulatory_refs)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _find_regulatory_context(self, content: str, ref: str) -> str:
        """Find context around regulatory reference"""
        try:
            if ref not in content:
                return ""
            
            # Find position of reference
            pos = content.find(ref)
            start = max(0, pos - 100)
            end = min(len(content), pos + len(ref) + 100)
            
            return content[start:end].strip()
            
        except Exception:
            return ""
    
    def _assess_workflow_readiness(self, doc: Dict[str, Any], enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess document readiness for QA workflow steps"""
        try:
            content = doc['content'].lower()
            
            # Workflow steps from both configs
            workflow_steps = []
            
            for config_type, config in self.qa_criteria_config.items():
                steps = config.get('workflow_steps', [])
                workflow_steps.extend(steps)
            
            readiness_assessment = {
                'workflow_steps_identified': len(workflow_steps),
                'document_ready_for': [],
                'missing_prerequisites': [],
                'estimated_completion_time': 0
            }
            
            # Check document readiness for each workflow step
            for step in workflow_steps:
                step_id = step.get('id', 'unknown')
                step_name = step.get('name', 'Unknown')
                completion_criteria = step.get('completion_criteria', [])
                
                # Check if document meets completion criteria
                criteria_met = 0
                for criterion in completion_criteria:
                    if criterion.lower() in content:
                        criteria_met += 1
                
                if criteria_met > 0:
                    readiness_assessment['document_ready_for'].append({
                        'step_id': step_id,
                        'step_name': step_name,
                        'criteria_met': criteria_met,
                        'total_criteria': len(completion_criteria)
                    })
                else:
                    readiness_assessment['missing_prerequisites'].append({
                        'step_id': step_id,
                        'step_name': step_name,
                        'missing_criteria': completion_criteria
                    })
            
            # Calculate estimated completion time
            total_time = 0
            for step in workflow_steps:
                time_str = step.get('estimated_time', '0 minutes')
                if 'hour' in time_str:
                    hours = int(time_str.split()[0])
                    total_time += hours * 60
                elif 'minute' in time_str:
                    minutes = int(time_str.split()[0])
                    total_time += minutes
            
            readiness_assessment['estimated_completion_time'] = total_time
            
            return readiness_assessment
            
        except Exception as e:
            return {'error': str(e)}
    
    def _save_datamart_report(self):
        """Save DataMart report"""
        output_file = Path(__file__).parent / 'input_folder_datamart_report.json'
        self.datamart.save_datamart_report(str(output_file))


def main():
    """Main function to run input folder DataMart testing"""
    tester = InputFolderDataMartEnhancedQATester()
    tester.run_input_folder_datamart_test()


if __name__ == '__main__':
    main()
