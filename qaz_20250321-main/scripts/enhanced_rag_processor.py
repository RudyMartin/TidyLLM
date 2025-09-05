#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RAG Processor

This script processes PDFs with advanced capabilities:
- Image detection and handling
- Table extraction and analysis
- Reference validation
- Contradiction detection
- Argument reasoning analysis
- Cross-document consistency checking
"""

import os
import sys
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import fitz  # pymupdf
import pandas as pd
from datetime import datetime

class EnhancedRAGProcessor:
    """Enhanced RAG processor with advanced document analysis capabilities"""
    
    def __init__(self):
        self.processed_documents = {}
        self.reference_database = {}
        self.contradiction_log = []
        self.argument_analysis = {}
        
    def extract_images_and_tables(self, pdf_path: str) -> Dict[str, Any]:
        """Extract images and tables from PDF with detailed analysis"""
        try:
            doc = fitz.open(pdf_path)
            document_analysis = {
                "images": [],
                "tables": [],
                "charts": [],
                "diagrams": []
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = {
                                "page": page_num + 1,
                                "index": img_index,
                                "width": pix.width,
                                "height": pix.height,
                                "colorspace": pix.colorspace.name,
                                "size_bytes": len(pix.tobytes()),
                                "type": self._classify_image(pix)
                            }
                            document_analysis["images"].append(img_data)
                        pix = None
                    except Exception as e:
                        print(f"⚠️ Error processing image on page {page_num + 1}: {e}")
                
                # Extract tables using PyMuPDF's table finder
                try:
                    tables = page.find_tables()
                    for table_index, table in enumerate(tables):
                        table_data = {
                            "page": page_num + 1,
                            "index": table_index,
                            "rows": len(table.rows),
                            "columns": len(table.header),
                            "content": table.extract(),
                            "type": self._classify_table(table.extract())
                        }
                        document_analysis["tables"].append(table_data)
                except Exception as e:
                    print(f"⚠️ Error extracting tables from page {page_num + 1}: {e}")
            
            doc.close()
            return document_analysis
            
        except Exception as e:
            print(f"❌ Error extracting images and tables: {e}")
            return {"images": [], "tables": [], "charts": [], "diagrams": []}
    
    def _classify_image(self, pix) -> str:
        """Classify image type based on characteristics"""
        if pix.width > pix.height * 1.5:
            return "chart"
        elif pix.width < 200 and pix.height < 200:
            return "icon"
        elif pix.colorspace.name == "DeviceGray":
            return "diagram"
        else:
            return "image"
    
    def _classify_table(self, table_content: List[List[str]]) -> str:
        """Classify table type based on content"""
        if not table_content:
            return "unknown"
        
        # Check for numerical data
        numeric_count = 0
        total_cells = 0
        
        for row in table_content:
            for cell in row:
                total_cells += 1
                if re.search(r'\d+\.?\d*', cell):
                    numeric_count += 1
        
        if numeric_count / total_cells > 0.5:
            return "data_table"
        elif len(table_content) > 10:
            return "reference_table"
        else:
            return "summary_table"
    
    def extract_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract and validate academic references"""
        references = []
        
        # Common reference patterns
        patterns = [
            r'([A-Z][a-z]+,\s*[A-Z]\.\s*\d{4})',  # Author, A. YYYY
            r'([A-Z][a-z]+\s+et\s+al\.\s*\d{4})',  # Author et al. YYYY
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+,\s*\d{4})',  # Author, Author, YYYY
            r'([A-Z][a-z]+\s+&\s+[A-Z][a-z]+,\s*\d{4})',  # Author & Author, YYYY
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                ref = {
                    "text": match.group(1),
                    "position": match.start(),
                    "validated": False,
                    "type": "academic_reference"
                }
                references.append(ref)
        
        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        url_matches = re.finditer(url_pattern, text)
        for match in url_matches:
            ref = {
                "text": match.group(0),
                "position": match.start(),
                "validated": False,
                "type": "url"
            }
            references.append(ref)
        
        return references
    
    def detect_contradictions(self, text: str, document_name: str) -> List[Dict[str, Any]]:
        """Detect contradictions and inconsistencies in text"""
        contradictions = []
        
        # Contradiction indicators
        contradiction_phrases = [
            r'however.*but',
            r'although.*nevertheless',
            r'on the other hand',
            r'in contrast',
            r'contradicts',
            r'inconsistent',
            r'disagrees with',
            r'conflicts with'
        ]
        
        for phrase in contradiction_phrases:
            matches = re.finditer(phrase, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                context = text[start:end]
                
                contradiction = {
                    "document": document_name,
                    "phrase": match.group(0),
                    "position": match.start(),
                    "context": context,
                    "severity": self._assess_contradiction_severity(context)
                }
                contradictions.append(contradiction)
        
        return contradictions
    
    def _assess_contradiction_severity(self, context: str) -> str:
        """Assess the severity of a contradiction"""
        strong_indicators = ['contradicts', 'conflicts', 'inconsistent', 'disagrees']
        moderate_indicators = ['however', 'although', 'nevertheless']
        
        context_lower = context.lower()
        
        if any(indicator in context_lower for indicator in strong_indicators):
            return "high"
        elif any(indicator in context_lower for indicator in moderate_indicators):
            return "medium"
        else:
            return "low"
    
    def analyze_argument_reasoning(self, text: str) -> Dict[str, Any]:
        """Analyze the reasoning and argument structure"""
        analysis = {
            "claims": [],
            "evidence": [],
            "assumptions": [],
            "logical_fallacies": [],
            "reasoning_quality": "unknown"
        }
        
        # Extract claims (statements that need support)
        claim_patterns = [
            r'[A-Z][^.!?]*\s+(is|are|was|were|will be|should be|must be)[^.!?]*[.!?]',
            r'[A-Z][^.!?]*\s+(proves|demonstrates|shows|indicates|suggests)[^.!?]*[.!?]',
            r'[A-Z][^.!?]*\s+(therefore|thus|consequently|as a result)[^.!?]*[.!?]'
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                analysis["claims"].append({
                    "text": match.group(0),
                    "type": "claim",
                    "position": match.start()
                })
        
        # Extract evidence (data, statistics, examples)
        evidence_patterns = [
            r'\d+%',
            r'\d+\.\d+',
            r'study shows',
            r'research indicates',
            r'data suggests',
            r'example',
            r'case study'
        ]
        
        for pattern in evidence_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                analysis["evidence"].append({
                    "text": match.group(0),
                    "type": "evidence",
                    "position": match.start()
                })
        
        # Assess reasoning quality
        claim_count = len(analysis["claims"])
        evidence_count = len(analysis["evidence"])
        
        if evidence_count > claim_count * 2:
            analysis["reasoning_quality"] = "strong"
        elif evidence_count > claim_count:
            analysis["reasoning_quality"] = "moderate"
        else:
            analysis["reasoning_quality"] = "weak"
        
        return analysis
    
    def cross_document_consistency_check(self, current_doc: str, all_documents: Dict) -> List[Dict[str, Any]]:
        """Check consistency across multiple documents"""
        inconsistencies = []
        
        if not all_documents:
            return inconsistencies
        
        # Extract key claims from current document
        current_claims = self._extract_key_claims(all_documents.get(current_doc, {}))
        
        for doc_name, doc_data in all_documents.items():
            if doc_name == current_doc:
                continue
            
            other_claims = self._extract_key_claims(doc_data)
            
            # Compare claims for inconsistencies
            for current_claim in current_claims:
                for other_claim in other_claims:
                    if self._claims_contradict(current_claim, other_claim):
                        inconsistency = {
                            "document1": current_doc,
                            "document2": doc_name,
                            "claim1": current_claim,
                            "claim2": other_claim,
                            "type": "contradiction"
                        }
                        inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _extract_key_claims(self, doc_data: Dict) -> List[str]:
        """Extract key claims from document data"""
        claims = []
        if "text_content" in doc_data:
            text = doc_data["text_content"]
            # Extract sentences that make claims
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['is', 'are', 'proves', 'shows', 'indicates']):
                    claims.append(sentence.strip())
        return claims
    
    def _claims_contradict(self, claim1: str, claim2: str) -> bool:
        """Check if two claims contradict each other"""
        # Simple contradiction detection
        negation_words = ['not', 'never', 'no', 'none', 'neither', 'nor']
        
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()
        
        # Check for direct negations
        for negation in negation_words:
            if negation in claim1_lower and negation not in claim2_lower:
                # Check if they're talking about the same subject
                if self._same_subject(claim1, claim2):
                    return True
        
        return False
    
    def _same_subject(self, claim1: str, claim2: str) -> bool:
        """Check if two claims are about the same subject"""
        # Extract key nouns (simplified)
        nouns1 = re.findall(r'\b[A-Z][a-z]+\b', claim1)
        nouns2 = re.findall(r'\b[A-Z][a-z]+\b', claim2)
        
        # Check for overlap in key terms
        overlap = set(nouns1) & set(nouns2)
        return len(overlap) >= 2  # At least 2 common terms
    
    def process_document_enhanced(self, pdf_path: str) -> Dict[str, Any]:
        """Process a document with all enhanced capabilities"""
        print(f"🔍 Processing: {os.path.basename(pdf_path)}")
        
        document_analysis = {
            "document_name": os.path.splitext(os.path.basename(pdf_path))[0],
            "file_size": os.path.getsize(pdf_path),
            "processing_timestamp": datetime.now().isoformat(),
            "images_and_tables": {},
            "references": [],
            "contradictions": [],
            "argument_analysis": {},
            "cross_document_inconsistencies": [],
            "chunks": [],
            "summary": {}
        }
        
        try:
            # Extract images and tables
            print("  📊 Extracting images and tables...")
            document_analysis["images_and_tables"] = self.extract_images_and_tables(pdf_path)
            
            # Process text content
            print("  📝 Processing text content...")
            doc = fitz.open(pdf_path)
            all_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                all_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            doc.close()
            
            # Extract references
            print("  📚 Extracting references...")
            document_analysis["references"] = self.extract_references(all_text)
            
            # Detect contradictions
            print("  ⚠️ Detecting contradictions...")
            document_analysis["contradictions"] = self.detect_contradictions(
                all_text, document_analysis["document_name"]
            )
            
            # Analyze argument reasoning
            print("  🧠 Analyzing argument reasoning...")
            document_analysis["argument_analysis"] = self.analyze_argument_reasoning(all_text)
            
            # Create chunks with enhanced metadata
            print("  ✂️ Creating enhanced chunks...")
            chunks = self._create_enhanced_chunks(all_text, document_analysis["document_name"])
            document_analysis["chunks"] = chunks
            
            # Generate summary
            print("  📋 Generating summary...")
            document_analysis["summary"] = self._generate_document_summary(document_analysis)
            
            # Store in database
            print("  🗄️ Storing in database...")
            self._store_enhanced_document(document_analysis)
            
            # Update processed documents
            self.processed_documents[document_analysis["document_name"]] = document_analysis
            
            print(f"✅ Completed enhanced processing of {document_analysis['document_name']}")
            return document_analysis
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            return document_analysis
    
    def _create_enhanced_chunks(self, text: str, document_name: str) -> List[Dict]:
        """Create chunks with enhanced metadata"""
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = []
        current_length = 0
        chunk_index = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > 200 and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunk_data = {
                    "chunk_id": f"{document_name}_chunk_{chunk_index:03d}",
                    "text": chunk_text,
                    "metadata": {
                        "document_name": document_name,
                        "chunk_number": chunk_index,
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        "has_references": bool(self.extract_references(chunk_text)),
                        "has_contradictions": bool(self.detect_contradictions(chunk_text, document_name)),
                        "argument_quality": self.analyze_argument_reasoning(chunk_text)["reasoning_quality"]
                    }
                }
                chunks.append(chunk_data)
                
                current_chunk = []
                current_length = 0
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_data = {
                "chunk_id": f"{document_name}_chunk_{chunk_index:03d}",
                "text": chunk_text,
                "metadata": {
                    "document_name": document_name,
                    "chunk_number": chunk_index,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text),
                    "has_references": bool(self.extract_references(chunk_text)),
                    "has_contradictions": bool(self.detect_contradictions(chunk_text, document_name)),
                    "argument_quality": self.analyze_argument_reasoning(chunk_text)["reasoning_quality"]
                }
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _generate_document_summary(self, analysis: Dict) -> Dict:
        """Generate comprehensive document summary"""
        summary = {
            "total_chunks": len(analysis["chunks"]),
            "total_images": len(analysis["images_and_tables"]["images"]),
            "total_tables": len(analysis["images_and_tables"]["tables"]),
            "total_references": len(analysis["references"]),
            "total_contradictions": len(analysis["contradictions"]),
            "argument_quality": analysis["argument_analysis"]["reasoning_quality"],
            "key_findings": [],
            "recommendations": []
        }
        
        # Generate key findings
        if analysis["contradictions"]:
            summary["key_findings"].append(f"Found {len(analysis['contradictions'])} potential contradictions")
        
        if analysis["references"]:
            summary["key_findings"].append(f"Contains {len(analysis['references'])} references")
        
        if analysis["images_and_tables"]["tables"]:
            summary["key_findings"].append(f"Contains {len(analysis['images_and_tables']['tables'])} data tables")
        
        # Generate recommendations
        if analysis["argument_analysis"]["reasoning_quality"] == "weak":
            summary["recommendations"].append("Document has weak reasoning - needs more evidence")
        
        if len(analysis["contradictions"]) > 5:
            summary["recommendations"].append("High number of contradictions - requires careful review")
        
        return summary
    
    def _store_enhanced_document(self, analysis: Dict):
        """Store enhanced document analysis in database"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            load_dotenv('src/backend/config/credentials.env')
            database_url = os.getenv('DATABASE_URL')
            
            if not database_url:
                print("⚠️ DATABASE_URL not found - skipping database storage")
                return
            
            conn = psycopg2.connect(database_url)
            
            # Store enhanced document metadata
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS enhanced_documents (
                        id SERIAL PRIMARY KEY,
                        document_name VARCHAR(255) UNIQUE NOT NULL,
                        file_size BIGINT,
                        processing_timestamp TIMESTAMP,
                        total_chunks INTEGER,
                        total_images INTEGER,
                        total_tables INTEGER,
                        total_references INTEGER,
                        total_contradictions INTEGER,
                        argument_quality VARCHAR(50),
                        analysis_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                cursor.execute("""
                    INSERT INTO enhanced_documents 
                    (document_name, file_size, processing_timestamp, total_chunks, 
                     total_images, total_tables, total_references, total_contradictions, 
                     argument_quality, analysis_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_name) DO UPDATE SET
                    processing_timestamp = EXCLUDED.processing_timestamp,
                    analysis_data = EXCLUDED.analysis_data
                """, (
                    analysis["document_name"],
                    analysis["file_size"],
                    analysis["processing_timestamp"],
                    analysis["summary"]["total_chunks"],
                    analysis["summary"]["total_images"],
                    analysis["summary"]["total_tables"],
                    analysis["summary"]["total_references"],
                    analysis["summary"]["total_contradictions"],
                    analysis["summary"]["argument_quality"],
                    json.dumps(analysis)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Error storing enhanced document: {e}")
    
    def run_cross_document_analysis(self):
        """Run cross-document consistency analysis"""
        print("🔍 Running cross-document consistency analysis...")
        
        for doc_name in self.processed_documents:
            inconsistencies = self.cross_document_consistency_check(
                doc_name, self.processed_documents
            )
            
            if inconsistencies:
                self.processed_documents[doc_name]["cross_document_inconsistencies"] = inconsistencies
                print(f"  ⚠️ Found {len(inconsistencies)} inconsistencies in {doc_name}")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        report = {
            "processing_summary": {
                "total_documents": len(self.processed_documents),
                "total_chunks": sum(len(doc["chunks"]) for doc in self.processed_documents.values()),
                "total_images": sum(len(doc["images_and_tables"]["images"]) for doc in self.processed_documents.values()),
                "total_tables": sum(len(doc["images_and_tables"]["tables"]) for doc in self.processed_documents.values()),
                "total_references": sum(len(doc["references"]) for doc in self.processed_documents.values()),
                "total_contradictions": sum(len(doc["contradictions"]) for doc in self.processed_documents.values()),
            },
            "document_analyses": self.processed_documents,
            "cross_document_findings": {},
            "recommendations": []
        }
        
        # Generate recommendations
        total_contradictions = report["processing_summary"]["total_contradictions"]
        if total_contradictions > 20:
            report["recommendations"].append("High number of contradictions across documents - requires systematic review")
        
        weak_reasoning_docs = [
            doc_name for doc_name, doc_data in self.processed_documents.items()
            if doc_data["argument_analysis"]["reasoning_quality"] == "weak"
        ]
        
        if weak_reasoning_docs:
            report["recommendations"].append(f"Documents with weak reasoning: {', '.join(weak_reasoning_docs)}")
        
        return report

def main():
    """Main function to process all documents with enhanced capabilities"""
    processor = EnhancedRAGProcessor()
    
    # Get all PDF files
    reviews_dir = Path("data/input/reviews")
    pdf_files = list(reviews_dir.glob("*.pdf"))
    
    print(f"🚀 Enhanced RAG Processing - {len(pdf_files)} documents")
    print("=" * 60)
    
    # Process each document
    for pdf_file in pdf_files:
        if pdf_file.name in ["Whitepaper-Model-Validation-Best-Practices-1.pdf", "investment-model-validation.pdf"]:
            print(f"⏭️ Skipping already processed: {pdf_file.name}")
            continue
        
        try:
            analysis = processor.process_document_enhanced(str(pdf_file))
            print(f"✅ Completed: {pdf_file.name}")
        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")
    
    # Run cross-document analysis
    processor.run_cross_document_analysis()
    
    # Generate comprehensive report
    report = processor.generate_comprehensive_report()
    
    # Save report
    with open("enhanced_rag_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n🎉 Enhanced RAG processing completed!")
    print(f"📊 Processed {report['processing_summary']['total_documents']} documents")
    print(f"📝 Generated {report['processing_summary']['total_chunks']} chunks")
    print(f"📊 Found {report['processing_summary']['total_images']} images")
    print(f"📋 Found {report['processing_summary']['total_tables']} tables")
    print(f"📚 Found {report['processing_summary']['total_references']} references")
    print(f"⚠️ Found {report['processing_summary']['total_contradictions']} contradictions")
    print(f"📄 Report saved to: enhanced_rag_analysis_report.json")

if __name__ == "__main__":
    main()
