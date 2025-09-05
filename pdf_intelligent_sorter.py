#!/usr/bin/env python3
"""

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

Intelligent PDF Sorter for Knowledge Base
=========================================

Automatically sorts PDFs into three categories based on filename analysis:
- sop: Standard Operating Procedures, guidelines, best practices
- checklist: Regulatory requirements, compliance checklists, mandated processes  
- modeling: Technical modeling, validation methods, mathematical approaches

Uses intelligent keyword matching and filename analysis.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

class IntelligentPDFSorter:
    """Intelligently sorts PDFs based on content type analysis"""
    
    def __init__(self, source_dir: str = build_s3_path("knowledge_base", "pdfs")):
        self.source_dir = Path(source_dir)
        self.target_base = Path("knowledge_base")
        
        # Create target directories
        self.sop_dir = self.target_base / "sop"
        self.checklist_dir = self.target_base / "checklist" 
        self.modeling_dir = self.target_base / "modeling"
        
        for directory in [self.sop_dir, self.checklist_dir, self.modeling_dir]:
            directory.mkdir(exist_ok=True)
        
        # Define classification keywords
        self.classification_rules = {
            'checklist': {
                'keywords': [
                    'supervisory', 'regulatory', 'compliance', 'board', 'federal', 
                    'reserve', 'occ', 'basel', 'regulation', 'requirement', 'mandate',
                    'guidance', 'bulletin', 'circular', 'directive', 'framework',
                    'stress-testing', 'supervisory-stress-testing', 'instructions',
                    'reporting', 'bcbs', 'sr-', 'sr_', 'occ-'
                ],
                'description': 'Regulatory/compliance checklists and mandated requirements'
            },
            'modeling': {
                'keywords': [
                    'model-validation', 'validation', 'mathematical', 'algorithm',
                    'machine-learning', 'ml-model', 'statistical', 'quantitative',
                    'technical', 'methodology', 'approach', 'method', 'tool',
                    'benchmark', 'performance', 'accuracy', 'testing', 'tuning',
                    'distract', 'longmem', 'gem15', 'mich', 'abench'
                ],
                'description': 'Technical modeling, validation methods, and mathematical approaches'
            },
            'sop': {
                'keywords': [
                    'best-practice', 'practice', 'procedure', 'process', 'workflow',
                    'standard', 'guideline', 'overview', 'demo', 'tutorial', 
                    'reference', 'manual', 'handbook', 'guide', 'master',
                    'investment', 'credit-risk', 'risk-management', 'operational',
                    'bookshelf', 'canaries', 'scor'
                ],
                'description': 'Standard operating procedures, guidelines, and best practices'
            }
        }
    
    def analyze_filename(self, filename: str) -> Tuple[str, float, List[str]]:
        """
        Analyze filename to determine category
        
        Returns:
            Tuple of (category, confidence_score, matched_keywords)
        """
        filename_lower = filename.lower()
        
        category_scores = {}
        category_matches = {}
        
        for category, rules in self.classification_rules.items():
            matches = []
            score = 0
            
            for keyword in rules['keywords']:
                if keyword in filename_lower:
                    matches.append(keyword)
                    # Weight longer keywords more heavily
                    score += len(keyword) * 0.1
                    # Boost exact matches
                    if keyword == filename_lower.replace('.pdf', ''):
                        score += 2.0
            
            category_scores[category] = score
            category_matches[category] = matches
        
        # Determine best category
        if not any(category_scores.values()):
            # No keywords matched - classify as SOP by default
            return 'sop', 0.1, ['default_classification']
        
        best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
        best_score = category_scores[best_category]
        best_matches = category_matches[best_category]
        
        # Calculate confidence (normalize to 0-1 range)
        max_possible_score = len(self.classification_rules[best_category]['keywords']) * 0.5
        confidence = min(1.0, best_score / max_possible_score) if max_possible_score > 0 else 0.1
        
        return best_category, confidence, best_matches
    
    def sort_pdfs(self, dry_run: bool = False) -> Dict[str, List[Dict]]:
        """
        Sort all PDFs in the source directory
        
        Args:
            dry_run: If True, only analyze without moving files
            
        Returns:
            Dictionary with sorting results
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory {self.source_dir} not found")
        
        pdf_files = list(self.source_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"[WARNING] No PDF files found in {self.source_dir}")
            return {}
        
        print("="*60)
        print("INTELLIGENT PDF SORTER - KNOWLEDGE BASE ORGANIZATION")
        print("="*60)
        print(f"Source: {self.source_dir}")
        print(f"Found: {len(pdf_files)} PDF files to sort")
        print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
        print("="*60)
        
        results = {
            'sop': [],
            'checklist': [],
            'modeling': [],
            'summary': {
                'total_files': len(pdf_files),
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            }
        }
        
        for pdf_file in pdf_files:
            category, confidence, matches = self.analyze_filename(pdf_file.name)
            
            file_info = {
                'filename': pdf_file.name,
                'category': category,
                'confidence': confidence,
                'matched_keywords': matches,
                'source_path': str(pdf_file),
                'target_path': str(getattr(self, f"{category}_dir") / pdf_file.name)
            }
            
            results[category].append(file_info)
            
            # Update confidence statistics
            if confidence >= 0.7:
                results['summary']['high_confidence'] += 1
                confidence_label = "HIGH"
            elif confidence >= 0.4:
                results['summary']['medium_confidence'] += 1
                confidence_label = "MEDIUM"
            else:
                results['summary']['low_confidence'] += 1
                confidence_label = "LOW"
            
            print(f"[{confidence_label:6}] {category.upper():9} | {pdf_file.name}")
            print(f"         Confidence: {confidence:.1%} | Keywords: {', '.join(matches[:3])}")
            
            # Move file if not dry run
            if not dry_run:
                try:
                    target_path = getattr(self, f"{category}_dir") / pdf_file.name
                    shutil.move(str(pdf_file), str(target_path))
                    print(f"         -> Moved to {category}/")
                except Exception as e:
                    print(f"         -> ERROR: Failed to move file: {e}")
            else:
                print(f"         -> Would move to {category}/")
            
            print()
        
        return results
    
    def print_summary_report(self, results: Dict[str, List[Dict]]):
        """Print a comprehensive summary report"""
        print("="*60)
        print("SORTING SUMMARY REPORT")
        print("="*60)
        
        for category in ['checklist', 'sop', 'modeling']:
            files = results.get(category, [])
            print(f"\n📁 {category.upper()} FOLDER ({len(files)} files):")
            print(f"   {self.classification_rules[category]['description']}")
            
            if files:
                # Show high-confidence files
                high_conf_files = [f for f in files if f['confidence'] >= 0.7]
                if high_conf_files:
                    print(f"   HIGH CONFIDENCE ({len(high_conf_files)}):")
                    for f in high_conf_files[:5]:  # Show first 5
                        print(f"     • {f['filename']}")
                
                # Show medium/low confidence files that might need review
                review_files = [f for f in files if f['confidence'] < 0.7]
                if review_files:
                    print(f"   REVIEW SUGGESTED ({len(review_files)}):")
                    for f in review_files[:3]:  # Show first 3
                        print(f"     • {f['filename']} (confidence: {f['confidence']:.1%})")
        
        print(f"\n📊 CONFIDENCE DISTRIBUTION:")
        summary = results.get('summary', {})
        total = summary.get('total_files', 0)
        print(f"   High Confidence (≥70%): {summary.get('high_confidence', 0)}/{total}")
        print(f"   Medium Confidence (40-69%): {summary.get('medium_confidence', 0)}/{total}")
        print(f"   Low Confidence (<40%): {summary.get('low_confidence', 0)}/{total}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Review low-confidence classifications")
        print(f"   2. Manually adjust any misclassified files")
        print(f"   3. Run the S3 one-click builder with sorted folders")
        
        print("="*60)

def main():
    """Main function to run the PDF sorter"""
    
    sorter = IntelligentPDFSorter()
    
    # First run as dry-run to preview results
    print("PREVIEW MODE - Analyzing PDF classifications...")
    print("="*60)
    
    try:
        preview_results = sorter.sort_pdfs(dry_run=True)
        sorter.print_summary_report(preview_results)
        
        print(f"\n🤔 REVIEW CLASSIFICATION RESULTS ABOVE")
        print(f"   Run with --execute to actually move files")
        print(f"   Or modify classification rules if needed")
        
        # Ask for confirmation to execute
        response = input(f"\nDo you want to execute the sorting? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            print(f"\n🚀 EXECUTING SORT...")
            print("="*60)
            actual_results = sorter.sort_pdfs(dry_run=False)
            
            print(f"\n✅ SORTING COMPLETE!")
            print(f"   Files organized into knowledge_base/sop/, /checklist/, /modeling/")
            print(f"   Ready for S3 one-click domainRAG builder!")
            
        else:
            print(f"\n⏸️  SORTING CANCELLED")
            print(f"   Files remain in original location")
            print(f"   Modify script if classifications need adjustment")
    
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    
    # Check for execute flag
    if "--execute" in sys.argv:
        sorter = IntelligentPDFSorter()
        results = sorter.sort_pdfs(dry_run=False)
        sorter.print_summary_report(results)
    else:
        main()