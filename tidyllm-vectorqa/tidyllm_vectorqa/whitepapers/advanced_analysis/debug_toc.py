#!/usr/bin/env python3
"""Debug TOC extraction to see what's happening with PDF parsing."""

from toc_analyzer import TOCAnalyzer
from pathlib import Path

def debug_single_paper():
    analyzer = TOCAnalyzer()
    
    # Find a paper to debug
    paper_repo = Path(__file__).parent.parent / "paper_repository"
    pdf_files = list(paper_repo.glob("**/*.pdf"))
    
    if pdf_files:
        test_paper = pdf_files[0]
        print(f"Debugging: {test_paper.name}")
        print("=" * 50)
        
        # Extract text
        full_text, pages = analyzer.extract_pdf_text_structured(str(test_paper))
        
        print(f"Extracted {len(pages)} pages")
        print(f"Total text length: {len(full_text)}")
        
        if pages:
            print("\nFirst page content (first 500 chars):")
            print("-" * 30)
            print(pages[0][:500])
            print("-" * 30)
            
            # Try to detect TOC
            toc_text = analyzer.detect_toc_section(full_text, pages)
            print(f"\nTOC detection result: {'Found' if toc_text else 'Not found'}")
            
            if toc_text:
                print("TOC text (first 300 chars):")
                print(toc_text[:300])
                print("-" * 30)
                
                # Parse entries
                entries = analyzer.parse_toc_entries(toc_text)
                print(f"\nParsed {len(entries)} entries:")
                for entry in entries:
                    print(f"  L{entry.level}: {entry.title} ({entry.content_type})")
            else:
                print("No TOC found, trying direct parsing on first page...")
                entries = analyzer.parse_toc_entries(pages[0])
                print(f"Direct parse found {len(entries)} entries:")
                for entry in entries:
                    print(f"  L{entry.level}: {entry.title} ({entry.content_type})")
        
        print("\nFull analysis result:")
        analysis = analyzer.analyze_paper(str(test_paper))
        print(f"  Scale: {analysis.scale_assessment}")
        print(f"  Methodology depth: {analysis.methodology_depth:.2f}")
        print(f"  Organization quality: {analysis.organization_quality:.2f}")
        print(f"  Entries found: {len(analysis.entries)}")
        
    else:
        print("No PDF files found in repository")

if __name__ == "__main__":
    debug_single_paper()