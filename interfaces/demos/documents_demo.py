"""
Simple Documents Module Demo
============================

Test the V2 documents module with V1 integration.
Clean, simple architecture without complex async patterns.
"""

import sys
import json
from pathlib import Path

# Add the v2 directory to path so we can import documents
sys.path.append(str(Path(__file__).parent))

try:
    from documents import process_document, process_image_corporate, get_processing_capabilities
    DOCUMENTS_AVAILABLE = True
except ImportError as e:
    DOCUMENTS_AVAILABLE = False
    print(f"❌ Documents module not available: {e}")

def main():
    """Demo the documents module capabilities."""
    print("="*60)
    print("V2 DOCUMENTS MODULE DEMO")
    print("="*60)
    
    if not DOCUMENTS_AVAILABLE:
        print("ERROR: Documents module not available. Check imports.")
        return 1
    
    # Show capabilities
    print("\nPROCESSING CAPABILITIES:")
    print("-" * 30)
    capabilities = get_processing_capabilities()
    
    for category, details in capabilities.items():
        if isinstance(details, dict):
            print(f"\n{category.replace('_', ' ').title()}:")
            for key, value in details.items():
                status = "YES" if value else "NO"
                print(f"  {status} {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"{category.replace('_', ' ').title()}: {details}")
    
    print(f"\nArchitecture: {capabilities.get('architecture', 'unknown')}")
    
    # Test document processing if we have a sample file
    sample_files = [
        Path("sample.pdf"),
        Path("test.pdf"),
        Path("document.pdf"),
        Path("../test.pdf"),
        Path("../../test.pdf")
    ]
    
    sample_file = None
    for file_path in sample_files:
        if file_path.exists():
            sample_file = file_path
            break
    
    if sample_file:
        print(f"\nTESTING DOCUMENT PROCESSING:")
        print(f"Sample file: {sample_file}")
        print("-" * 30)
        
        try:
            result = process_document(sample_file, max_pages=3, create_chunks=True)
            
            print(f"Success: {result.success}")
            print(f"Processing time: {result.processing_time:.3f}s")
            print(f"Text length: {len(result.text):,} characters")
            print(f"Chunks created: {len(result.chunks)}")
            
            if result.classification:
                print(f"Document type: {result.classification.get('document_type', 'unknown')}")
                print(f"Confidence: {result.classification.get('confidence', 0):.1%}")
                
                keywords = result.classification.get('matched_keywords', [])
                if keywords:
                    print(f"Keywords: {', '.join(keywords[:5])}")
            
            if result.text:
                preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
                print(f"\nText preview:")
                print(f"    {preview}")
            
            if result.chunks:
                print(f"\nFirst chunk preview:")
                first_chunk = result.chunks[0]['text']
                chunk_preview = first_chunk[:150] + "..." if len(first_chunk) > 150 else first_chunk
                print(f"    {chunk_preview}")
            
            if result.warnings:
                print(f"\nWarnings:")
                for warning in result.warnings:
                    print(f"    {warning}")
            
            if result.errors:
                print(f"\nErrors:")
                for error in result.errors:
                    print(f"    {error}")
            
        except Exception as e:
            print(f"Document processing failed: {e}")
    
    else:
        print(f"\nNo sample PDF files found for testing")
        print("   Create a test.pdf file to see document processing in action")
    
    # Test image processing if we have a sample image
    image_files = [
        Path("sample.png"),
        Path("test.jpg"),
        Path("image.png"),
        Path("../test.jpg"),
        Path("../../sample.png")
    ]
    
    sample_image = None
    for img_path in image_files:
        if img_path.exists():
            sample_image = img_path
            break
    
    if sample_image:
        print(f"\n🖼️ TESTING IMAGE PROCESSING:")
        print(f"Sample image: {sample_image}")
        print("-" * 30)
        
        try:
            img_result = process_image_corporate(
                sample_image, 
                convert_to_jpeg=True,
                create_html=True,
                create_ascii=False
            )
            
            if img_result.get("success"):
                print("✅ Image processing successful")
                
                converted = img_result.get("converted_files", [])
                if converted:
                    print(f"📸 Converted to JPEG: {len(converted)} files")
                    for conv_file in converted[:3]:
                        print(f"    {Path(conv_file).name}")
                
                html_file = img_result.get("html_file")
                if html_file:
                    print(f"🌐 HTML embedding created: {Path(html_file).name}")
                
                ascii_file = img_result.get("ascii_file")
                if ascii_file:
                    print(f"🎨 ASCII art created: {Path(ascii_file).name}")
            
            else:
                error = img_result.get("error", "Unknown error")
                print(f"❌ Image processing failed: {error}")
        
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
    
    else:
        print(f"\n🖼️ No sample image files found for testing")
        print("   Create a test.jpg or sample.png file to see image processing in action")
    
    print(f"\n" + "="*60)
    print("✨ V2 DOCUMENTS MODULE DEMO COMPLETE")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)