"""
Simple Documents Module Demo - No Unicode
=========================================

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
    print(f"ERROR: Documents module not available: {e}")

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
    print(f"\nV1 Features:")
    v1_features = capabilities.get('v1_features', {})
    for feature, status in v1_features.items():
        status_text = "AVAILABLE" if status else "NOT AVAILABLE"
        print(f"  {feature.replace('_', ' ').title()}: {status_text}")
    
    print("\n" + "="*60)
    print("V2 DOCUMENTS MODULE DEMO COMPLETE")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)