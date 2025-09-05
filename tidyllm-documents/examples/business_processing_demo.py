#!/usr/bin/env python3
"""
Business Document Processing Demo

Demonstrates comprehensive business document processing capabilities including:
- Document classification with confidence scoring
- Metadata extraction using business patterns
- Template-based processing for common document types

Part of the tidyllm-verse: Educational ML with complete transparency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tidyllm_documents import BusinessDocumentProcessor, TextExtractor, MetadataExtractor

def create_sample_documents():
    """Create sample business documents for demonstration."""
    return {
        "sample_invoice.txt": """
        INVOICE #INV-2024-001
        
        Bill To:
        Acme Corporation
        123 Business Ave
        City, ST 12345
        
        From:
        Professional Services LLC
        456 Service Road
        Town, ST 67890
        Email: billing@proservices.com
        Phone: (555) 123-4567
        
        Date: January 15, 2024
        Due Date: February 15, 2024
        
        Description                    Amount
        Consulting Services            $2,500.00
        Project Management             $1,500.00
        
        Total Amount Due: $4,000.00
        
        Payment Terms: Net 30
        Account Number: 1234567890
        """,
        
        "sample_contract.txt": """
        SERVICE AGREEMENT #SA-2024-001
        
        This Agreement is entered into on January 1, 2024
        between Client Corp (Client) and Service Provider Inc (Provider).
        
        Contract Number: SA-2024-001
        
        TERMS:
        1. Services to be provided: Software Development
        2. Contract Value: $50,000
        3. Start Date: January 1, 2024
        4. End Date: December 31, 2024
        5. Payment Schedule: Monthly
        
        Client Contact: John Smith
        Email: john.smith@clientcorp.com
        Phone: (555) 987-6543
        
        Provider Contact: Jane Doe
        Email: jane.doe@provider.com
        Phone: (555) 456-7890
        
        Reference Number: REF-2024-001
        
        Signatures:
        Client: _________________
        Provider: _______________
        """,
        
        "sample_purchase_order.txt": """
        PURCHASE ORDER PO-2024-5678
        
        Vendor: Office Supplies Plus
        456 Supply Street
        Business Park, TX 75001
        Email: orders@officesupplies.com
        Phone: (555) 345-6789
        
        Ship To: Corporate Headquarters
        789 Company Blvd
        Metro City, NY 10001
        
        Purchase Date: January 8, 2024
        Required Delivery Date: January 22, 2024
        
        Items:
        - Office Chairs (10) - $2,500.00
        - Desk Supplies - $150.00
        - Printer Paper - $75.00
        
        Subtotal: $2,725.00
        Tax: $218.00
        Total: $2,943.00
        
        Account: 9876543210
        Reference: REF-FURNITURE-001
        
        Authorized By: Jane Manager
        Email: jane.manager@company.com
        """
    }

def main():
    """Demonstrate business document processing capabilities."""
    print("BUSINESS DOCUMENT PROCESSING DEMONSTRATION")
    print("=" * 70)
    print("Comprehensive processing of common business document types")
    print("Part of the tidyllm-verse: Educational ML with complete transparency\n")
    
    # Initialize processor
    processor = BusinessDocumentProcessor()
    
    # Show template summary
    template_summary = processor.get_template_summary()
    print(f"Available Templates: {template_summary['total_templates']}")
    for template_name, details in template_summary['template_details'].items():
        print(f"  - {template_name}: {details['keywords_count']} keywords, "
              f"{details['required_patterns_count']} required patterns")
    
    print(f"\nSupported Formats: {processor.get_supported_formats()}\n")
    
    # Create sample documents
    sample_documents = create_sample_documents()
    
    # Process each document
    for filename, content in sample_documents.items():
        print(f"Processing: {filename}")
        print("-" * 40)
        
        # Save temporary file
        temp_path = f"temp_{filename}"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Process document
            result = processor.process_document(temp_path)
            
            # Display results
            print(f"Document Type: {result['document_type']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Template Used: {result['template_used']}")
            print(f"Text Length: {result['text_length']} characters")
            
            if result['extracted_fields']:
                print("Extracted Fields:")
                for field, value in result['extracted_fields'].items():
                    print(f"  {field}: {value}")
            
            # Show detailed metadata with confidence
            if result['detailed_metadata']:
                print("Detailed Metadata (with confidence):")
                for field, details in result['detailed_metadata'].items():
                    print(f"  {field}: {details['value']} (confidence: {details['confidence']:.3f})")
            
            print()
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Demonstrate text-only processing
    print("TEXT-ONLY PROCESSING EXAMPLE")
    print("-" * 40)
    
    sample_text = """
    MONTHLY STATEMENT
    Account Number: 1122334455667788
    Statement Date: December 31, 2023
    
    Beginning Balance: $5,678.90
    Ending Balance: $6,143.76
    Total Deposits: $1,750.00
    Total Withdrawals: $1,285.14
    
    Contact: statements@bank.com
    Phone: (555) 111-2222
    """
    
    # Extract text directly
    text_extractor = TextExtractor()
    metadata_extractor = MetadataExtractor()
    
    # Classify the text
    from tidyllm_documents import DocumentClassifier
    classifier = DocumentClassifier(['financial_statement', 'invoice', 'report'])
    
    # Process text directly
    text_result = classifier.classify_text(sample_text)
    metadata_result = metadata_extractor.extract_with_confidence(sample_text)
    
    print(f"Text Classification: {text_result['category']} (confidence: {text_result['confidence']:.3f})")
    print("Extracted Metadata:")
    for field, details in metadata_result.items():
        print(f"  {field}: {details['value']} (confidence: {details['confidence']:.3f})")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("Ready for production business document processing!")
    print("Features demonstrated:")
    print("- Multi-format document processing (PDF, DOCX, TXT)")
    print("- Business template-based classification")
    print("- Confidence-scored metadata extraction")
    print("- Pattern-based field validation")
    print("- Pure Python implementation with complete transparency")

if __name__ == "__main__":
    main()