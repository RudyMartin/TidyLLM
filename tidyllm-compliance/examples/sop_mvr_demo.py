#!/usr/bin/env python3
"""
SOP-Guided MVR Analysis Demo
============================

Demonstrates the SOP Golden Answers system for MVR workflow compliance.
This shows how SOP guidance takes precedence over general compliance rules.

Part of tidyllm-compliance: Building actual SOP functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tidyllm_compliance.sop_golden_answers import SOPValidator

def demonstrate_sop_guided_mvr_workflow():
    """Demonstrate complete SOP-guided MVR analysis workflow."""
    
    print("SOP-GUIDED MVR ANALYSIS WORKFLOW DEMO")
    print("=" * 60)
    print("Demonstrates SOP Golden Answers precedence in MVR compliance\n")
    
    # Initialize SOP validator
    sop_validator = SOPValidator()
    
    # Simulate MVR workflow stages with SOP guidance
    workflow_stages = [
        {
            'stage': 'mvr_tag',
            'question': 'How should I classify and tag this MVR document?',
            'context': {
                'workflow_stage': 'mvr_tag',
                'document_text': 'REV12345 Motor Vehicle Record for John Doe...',
                'completed_checklist_items': ['REV00000 format ID extracted', 'Document type classified (MVR/VST)']
            }
        },
        {
            'stage': 'mvr_qa', 
            'question': 'How do I perform MVR vs VST comparison?',
            'context': {
                'workflow_stage': 'mvr_qa',
                'document_text': 'MVR document with matching VST REV12345...',
                'completed_checklist_items': ['REV00000 metadata matching confirmed', 'Section-by-section comparison completed']
            }
        },
        {
            'stage': 'mvr_peer',
            'question': 'What is required for peer review of this analysis?',
            'context': {
                'workflow_stage': 'mvr_peer',
                'document_text': 'Analysis results from MVR QA stage...',
                'completed_checklist_items': ['Domain RAG knowledge loaded', 'MVR text analysis completed']
            }
        },
        {
            'stage': 'mvr_report',
            'question': 'How should I generate the final MVR report?',
            'context': {
                'workflow_stage': 'mvr_report', 
                'document_text': 'Final analysis ready for reporting...',
                'completed_checklist_items': ['Comprehensive markdown report created', 'Formatted PDF report generated']
            }
        }
    ]
    
    # Process each workflow stage
    for stage_demo in workflow_stages:
        print(f"\n{'='*20} STAGE: {stage_demo['stage'].upper()} {'='*20}")
        print(f"Question: {stage_demo['question']}")
        
        # Get SOP guidance for this stage
        validation_result = sop_validator.validate_with_sop_precedence(
            stage_demo['question'], 
            stage_demo['context']
        )
        
        # Display SOP guidance
        print(f"\nSOP GUIDANCE (Confidence: {validation_result.sop_score:.1%}):")
        if validation_result.sop_answers:
            for sop_answer in validation_result.sop_answers:
                print(f"Source: {sop_answer.sop_source}")
                print(f"Answer: {sop_answer.answer}")
                
                if sop_answer.checklist_items:
                    print(f"\nRequired Checklist Items:")
                    completed_items = stage_demo['context'].get('completed_checklist_items', [])
                    for item in sop_answer.checklist_items:
                        status = "[DONE]" if item in completed_items else "[TODO]"
                        print(f"  {status} {item}")
        
        # Show compliance status
        print(f"\nCompliance Status: {validation_result.overall_compliance}")
        
        # Show recommendations
        if validation_result.recommendations:
            print(f"\nRecommendations:")
            for rec in validation_result.recommendations:
                print(f"  - {rec}")
        
        print(f"\nFallback Used: {'Yes' if validation_result.fallback_used else 'No (SOP guidance available)'}")

def demonstrate_sop_chat_interface():
    """Demonstrate SOP chat interface functionality."""
    
    print(f"\n{'='*60}")
    print("SOP CHAT INTERFACE DEMO")
    print("=" * 60)
    print("Simulates chat with SOP during MVR analysis\n")
    
    sop_validator = SOPValidator()
    
    # Simulate chat questions during MVR analysis
    chat_scenarios = [
        {
            'question': 'What if the REV numbers don\'t match between MVR and VST?',
            'context': {
                'workflow_stage': 'mvr_qa',
                'document_text': 'MVR REV12345 vs VST REV12346',
                'completed_checklist_items': []
            }
        },
        {
            'question': 'How do I handle documents with high noise factor?',
            'context': {
                'workflow_stage': 'mvr_tag',
                'document_text': 'Document with YNSR noise factor 0.8',
                'completed_checklist_items': ['YNSR noise analysis completed']
            }
        },
        {
            'question': 'What constitutes sufficient peer review consensus?',
            'context': {
                'workflow_stage': 'mvr_peer',
                'document_text': 'Three analysis sources with partial disagreement',
                'completed_checklist_items': ['MVR text analysis completed', 'Digest review analysis completed']
            }
        }
    ]
    
    for i, scenario in enumerate(chat_scenarios, 1):
        print(f"\nCHAT SCENARIO {i}:")
        print(f"Analyst Question: {scenario['question']}")
        
        # Get SOP chat response
        chat_response = sop_validator.chat_with_sop(scenario['question'], scenario['context'])
        
        print(f"\nSOP Response (Confidence: {chat_response['confidence']:.1%}):")
        print(f"{chat_response['sop_guidance']}")
        
        print(f"\nCompliance Status: {chat_response['compliance_status']}")
        
        if chat_response['recommendations']:
            print(f"\nRecommendations:")
            for rec in chat_response['recommendations']:
                print(f"  - {rec}")
        
        if chat_response['checklist_items']:
            print(f"\nRelevant Checklist Items:")
            for item in chat_response['checklist_items'][:3]:  # Show first 3
                print(f"  - {item}")

def demonstrate_stage_requirements():
    """Demonstrate getting SOP requirements for each workflow stage."""
    
    print(f"\n{'='*60}")
    print("WORKFLOW STAGE SOP REQUIREMENTS")
    print("=" * 60)
    print("Complete SOP requirements for each MVR workflow stage\n")
    
    sop_validator = SOPValidator()
    stages = ['mvr_tag', 'mvr_qa', 'mvr_peer', 'mvr_report']
    
    for stage in stages:
        print(f"\n{'='*15} {stage.upper()} REQUIREMENTS {'='*15}")
        
        requirements = sop_validator.get_workflow_stage_requirements(stage)
        
        print(f"SOP Answers Available: {len(requirements['sop_answers'])}")
        print(f"Total Checklist Items: {len(requirements['checklist_items'])}")
        
        if requirements['checklist_items']:
            print(f"\nChecklist Items for {stage}:")
            for item in requirements['checklist_items']:
                print(f"  [ ] {item}")

def main():
    """Run complete SOP MVR demo."""
    demonstrate_sop_guided_mvr_workflow()
    demonstrate_sop_chat_interface()
    demonstrate_stage_requirements()
    
    print(f"\n{'='*60}")
    print("SOP MVR ANALYSIS DEMO COMPLETE")
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("  [DONE] SOP Golden Answers with precedence logic")
    print("  [DONE] Workflow stage-specific guidance")
    print("  [DONE] Interactive SOP chat interface") 
    print("  [DONE] Checklist-driven compliance checking")
    print("  [DONE] Integration with existing compliance validators")
    print("\nThis demonstrates the actual SOP functionality needed")
    print("for the MVR analysis workflow with chat interface!")

if __name__ == "__main__":
    main()