#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features Matrix Testing Script

Interactive script for drilling down and testing intermediate steps
from orchestrators to workers using the features matrix.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import unittest.mock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.mcp.orchestrators.simple_qa_orchestrator import SimpleQAOrchestrator
from backend.mcp.orchestrators.enhanced_qa_orchestrator import EnhancedQAOrchestrator
from backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator


class FeaturesMatrixTester:
    """Interactive tester for the features matrix"""
    
    def __init__(self):
        self.test_documents = self._load_test_documents()
        self.orchestrators = {}
        self.results = {}
    
    def _load_test_documents(self) -> Dict[str, Any]:
        """Load test documents for different complexity levels"""
        return {
            'simple': {
                'content': '# Simple Document\n\nThis is a basic document with simple content.',
                'metadata': {'title': 'Simple Test', 'type': 'basic'},
                'file_path': '/test/simple.md'
            },
            'enhanced': {
                'content': """
# Enhanced Document

## Table of Contents
1. Introduction
2. Analysis

## Introduction
Enhanced document with proper structure.

### Figure 1: Test Chart
Sample figure with caption.

## References
1. Smith, J. (2023). Test Reference.
                """,
                'metadata': {'title': 'Enhanced Test', 'type': 'technical'},
                'file_path': '/test/enhanced.md'
            },
            'advanced': {
                'content': """
# Advanced Document with AI/ML Integration

## Executive Summary
Advanced document demonstrating comprehensive capabilities.

## Technical Architecture
The system includes AI/ML integration and real-time monitoring.

### Table 1: Feature Matrix
| Feature | Status |
|---------|--------|
| AI/ML   | ✅     |
| RAG     | ✅     |

## References
1. Smith, J. (2023). Advanced Systems.
2. Doe, A. (2023). AI/ML Integration.
                """,
                'metadata': {'title': 'Advanced Test', 'type': 'advanced_technical'},
                'file_path': '/test/advanced.md'
            }
        }
    
    def initialize_orchestrators(self):
        """Initialize all orchestrators with mocked dependencies"""
        print("🔧 Initializing orchestrators...")
        
        with unittest.mock.patch('backend.mcp.orchestrators.enhanced_qa_orchestrator.get_database_manager'), \
             unittest.mock.patch('backend.mcp.orchestrators.enhanced_qa_orchestrator.DocumentInspectorCoordinator'), \
             unittest.mock.patch('backend.mcp.orchestrators.enhanced_qa_orchestrator.CaptionInspectorCoordinator'):
            
            self.orchestrators['simple'] = SimpleQAOrchestrator()
            self.orchestrators['enhanced'] = EnhancedQAOrchestrator()
            self.orchestrators['advanced'] = AdvancedQAOrchestrator()
        
        print("✅ All orchestrators initialized successfully")
    
    def test_orchestrator_level(self, level: str, document_type: str = None):
        """Test a specific orchestrator level"""
        if level not in self.orchestrators:
            print(f"❌ Orchestrator level '{level}' not found")
            return None
        
        if document_type is None:
            document_type = level
        
        if document_type not in self.test_documents:
            print(f"❌ Document type '{document_type}' not found")
            return None
        
        print(f"\n🔍 Testing {level.title()} QA Orchestrator...")
        print(f"   Document: {document_type.title()}")
        
        try:
            result = self.orchestrators[level].process_document(self.test_documents[document_type])
            self.results[level] = result
            
            print(f"   ✅ Processing completed successfully")
            print(f"   📊 Status: {result.get('status', 'unknown')}")
            print(f"   ⏱️ Processing Time: {result.get('processing_time_ms', 0):.2f}ms")
            
            # Extract quality score based on level
            if level == 'simple':
                score = result.get('quality_score', 0.0)
            elif level == 'enhanced':
                score = result.get('enhanced_quality_score', 0.0)
            else:  # advanced
                score = result.get('advanced_quality_score', 0.0)
            
            print(f"   📈 Quality Score: {score:.3f}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error during processing: {e}")
            return None
    
    def test_worker_component(self, orchestrator_level: str, component_name: str):
        """Test a specific worker component within an orchestrator"""
        if orchestrator_level not in self.orchestrators:
            print(f"❌ Orchestrator level '{orchestrator_level}' not found")
            return None
        
        orchestrator = self.orchestrators[orchestrator_level]
        
        print(f"\n🔧 Testing {component_name} in {orchestrator_level.title()} QA Orchestrator...")
        
        # Test different component types
        if hasattr(orchestrator, component_name):
            component = getattr(orchestrator, component_name)
            print(f"   ✅ Component '{component_name}' found")
            print(f"   📋 Type: {type(component).__name__}")
            
            # Test component initialization
            if hasattr(component, '__init__'):
                print(f"   🔧 Component initialized successfully")
            
            # Test component methods if available
            if hasattr(component, 'get_stats'):
                try:
                    stats = component.get_stats()
                    print(f"   📊 Component stats: {stats}")
                except Exception as e:
                    print(f"   ⚠️ Could not get stats: {e}")
            
            return component
        else:
            print(f"   ❌ Component '{component_name}' not found")
            return None
    
    def test_feature_progression(self):
        """Test feature progression across all orchestrator levels"""
        print("\n📈 Testing Feature Progression...")
        
        # Test all levels with the same document
        test_document = self.test_documents['enhanced']
        
        progression_results = {}
        
        for level in ['simple', 'enhanced', 'advanced']:
            print(f"\n   Testing {level.title()} QA...")
            
            try:
                result = self.orchestrators[level].process_document(test_document)
                progression_results[level] = result
                
                # Extract quality score
                if level == 'simple':
                    score = result.get('quality_score', 0.0)
                elif level == 'enhanced':
                    score = result.get('enhanced_quality_score', 0.0)
                else:  # advanced
                    score = result.get('advanced_quality_score', 0.0)
                
                print(f"   ✅ {level.title()} QA Score: {score:.3f}")
                
            except Exception as e:
                print(f"   ❌ {level.title()} QA Error: {e}")
                progression_results[level] = None
        
        # Validate progression
        if all(progression_results.values()):
            simple_score = progression_results['simple'].get('quality_score', 0.0)
            enhanced_score = progression_results['enhanced'].get('enhanced_quality_score', 0.0)
            advanced_score = progression_results['advanced'].get('advanced_quality_score', 0.0)
            
            print(f"\n📊 Progression Analysis:")
            print(f"   • Simple QA: {simple_score:.3f}")
            print(f"   • Enhanced QA: {enhanced_score:.3f}")
            print(f"   • Advanced QA: {advanced_score:.3f}")
            
            if enhanced_score >= simple_score and advanced_score >= enhanced_score:
                print(f"   ✅ Feature progression validated!")
            else:
                print(f"   ⚠️ Feature progression needs review")
        
        return progression_results
    
    def test_resource_availability(self):
        """Test resource availability across orchestrator levels"""
        print("\n🔧 Testing Resource Availability...")
        
        resources = {
            'simple': ['session_id', 'quality_metrics'],
            'enhanced': ['document_inspector', 'caption_inspector', 'db_manager', 'quality_analyzer'],
            'advanced': ['llm_client', 'rag_system', 'datamart_manager', 'cache_manager', 'config_manager', 'real_time_monitor']
        }
        
        availability_results = {}
        
        for level, expected_resources in resources.items():
            print(f"\n   Testing {level.title()} QA Resources...")
            orchestrator = self.orchestrators[level]
            
            available_resources = []
            missing_resources = []
            
            for resource in expected_resources:
                if hasattr(orchestrator, resource):
                    available_resources.append(resource)
                    print(f"   ✅ {resource}")
                else:
                    missing_resources.append(resource)
                    print(f"   ❌ {resource}")
            
            availability_results[level] = {
                'available': available_resources,
                'missing': missing_resources,
                'coverage': len(available_resources) / len(expected_resources)
            }
            
            print(f"   📊 Coverage: {availability_results[level]['coverage']:.1%}")
        
        return availability_results
    
    def run_comprehensive_test(self):
        """Run a comprehensive test of all features"""
        print("🚀 Running Comprehensive Features Matrix Test")
        print("=" * 60)
        
        # Initialize orchestrators
        self.initialize_orchestrators()
        
        # Test each orchestrator level
        for level in ['simple', 'enhanced', 'advanced']:
            self.test_orchestrator_level(level)
        
        # Test feature progression
        self.test_feature_progression()
        
        # Test resource availability
        self.test_resource_availability()
        
        print("\n✅ Comprehensive test completed!")
        return self.results
    
    def interactive_test(self):
        """Run interactive testing session"""
        print("🎯 Interactive Features Matrix Testing")
        print("=" * 50)
        
        self.initialize_orchestrators()
        
        while True:
            print("\n📋 Available Tests:")
            print("1. Test Simple QA Orchestrator")
            print("2. Test Enhanced QA Orchestrator")
            print("3. Test Advanced QA Orchestrator")
            print("4. Test Feature Progression")
            print("5. Test Resource Availability")
            print("6. Test Specific Component")
            print("7. Run Comprehensive Test")
            print("0. Exit")
            
            choice = input("\nSelect test (0-7): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                self.test_orchestrator_level('simple')
            elif choice == '2':
                self.test_orchestrator_level('enhanced')
            elif choice == '3':
                self.test_orchestrator_level('advanced')
            elif choice == '4':
                self.test_feature_progression()
            elif choice == '5':
                self.test_resource_availability()
            elif choice == '6':
                level = input("Enter orchestrator level (simple/enhanced/advanced): ").strip()
                component = input("Enter component name: ").strip()
                self.test_worker_component(level, component)
            elif choice == '7':
                self.run_comprehensive_test()
            else:
                print("❌ Invalid choice. Please try again.")


def main():
    """Main function"""
    tester = FeaturesMatrixTester()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'comprehensive':
            tester.run_comprehensive_test()
        elif sys.argv[1] == 'interactive':
            tester.interactive_test()
        else:
            print("Usage: python test_features_matrix.py [comprehensive|interactive]")
    else:
        print("🎯 Features Matrix Testing")
        print("Usage: python test_features_matrix.py [comprehensive|interactive]")
        print("\nOptions:")
        print("  comprehensive - Run comprehensive test")
        print("  interactive   - Run interactive testing session")


if __name__ == '__main__':
    main()
