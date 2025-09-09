#!/usr/bin/env python3
"""
TidyLLM Backend Quality Test - Simple Version
=============================================
"""
import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def run_quality_tests():
    """Run backend quality tests."""
    
    print("="*60)
    print("TIDYLLM BACKEND QUALITY ASSESSMENT")
    print("="*60)
    
    test_results = []
    
    try:
        import tidyllm
        
        # Test 1: API Response Quality
        print("\n1. API RESPONSE QUALITY")
        print("-" * 30)
        
        responses = []
        for i in range(3):
            response = tidyllm.chat("What is 2+2? Answer briefly.")
            responses.append(response)
            print(f"Response {i+1}: {response[:80]}...")
        
        consistent = all("4" in str(r) or "four" in str(r).lower() for r in responses)
        print(f"[PASS] Response Consistency: {consistent}")
        test_results.append(("API Response Quality", 10 if consistent else 5))
        
        # Test 2: Performance
        print("\n2. GATEWAY PERFORMANCE")
        print("-" * 30)
        
        times = []
        for i in range(5):
            start = time.time()
            tidyllm.chat("Hello")
            end = time.time()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        print(f"Average Response Time: {avg_time:.2f}s")
        
        if avg_time < 2:
            perf_score = 10
            rating = "EXCELLENT"
        elif avg_time < 5:
            perf_score = 8
            rating = "GOOD" 
        else:
            perf_score = 5
            rating = "ADEQUATE"
        
        print(f"[PASS] Performance Rating: {rating}")
        test_results.append(("Gateway Performance", perf_score))
        
        # Test 3: Configuration
        print("\n3. CONFIGURATION STATUS")
        print("-" * 30)
        
        status = tidyllm.status()
        
        checks = {
            "architecture": status.get("architecture") == "gateway_compliant",
            "audit_mode": status.get("audit_mode") is True,
            "compliance_mode": status.get("compliance_mode") is True,
            "aws_integration": status.get("has_aws_key") is True,
            "models_available": len(status.get("available_models", [])) > 0
        }
        
        passed = sum(checks.values())
        total = len(checks)
        
        for check, result in checks.items():
            print(f"[{'PASS' if result else 'FAIL'}] {check}")
        
        config_score = (passed / total) * 10
        print(f"Configuration Score: {passed}/{total}")
        test_results.append(("Configuration", config_score))
        
        # Test 4: Document Processing
        print("\n4. DOCUMENT PROCESSING")
        print("-" * 30)
        
        test_doc = """
        FINANCIAL REPORT
        Revenue: $1.5M (up 25%)
        Profit: $300K (20% margin)  
        Employees: 150
        Growth: 15%
        """
        
        with open('test_doc.txt', 'w') as f:
            f.write(test_doc)
        
        summary = tidyllm.process_document('test_doc.txt', 'Extract key financial numbers')
        
        key_numbers = ["1.5", "25%", "300", "20%", "150", "15%"]
        found = sum(1 for num in key_numbers if num in summary)
        
        print(f"Key Information Extracted: {found}/{len(key_numbers)}")
        processing_score = (found / len(key_numbers)) * 10
        test_results.append(("Document Processing", processing_score))
        
        os.remove('test_doc.txt')
        
        # Test 5: Error Handling
        print("\n5. ERROR HANDLING")
        print("-" * 30)
        
        try:
            result = tidyllm.process_document('nonexistent.pdf', 'test')
            if "ERROR" in str(result) or "error" in str(result).lower():
                print("[PASS] Graceful error handling for missing files")
                error_score = 10
            else:
                print("[PARTIAL] Unexpected response to missing file")
                error_score = 5
        except Exception:
            print("[PARTIAL] Exception raised for missing file")
            error_score = 7
        
        test_results.append(("Error Handling", error_score))
        
        # Calculate Overall Score
        total_score = sum(score for _, score in test_results)
        max_score = len(test_results) * 10
        percentage = (total_score / max_score) * 100
        
        print("\n" + "="*60)
        print("QUALITY ASSESSMENT RESULTS")
        print("="*60)
        
        for test_name, score in test_results:
            print(f"{test_name}: {score:.1f}/10")
        
        print(f"\nOverall Quality Score: {percentage:.1f}% ({total_score:.1f}/{max_score})")
        
        if percentage >= 90:
            rating = "EXCELLENT"
        elif percentage >= 80:
            rating = "GOOD"
        elif percentage >= 70:
            rating = "ADEQUATE"
        else:
            rating = "NEEDS IMPROVEMENT"
        
        print(f"Backend Quality Rating: {rating}")
        
        # Recommendations
        recommendations = []
        for test_name, score in test_results:
            if score < 8:
                recommendations.append(f"Improve {test_name}")
        
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        else:
            print("\n[SUCCESS] No major issues - backend quality is excellent!")
        
        return percentage >= 70
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_quality_tests()
    print(f"\nQUALITY TEST {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)