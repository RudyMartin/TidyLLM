#!/usr/bin/env python3
"""
TidyLLM Backend Quality Assessment
=================================
Comprehensive testing of backend processes and components
"""
import sys
import os
import time
import json
import traceback
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_backend_quality():
    """Run comprehensive backend quality tests."""
    
    print("="*60)
    print("TIDYLLM BACKEND QUALITY ASSESSMENT")
    print("="*60)
    print(f"Test started: {datetime.now()}")
    print()
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "overall_score": 0,
        "recommendations": []
    }
    
    # Test 1: API Response Quality
    print("1. API RESPONSE QUALITY TEST")
    print("-" * 40)
    
    try:
        import tidyllm
        
        # Test response consistency
        responses = []
        for i in range(3):
            response = tidyllm.chat("What is 2+2? Answer briefly.")
            responses.append(response)
            print(f"Response {i+1}: {response[:100]}...")
        
        # Check consistency
        consistent = all("4" in str(r) or "four" in str(r).lower() for r in responses)
        test_results["tests"]["api_response_quality"] = {
            "score": 10 if consistent else 5,
            "consistent_responses": consistent,
            "response_count": len(responses)
        }
        print(f"✓ Response Consistency: {'PASS' if consistent else 'PARTIAL'}")
        
    except Exception as e:
        test_results["tests"]["api_response_quality"] = {"score": 0, "error": str(e)}
        print(f"✗ API Response Quality: FAIL - {e}")
    
    # Test 2: Gateway System Performance
    print("\n2. GATEWAY SYSTEM PERFORMANCE")
    print("-" * 40)
    
    try:
        import tidyllm
        
        # Measure response times
        times = []
        for i in range(5):
            start = time.time()
            tidyllm.chat("Hello")
            end = time.time()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Score based on performance (< 2s excellent, < 5s good, > 5s poor)
        if avg_time < 2:
            perf_score = 10
            perf_rating = "EXCELLENT"
        elif avg_time < 5:
            perf_score = 8
            perf_rating = "GOOD"
        else:
            perf_score = 5
            perf_rating = "ADEQUATE"
        
        test_results["tests"]["gateway_performance"] = {
            "score": perf_score,
            "avg_response_time": avg_time,
            "max_response_time": max_time,
            "rating": perf_rating
        }
        
        print(f"✓ Average Response Time: {avg_time:.2f}s ({perf_rating})")
        print(f"✓ Max Response Time: {max_time:.2f}s")
        
    except Exception as e:
        test_results["tests"]["gateway_performance"] = {"score": 0, "error": str(e)}
        print(f"✗ Gateway Performance: FAIL - {e}")
    
    # Test 3: Error Handling Quality
    print("\n3. ERROR HANDLING QUALITY")
    print("-" * 40)
    
    try:
        import tidyllm
        
        error_tests = [
            ("missing_file", lambda: tidyllm.process_document("nonexistent.pdf", "test")),
            ("empty_prompt", lambda: tidyllm.chat("")),
            ("invalid_model", lambda: tidyllm.set_model("invalid-model-name"))
        ]
        
        error_handling_score = 0
        for test_name, test_func in error_tests:
            try:
                result = test_func()
                if "ERROR" in str(result) or "error" in str(result).lower():
                    error_handling_score += 3
                    print(f"✓ {test_name}: Graceful error handling")
                else:
                    print(f"~ {test_name}: Unexpected response: {result[:50]}...")
            except Exception as e:
                error_handling_score += 2
                print(f"~ {test_name}: Exception raised: {str(e)[:50]}...")
        
        test_results["tests"]["error_handling"] = {
            "score": min(10, error_handling_score),
            "tests_run": len(error_tests)
        }
        
    except Exception as e:
        test_results["tests"]["error_handling"] = {"score": 0, "error": str(e)}
        print(f"✗ Error Handling: FAIL - {e}")
    
    # Test 4: Configuration Management
    print("\n4. CONFIGURATION MANAGEMENT")
    print("-" * 40)
    
    try:
        import tidyllm
        
        status = tidyllm.status()
        
        config_checks = {
            "architecture": status.get("architecture") == "gateway_compliant",
            "audit_mode": status.get("audit_mode") is True,
            "compliance_mode": status.get("compliance_mode") is True,
            "aws_integration": status.get("has_aws_key") is True,
            "models_available": len(status.get("available_models", [])) > 0
        }
        
        config_score = sum(config_checks.values()) * 2
        
        test_results["tests"]["configuration"] = {
            "score": config_score,
            "checks": config_checks
        }
        
        for check, passed in config_checks.items():
            print(f"{'✓' if passed else '✗'} {check}: {'PASS' if passed else 'FAIL'}")
        
    except Exception as e:
        test_results["tests"]["configuration"] = {"score": 0, "error": str(e)}
        print(f"✗ Configuration Management: FAIL - {e}")
    
    # Test 5: Data Processing Quality
    print("\n5. DATA PROCESSING QUALITY")
    print("-" * 40)
    
    try:
        import tidyllm
        
        # Create test document
        test_content = """
        BUSINESS REPORT: Q4 Financial Results
        
        Revenue: $1.5M (up 25% from Q3)
        Profit: $300K (20% margin)
        Employees: 150 (grew by 15%)
        Customer Satisfaction: 92%
        
        Key Achievements:
        - Launched new product line
        - Expanded to European market
        - Achieved ISO certification
        """
        
        with open('test_business_report.txt', 'w') as f:
            f.write(test_content)
        
        # Test document processing
        summary = tidyllm.process_document('test_business_report.txt', 'Extract the key financial metrics')
        
        # Check if AI extracted key information
        key_info_found = sum([
            "1.5" in summary or "1.5M" in summary,
            "25%" in summary,
            "300" in summary or "300K" in summary,
            "20%" in summary,
            "150" in summary,
            "92%" in summary
        ])
        
        processing_score = min(10, key_info_found * 2)
        
        test_results["tests"]["data_processing"] = {
            "score": processing_score,
            "key_info_extracted": key_info_found,
            "total_key_points": 6
        }
        
        print(f"✓ Key Information Extracted: {key_info_found}/6")
        print(f"✓ Processing Quality: {'EXCELLENT' if processing_score >= 8 else 'GOOD' if processing_score >= 6 else 'ADEQUATE'}")
        
        # Cleanup
        os.remove('test_business_report.txt')
        
    except Exception as e:
        test_results["tests"]["data_processing"] = {"score": 0, "error": str(e)}
        print(f"✗ Data Processing: FAIL - {e}")
    
    # Calculate Overall Score
    total_score = sum(test.get("score", 0) for test in test_results["tests"].values())
    max_score = len(test_results["tests"]) * 10
    overall_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
    
    test_results["overall_score"] = overall_percentage
    test_results["total_points"] = f"{total_score}/{max_score}"
    
    # Generate Recommendations
    recommendations = []
    
    if test_results["tests"].get("api_response_quality", {}).get("score", 0) < 8:
        recommendations.append("Improve API response consistency")
    
    if test_results["tests"].get("gateway_performance", {}).get("score", 0) < 8:
        recommendations.append("Optimize gateway response times")
    
    if test_results["tests"].get("error_handling", {}).get("score", 0) < 8:
        recommendations.append("Enhance error handling and validation")
    
    if test_results["tests"].get("configuration", {}).get("score", 0) < 8:
        recommendations.append("Review configuration management")
    
    if test_results["tests"].get("data_processing", {}).get("score", 0) < 8:
        recommendations.append("Improve document processing accuracy")
    
    test_results["recommendations"] = recommendations
    
    # Final Report
    print("\n" + "="*60)
    print("BACKEND QUALITY ASSESSMENT RESULTS")
    print("="*60)
    
    print(f"Overall Score: {overall_percentage:.1f}% ({total_score}/{max_score} points)")
    
    if overall_percentage >= 90:
        quality_rating = "EXCELLENT"
    elif overall_percentage >= 80:
        quality_rating = "GOOD"
    elif overall_percentage >= 70:
        quality_rating = "ADEQUATE"
    else:
        quality_rating = "NEEDS IMPROVEMENT"
    
    print(f"Quality Rating: {quality_rating}")
    
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    else:
        print("\nNo major issues identified - backend quality is excellent!")
    
    # Save detailed report
    report_file = f"backend_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nDetailed report saved: {report_file}")
    
    return test_results

if __name__ == "__main__":
    try:
        results = test_backend_quality()
        exit_code = 0 if results["overall_score"] >= 70 else 1
        sys.exit(exit_code)
    except Exception as e:
        print(f"CRITICAL ERROR: Backend quality test failed: {e}")
        traceback.print_exc()
        sys.exit(1)