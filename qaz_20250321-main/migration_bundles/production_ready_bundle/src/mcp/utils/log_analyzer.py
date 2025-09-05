"""
Log Analyzer for SME Context Integration

Analyzes logs to identify patterns, issues, and areas for improvement.
Provides insights into system performance, error patterns, and fallback usage.
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import pandas as pd


class LogAnalyzer:
    """Analyzes logs to identify patterns and issues."""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = logs_dir
        self.log_files = {
            "dspy_errors": os.path.join(logs_dir, "dspy_errors.log"),
            "fallback_usage": os.path.join(logs_dir, "fallback_usage.log"),
            "context_errors": os.path.join(logs_dir, "context_errors.log"),
            "database_errors": os.path.join(logs_dir, "database_errors.log"),
            "import_errors": os.path.join(logs_dir, "import_errors.log"),
            "performance_metrics": os.path.join(logs_dir, "performance_metrics.log"),
            "sme_analysis": os.path.join(logs_dir, "sme_analysis.log")
        }
    
    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a log line and extract JSON data."""
        try:
            # Extract JSON data from log line - look for JSON after the log level
            # Format: timestamp - logger - level - message
            parts = line.split(' - ')
            if len(parts) >= 4:
                # Find the JSON part (starts with { and ends with })
                message_part = ' - '.join(parts[3:])
                json_match = re.search(r'\{.*\}', message_part, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError, IndexError):
            pass
        return None
    
    def _read_log_file(self, log_file: str) -> List[Dict[str, Any]]:
        """Read and parse a log file."""
        if not os.path.exists(log_file):
            return []
        
        parsed_entries = []
        current_json = ""
        in_json = False
        
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Check if this line starts a new log entry
                if line and not line.startswith(' ') and ' - ' in line:
                    # Parse the previous JSON if we have one
                    if in_json and current_json:
                        try:
                            parsed = json.loads(current_json)
                            parsed_entries.append(parsed)
                        except json.JSONDecodeError:
                            pass
                    
                    # Start new JSON parsing
                    current_json = ""
                    in_json = False
                    
                    # Check if this line contains JSON start
                    if '{' in line:
                        json_start = line.find('{')
                        current_json = line[json_start:]
                        in_json = True
                
                # Continue building JSON if we're in a JSON block
                elif in_json and line:
                    current_json += line
        
        # Parse the last JSON if we have one
        if in_json and current_json:
            try:
                parsed = json.loads(current_json)
                parsed_entries.append(parsed)
            except json.JSONDecodeError:
                pass
        
        return parsed_entries
    
    def analyze_dspy_errors(self) -> Dict[str, Any]:
        """Analyze DSPy errors to identify patterns and issues."""
        entries = self._read_log_file(self.log_files["dspy_errors"])
        
        if not entries:
            return {"message": "No DSPy errors found"}
        
        analysis = {
            "total_errors": len(entries),
            "error_types": Counter(),
            "operations_affected": Counter(),
            "signature_availability": {
                "sme_analysis": 0,
                "mvr_pattern": 0,
                "sme_validation": 0
            },
            "common_error_messages": Counter(),
            "time_distribution": defaultdict(int)
        }
        
        for entry in entries:
            # Count error types
            error_type = entry.get("error_type", "Unknown")
            analysis["error_types"][error_type] += 1
            
            # Count operations
            operation = entry.get("operation", "Unknown")
            analysis["operations_affected"][operation] += 1
            
            # Analyze signature availability
            signatures = entry.get("dspy_signatures_available", {})
            for sig_name, available in signatures.items():
                if not available:
                    analysis["signature_availability"][sig_name] += 1
            
            # Count common error messages
            error_msg = entry.get("error_message", "")
            analysis["common_error_messages"][error_msg] += 1
            
            # Time distribution
            timestamp = entry.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = dt.hour
                    analysis["time_distribution"][hour] += 1
                except:
                    pass
        
        return analysis
    
    def analyze_fallback_usage(self) -> Dict[str, Any]:
        """Analyze fallback usage patterns."""
        entries = self._read_log_file(self.log_files["fallback_usage"])
        
        if not entries:
            return {"message": "No fallback usage found"}
        
        analysis = {
            "total_fallbacks": len(entries),
            "fallback_types": Counter(),
            "operations_using_fallback": Counter(),
            "performance_impact": Counter(),
            "time_distribution": defaultdict(int)
        }
        
        for entry in entries:
            # Count fallback types
            fallback_type = entry.get("fallback_type", "Unknown")
            analysis["fallback_types"][fallback_type] += 1
            
            # Count operations
            operation = entry.get("operation", "Unknown")
            analysis["operations_using_fallback"][operation] += 1
            
            # Performance impact
            impact = entry.get("performance_impact", "Unknown")
            analysis["performance_impact"][impact] += 1
            
            # Time distribution
            timestamp = entry.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = dt.hour
                    analysis["time_distribution"][hour] += 1
                except:
                    pass
        
        return analysis
    
    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics."""
        entries = self._read_log_file(self.log_files["performance_metrics"])
        
        if not entries:
            return {"message": "No performance metrics found"}
        
        analysis = {
            "total_operations": len(entries),
            "success_rate": 0,
            "operations": defaultdict(list),
            "performance_by_operation": {},
            "error_patterns": Counter()
        }
        
        successful_ops = 0
        
        for entry in entries:
            operation = entry.get("operation", "Unknown")
            duration = entry.get("duration_ms", 0)
            success = entry.get("success", False)
            
            if success:
                successful_ops += 1
            
            analysis["operations"][operation].append({
                "duration": duration,
                "success": success,
                "timestamp": entry.get("timestamp", "")
            })
            
            if not success:
                error = entry.get("details", {}).get("error", "Unknown")
                analysis["error_patterns"][error] += 1
        
        # Calculate success rate
        analysis["success_rate"] = (successful_ops / len(entries)) * 100 if entries else 0
        
        # Calculate performance by operation
        for operation, ops in analysis["operations"].items():
            if ops:
                durations = [op["duration"] for op in ops if op["duration"] > 0]
                if durations:
                    analysis["performance_by_operation"][operation] = {
                        "count": len(ops),
                        "avg_duration_ms": sum(durations) / len(durations),
                        "min_duration_ms": min(durations),
                        "max_duration_ms": max(durations),
                        "success_rate": (sum(1 for op in ops if op["success"]) / len(ops)) * 100
                    }
        
        return analysis
    
    def analyze_sme_analysis(self) -> Dict[str, Any]:
        """Analyze SME analysis patterns."""
        entries = self._read_log_file(self.log_files["sme_analysis"])
        
        if not entries:
            return {"message": "No SME analysis found"}
        
        analysis = {
            "total_analyses": len(entries),
            "success_rate": 0,
            "risk_categories": Counter(),
            "risk_score_distribution": defaultdict(int),
            "fallback_usage_rate": 0,
            "sme_contexts_used": defaultdict(int),
            "historical_records_analyzed": []
        }
        
        successful_analyses = 0
        fallback_usage = 0
        
        for entry in entries:
            status = entry.get("analysis_status", "")
            risk_category = entry.get("risk_category", "Unknown")
            risk_score = entry.get("risk_score", 0)
            used_fallback = entry.get("used_fallback", False)
            sme_contexts = entry.get("sme_contexts_used", 0)
            historical_records = entry.get("historical_records_analyzed", 0)
            
            if status == "success" or status == "success (synthetic)":
                successful_analyses += 1
            
            analysis["risk_categories"][risk_category] += 1
            analysis["risk_score_distribution"][risk_score] += 1
            analysis["sme_contexts_used"][sme_contexts] += 1
            analysis["historical_records_analyzed"].append(historical_records)
            
            if used_fallback:
                fallback_usage += 1
        
        # Calculate rates
        analysis["success_rate"] = (successful_analyses / len(entries)) * 100 if entries else 0
        analysis["fallback_usage_rate"] = (fallback_usage / len(entries)) * 100 if entries else 0
        
        # Calculate historical records statistics
        if analysis["historical_records_analyzed"]:
            records = analysis["historical_records_analyzed"]
            analysis["historical_records_stats"] = {
                "avg": sum(records) / len(records),
                "min": min(records),
                "max": max(records)
            }
        
        return analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_period": "All available logs",
            "dspy_errors": self.analyze_dspy_errors(),
            "fallback_usage": self.analyze_fallback_usage(),
            "performance_metrics": self.analyze_performance_metrics(),
            "sme_analysis": self.analyze_sme_analysis(),
            "recommendations": []
        }
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # DSPy error recommendations
        dspy_analysis = report["dspy_errors"]
        if isinstance(dspy_analysis, dict) and "total_errors" in dspy_analysis:
            if dspy_analysis["total_errors"] > 0:
                recommendations.append({
                    "category": "DSPy Configuration",
                    "priority": "High",
                    "issue": f"Found {dspy_analysis['total_errors']} DSPy errors",
                    "recommendation": "Review DSPy configuration and ensure proper LM setup"
                })
        
        # Fallback usage recommendations
        fallback_analysis = report["fallback_usage"]
        if isinstance(fallback_analysis, dict) and "total_fallbacks" in fallback_analysis:
            if fallback_analysis["total_fallbacks"] > 0:
                recommendations.append({
                    "category": "System Reliability",
                    "priority": "Medium",
                    "issue": f"System used fallbacks {fallback_analysis['total_fallbacks']} times",
                    "recommendation": "Consider improving DSPy configuration to reduce fallback usage"
                })
        
        # Performance recommendations
        perf_analysis = report["performance_metrics"]
        if isinstance(perf_analysis, dict) and "success_rate" in perf_analysis:
            if perf_analysis["success_rate"] < 90:
                recommendations.append({
                    "category": "Performance",
                    "priority": "High",
                    "issue": f"Success rate is {perf_analysis['success_rate']:.1f}%",
                    "recommendation": "Investigate failed operations and improve error handling"
                })
        
        report["recommendations"] = recommendations
        
        return report
    
    def export_report_to_json(self, filename: str = "log_analysis_report.json"):
        """Export the comprehensive report to JSON file."""
        report = self.generate_comprehensive_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename
    
    def print_summary(self):
        """Print a summary of the log analysis."""
        report = self.generate_comprehensive_report()
        
        print("="*80)
        print("📊 LOG ANALYSIS SUMMARY")
        print("="*80)
        print(f"Analysis Time: {report['timestamp']}")
        print(f"Analysis Period: {report['analysis_period']}")
        print()
        
        # DSPy Errors
        dspy_analysis = report["dspy_errors"]
        if isinstance(dspy_analysis, dict) and "total_errors" in dspy_analysis:
            print(f"🔴 DSPy Errors: {dspy_analysis['total_errors']}")
            if dspy_analysis["total_errors"] > 0:
                print(f"   Most common error: {dspy_analysis['error_types'].most_common(1)[0] if dspy_analysis['error_types'] else 'N/A'}")
        else:
            print("🔴 DSPy Errors: No data")
        
        # Fallback Usage
        fallback_analysis = report["fallback_usage"]
        if isinstance(fallback_analysis, dict) and "total_fallbacks" in fallback_analysis:
            print(f"🟡 Fallback Usage: {fallback_analysis['total_fallbacks']}")
        else:
            print("🟡 Fallback Usage: No data")
        
        # Performance
        perf_analysis = report["performance_metrics"]
        if isinstance(perf_analysis, dict) and "success_rate" in perf_analysis:
            print(f"📈 Success Rate: {perf_analysis['success_rate']:.1f}%")
        else:
            print("📈 Success Rate: No data")
        
        # SME Analysis
        sme_analysis = report["sme_analysis"]
        if isinstance(sme_analysis, dict) and "total_analyses" in sme_analysis:
            print(f"🧠 SME Analyses: {sme_analysis['total_analyses']}")
            print(f"   Fallback Rate: {sme_analysis['fallback_usage_rate']:.1f}%")
        else:
            print("🧠 SME Analyses: No data")
        
        # Recommendations
        if report["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"   {i}. [{rec['priority']}] {rec['issue']}")
                print(f"      → {rec['recommendation']}")
        
        print("="*80)


def main():
    """Main function to run log analysis."""
    analyzer = LogAnalyzer()
    analyzer.print_summary()
    
    # Export detailed report
    report_file = analyzer.export_report_to_json()
    print(f"\n📄 Detailed report exported to: {report_file}")


if __name__ == "__main__":
    main()
