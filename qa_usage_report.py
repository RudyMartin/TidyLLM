#!/usr/bin/env python3
"""
QA Control Usage Report - Production Verification
=================================================

Generates detailed usage report to verify the QA Control process is running
in REAL mode, not simulation. Includes:

- Real vs Simulation detection
- Actual resource usage (tokens, API calls, processing time)
- Database connectivity verification
- S3 upload verification
- Gateway integration verification
- Audit trail completeness verification

Usage:
    python qa_usage_report.py
    python qa_usage_report.py --submission-id <id>
    python qa_usage_report.py --verify-production
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import argparse

class QAUsageReportGenerator:
    """Generates comprehensive usage reports to verify real processing."""
    
    def __init__(self):
        self.root = Path.cwd()
        self.audit_dir = self.root / "qa_audit"
        self.reports_dir = self.root / "qa_reports"
        
    def generate_usage_report(self, submission_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive usage report."""
        
        print("=" * 80)
        print("QA CONTROL USAGE REPORT - PRODUCTION VERIFICATION")
        print("=" * 80)
        
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "report_type": "usage_verification",
            "verification_status": "checking",
            "real_processing_indicators": {},
            "simulation_indicators": {},
            "audit_verification": {},
            "system_integration": {},
            "recommendations": []
        }
        
        if submission_id:
            report["scope"] = f"Single submission: {submission_id}"
            submissions = [submission_id] if self._submission_exists(submission_id) else []
        else:
            report["scope"] = "All recent submissions"
            submissions = self._get_recent_submissions()
        
        if not submissions:
            report["verification_status"] = "no_submissions"
            report["message"] = "No submissions found to verify"
            return report
        
        # Analyze each submission for real vs simulation indicators
        real_indicators = 0
        simulation_indicators = 0
        
        for sub_id in submissions:
            analysis = self._analyze_submission_reality(sub_id)
            
            if analysis["is_real"]:
                real_indicators += analysis["real_score"]
            else:
                simulation_indicators += analysis["simulation_score"]
            
            report["real_processing_indicators"][sub_id] = analysis["real_indicators"]
            report["simulation_indicators"][sub_id] = analysis["simulation_indicators"]
        
        # Overall verification
        total_score = real_indicators + simulation_indicators
        reality_percentage = (real_indicators / total_score) if total_score > 0 else 0
        
        report["verification_status"] = "real" if reality_percentage >= 0.7 else "simulated"
        report["reality_score"] = reality_percentage
        report["total_submissions"] = len(submissions)
        
        # System integration checks
        report["system_integration"] = self._verify_system_integration()
        
        # Audit verification
        report["audit_verification"] = self._verify_audit_completeness(submissions)
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        # Save report
        report_path = self.root / "qa_usage_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_report_summary(report)
        
        return report
    
    def _submission_exists(self, submission_id: str) -> bool:
        """Check if submission exists."""
        audit_file = self.audit_dir / submission_id / "audit.json"
        return audit_file.exists()
    
    def _get_recent_submissions(self) -> List[str]:
        """Get list of recent submissions."""
        submissions = []
        
        if self.audit_dir.exists():
            for item in self.audit_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    submissions.append(item.name)
        
        return sorted(submissions)[-10:]  # Last 10 submissions
    
    def _analyze_submission_reality(self, submission_id: str) -> Dict[str, Any]:
        """Analyze submission for real vs simulation indicators."""
        
        audit_file = self.audit_dir / submission_id / "audit.json"
        if not audit_file.exists():
            return {
                "is_real": False,
                "real_score": 0,
                "simulation_score": 10,
                "real_indicators": [],
                "simulation_indicators": ["No audit file found"]
            }
        
        audit_data = json.loads(audit_file.read_text())
        
        real_indicators = []
        simulation_indicators = []
        real_score = 0
        simulation_score = 0
        
        # Check 1: File hash verification
        if audit_data.get("file_hash"):
            if len(audit_data["file_hash"]) == 64:  # SHA-256
                real_indicators.append("Valid SHA-256 file hash")
                real_score += 5
            else:
                simulation_indicators.append("Invalid or missing file hash")
                simulation_score += 3
        
        # Check 2: Processing time analysis
        total_duration = audit_data.get("total_duration_seconds") or 0
        if total_duration > 5:  # Real processing takes time
            real_indicators.append(f"Realistic processing time: {total_duration:.2f}s")
            real_score += 5
        elif total_duration < 1:
            simulation_indicators.append(f"Suspiciously fast processing: {total_duration:.2f}s")
            simulation_score += 5
        
        # Check 3: Token usage verification
        stage_results = audit_data.get("stage_results", {})
        ai_analysis = stage_results.get("ai_analysis", {})
        tokens_used = ai_analysis.get("tokens_used", 0)
        
        if tokens_used > 100:
            real_indicators.append(f"Real token usage: {tokens_used} tokens")
            real_score += 10
        elif tokens_used == 0:
            simulation_indicators.append("No token usage recorded")
            simulation_score += 8
        
        # Check 4: Model configuration verification
        model_config = audit_data.get("model_config", {})
        if model_config.get("model_name") and model_config.get("gateway"):
            real_indicators.append(f"Model config: {model_config['model_name']} via {model_config['gateway']}")
            real_score += 3
        
        # Check 5: DSPy optimization indicators
        if model_config.get("dspy_parameters"):
            real_indicators.append("DSPy optimization configured")
            real_score += 3
        
        # Check 6: S3 location verification
        s3_locations = audit_data.get("s3_locations", [])
        if s3_locations:
            real_indicators.append(f"S3 uploads: {len(s3_locations)} locations")
            real_score += 5
        else:
            simulation_indicators.append("No S3 uploads recorded")
            simulation_score += 3
        
        # Check 7: Error handling
        errors = audit_data.get("errors", [])
        if errors:
            real_indicators.append(f"Real error handling: {len(errors)} errors logged")
            real_score += 2
        
        # Check 8: Metrics vs prompts processing
        metrics_processed = audit_data.get("metrics_processed", 0)
        prompts_processed = audit_data.get("prompts_processed", 0)
        
        if metrics_processed > 0 or prompts_processed > 0:
            real_indicators.append(f"Items processed: {metrics_processed + prompts_processed}")
            real_score += 5
        
        # Check 9: Mock response detection
        # Look for common simulation phrases
        mock_phrases = ["mock", "simulated", "demo", "test response", "[auto-analysis]"]
        audit_text = json.dumps(audit_data).lower()
        
        mock_count = sum(1 for phrase in mock_phrases if phrase in audit_text)
        if mock_count > 2:
            simulation_indicators.append(f"Mock language detected: {mock_count} instances")
            simulation_score += mock_count * 2
        
        # Check 10: Realistic timestamps
        submitted_at = audit_data.get("submitted_at")
        if submitted_at:
            try:
                submitted_time = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))
                if submitted_time > datetime.now() - timedelta(days=30):
                    real_indicators.append("Recent realistic timestamp")
                    real_score += 2
            except:
                simulation_indicators.append("Invalid timestamp format")
                simulation_score += 2
        
        return {
            "is_real": real_score > simulation_score,
            "real_score": real_score,
            "simulation_score": simulation_score,
            "real_indicators": real_indicators,
            "simulation_indicators": simulation_indicators
        }
    
    def _verify_system_integration(self) -> Dict[str, Any]:
        """Verify integration with external systems."""
        
        integration = {
            "database_connection": self._check_database_integration(),
            "s3_integration": self._check_s3_integration(),
            "ai_gateway": self._check_ai_gateway_integration(),
            "notification_system": self._check_notification_integration()
        }
        
        return integration
    
    def _check_database_integration(self) -> Dict[str, Any]:
        """Check database integration."""
        try:
            # Try to execute FLOW command and check for database indicators
            import subprocess
            result = subprocess.run([
                sys.executable, "3-demo.py", "[QA Control]"
            ], capture_output=True, text=True, timeout=30)
            
            if "database_ready: True" in result.stdout:
                return {
                    "status": "connected",
                    "evidence": "PostgreSQL connection confirmed in FLOW execution",
                    "real_indicator": True
                }
            else:
                return {
                    "status": "not_connected",
                    "evidence": "No database connection confirmed",
                    "real_indicator": False
                }
        except Exception as e:
            return {
                "status": "error",
                "evidence": f"Database check failed: {e}",
                "real_indicator": False
            }
    
    def _check_s3_integration(self) -> Dict[str, Any]:
        """Check S3 integration."""
        # Look for S3 configuration files
        config_paths = [
            Path.home() / ".aws" / "credentials",
            Path.home() / ".aws" / "config",
            self.root / "tidyllm" / "admin" / "settings.yaml"
        ]
        
        s3_configured = any(p.exists() for p in config_paths)
        
        return {
            "status": "configured" if s3_configured else "not_configured",
            "evidence": f"S3 config files: {[str(p) for p in config_paths if p.exists()]}",
            "real_indicator": s3_configured
        }
    
    def _check_ai_gateway_integration(self) -> Dict[str, Any]:
        """Check AI gateway integration."""
        # Check for gateway modules
        gateway_indicators = []
        
        gateway_paths = [
            self.root / "tidyllm" / "gateways",
            self.root / "tidyllm" / "flow" / "flow_agreements.py"
        ]
        
        for path in gateway_paths:
            if path.exists():
                gateway_indicators.append(str(path))
        
        return {
            "status": "available" if gateway_indicators else "missing",
            "evidence": f"Gateway modules: {gateway_indicators}",
            "real_indicator": len(gateway_indicators) > 0
        }
    
    def _check_notification_integration(self) -> Dict[str, Any]:
        """Check notification system."""
        # Check for notification configuration
        notification_indicators = []
        
        # Look for email/webhook configs
        if (self.root / "tidyllm" / "admin" / "settings.yaml").exists():
            notification_indicators.append("Settings file exists")
        
        return {
            "status": "configured" if notification_indicators else "not_configured", 
            "evidence": notification_indicators,
            "real_indicator": len(notification_indicators) > 0
        }
    
    def _verify_audit_completeness(self, submissions: List[str]) -> Dict[str, Any]:
        """Verify audit trail completeness."""
        
        audit_verification = {
            "master_log_exists": (self.audit_dir / "master_audit.jsonl").exists(),
            "individual_audits": 0,
            "complete_audits": 0,
            "missing_fields": [],
            "completeness_score": 0.0
        }
        
        required_fields = [
            "submission_id", "submitted_by", "submitted_at", "file_hash",
            "model_config", "status", "metrics_processed", "prompts_processed"
        ]
        
        for sub_id in submissions:
            audit_file = self.audit_dir / sub_id / "audit.json"
            if audit_file.exists():
                audit_verification["individual_audits"] += 1
                
                try:
                    audit_data = json.loads(audit_file.read_text())
                    missing = [field for field in required_fields if not audit_data.get(field)]
                    
                    if not missing:
                        audit_verification["complete_audits"] += 1
                    else:
                        audit_verification["missing_fields"].extend(missing)
                        
                except Exception as e:
                    audit_verification["missing_fields"].append(f"Parse error in {sub_id}")
        
        if audit_verification["individual_audits"] > 0:
            audit_verification["completeness_score"] = audit_verification["complete_audits"] / audit_verification["individual_audits"]
        
        return audit_verification
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on report findings."""
        
        recommendations = []
        reality_score = report.get("reality_score", 0)
        
        if reality_score < 0.5:
            recommendations.append("CRITICAL: System appears to be running in simulation mode")
            recommendations.append("REQUIRED: Configure real AI gateway connections")
            recommendations.append("REQUIRED: Set up AWS credentials for S3 integration") 
            recommendations.append("REQUIRED: Enable PostgreSQL database connection")
        elif reality_score < 0.8:
            recommendations.append("PARTIAL: Some simulation indicators detected")
            recommendations.append("IMPROVE: Review token usage - ensure real API calls")
            recommendations.append("IMPROVE: Verify S3 uploads are actually occurring")
        else:
            recommendations.append("GOOD: System appears to be running in real mode")
        
        # System integration recommendations
        integration = report.get("system_integration", {})
        
        if not integration.get("database_connection", {}).get("real_indicator"):
            recommendations.append("IMPROVE: Enable PostgreSQL connection for audit storage")
        
        if not integration.get("s3_integration", {}).get("real_indicator"):
            recommendations.append("IMPROVE: Configure AWS S3 credentials for report storage")
        
        if not integration.get("ai_gateway", {}).get("real_indicator"):
            recommendations.append("IMPROVE: Set up AI gateway modules for real processing")
        
        # Audit completeness
        audit = report.get("audit_verification", {})
        if audit.get("completeness_score", 0) < 0.9:
            recommendations.append("IMPROVE: Improve audit trail completeness")
        
        return recommendations
    
    def _print_report_summary(self, report: Dict[str, Any]):
        """Print report summary to console."""
        
        print(f"\nREPORT SUMMARY")
        print("-" * 40)
        print(f"Verification Status: {report['verification_status'].upper()}")
        print(f"Reality Score: {report['reality_score']:.1%}")
        print(f"Submissions Analyzed: {report['total_submissions']}")
        
        # System Integration Status
        integration = report["system_integration"]
        print(f"\nSYSTEM INTEGRATION:")
        print(f"- Database: {integration['database_connection']['status']}")
        print(f"- S3: {integration['s3_integration']['status']}")
        print(f"- AI Gateway: {integration['ai_gateway']['status']}")
        print(f"- Notifications: {integration['notification_system']['status']}")
        
        # Audit Status
        audit = report["audit_verification"]
        print(f"\nAUDIT COMPLETENESS:")
        print(f"- Individual Audits: {audit['individual_audits']}")
        print(f"- Complete Audits: {audit['complete_audits']}")
        print(f"- Completeness Score: {audit['completeness_score']:.1%}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  {rec}")
        
        print(f"\nDetailed report saved to: qa_usage_report.json")
    
    def verify_production_readiness(self) -> bool:
        """Verify system is production ready (not simulation)."""
        
        print("=" * 80)
        print("QA CONTROL PRODUCTION READINESS VERIFICATION")
        print("=" * 80)
        
        # Run a test submission
        print("\n[1] RUNNING TEST SUBMISSION...")
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "qa_control_enhanced.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("[SUCCESS] Test submission completed successfully")
            else:
                print("[ERROR] Test submission failed")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"[ERROR] Test submission error: {e}")
            return False
        
        # Generate usage report
        print("\n[2] ANALYZING USAGE PATTERNS...")
        report = self.generate_usage_report()
        
        # Determine production readiness
        reality_score = report.get("reality_score", 0)
        integration_count = sum(1 for v in report["system_integration"].values() if v.get("real_indicator"))
        
        production_ready = (
            reality_score >= 0.7 and 
            integration_count >= 2 and
            report["audit_verification"]["completeness_score"] >= 0.8
        )
        
        print(f"\n[3] PRODUCTION READINESS ASSESSMENT")
        print("-" * 40)
        if production_ready:
            print("[SUCCESS] SYSTEM IS PRODUCTION READY")
            print("   - Real processing confirmed")
            print("   - System integrations working")
            print("   - Audit trail complete")
        else:
            print("[FAILED] SYSTEM NOT PRODUCTION READY")
            print("   Issues to resolve:")
            for rec in report["recommendations"]:
                if "CRITICAL" in rec or "FAILED" in rec:
                    print(f"   {rec}")
        
        return production_ready

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="QA Control Usage Report")
    parser.add_argument("--submission-id", help="Analyze specific submission")
    parser.add_argument("--verify-production", action="store_true", 
                       help="Verify production readiness")
    
    args = parser.parse_args()
    
    generator = QAUsageReportGenerator()
    
    if args.verify_production:
        production_ready = generator.verify_production_readiness()
        sys.exit(0 if production_ready else 1)
    else:
        report = generator.generate_usage_report(args.submission_id)
        
        # Exit code based on reality score
        reality_score = report.get("reality_score", 0)
        sys.exit(0 if reality_score >= 0.7 else 1)

if __name__ == "__main__":
    main()