#!/usr/bin/env python3
"""
Risk Screening Service
=====================

Single service that generates one aggregate screening_{date}.json file
containing all risk tags, assessments, and regulatory overview.
"""

import json
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from contextlib import contextmanager

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_risk_tagging import EnhancedRiskTagger

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout operations."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

class RiskScreeningService:
    """Single service for comprehensive risk screening with date-stamped output."""
    
    def __init__(self, base_path: Path, max_retries: int = 3, timeout_seconds: int = 30):
        self.base_path = base_path
        self.tagger = EnhancedRiskTagger()
        self.screening_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"screening_{self.screening_date}.json"
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.max_files_per_directory = 100  # Limit files per directory
    
    def run_full_screening(self) -> Dict[str, Any]:
        """Run complete risk screening and save to single JSON file."""
        try:
            print(f"RISK SCREENING SERVICE - {self.screening_date}")
            print("=" * 60)
            
            # Initialize results structure
            screening_results = {
                'screening_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'screening_date': self.screening_date,
                    'base_path': str(self.base_path),
                    'service_version': '1.0.0',
                    'assessor': 'RiskScreeningService'
                },
                'file_assessments': {},
                'directory_summaries': {},
                'regulatory_overview': {},
                'risk_summary': {},
                'compliance_status': {},
                'production_readiness': {},
                'errors': []
            }
        
            # Scan each major directory
            directories_to_scan = ['tidyllm', 'v2', 'onboarding', 'pending']
            
            total_files = 0
            total_risk_score = 0.0
            all_compliance_requirements = set()
            all_security_concerns = []
            all_privacy_concerns = []
            
            for dir_name in directories_to_scan:
                try:
                    dir_path = self.base_path / dir_name
                    if dir_path.exists():
                        print(f"\nScanning {dir_name}/...")
                        
                        # Get directory scan results with timeout and retry
                        dir_results = self._scan_directory_with_retry(dir_path, dir_name)
                
                        # Store directory summary
                        screening_results['directory_summaries'][dir_name] = {
                            'total_files': dir_results['total_files_scanned'],
                            'file_types': dir_results['file_type_breakdown'],
                            'risk_levels': dir_results['risk_level_breakdown'],
                            'data_classifications': dir_results['data_classification_breakdown'],
                            'total_risk_score': dir_results['total_risk_score'],
                            'files_needing_audit': len(dir_results['files_needing_audit']),
                            'files_needing_review': len(dir_results['files_needing_review']),
                            'compliance_requirements': dir_results['compliance_requirements'],
                            'security_concerns': dir_results['security_concerns'],
                            'privacy_concerns': dir_results['privacy_concerns']
                        }
                        
                        # Store individual file assessments
                        for assessment in dir_results['detailed_assessments']:
                            try:
                                file_path = assessment.file_path
                                screening_results['file_assessments'][file_path] = {
                                    'file_type': assessment.file_type,
                                    'mime_type': assessment.mime_type,
                                    'risk_level': assessment.risk_level,
                                    'risk_score': assessment.risk_score,
                                    'compliance_requirements': assessment.compliance_requirements,
                                    'security_concerns': assessment.security_concerns,
                                    'privacy_concerns': assessment.privacy_concerns,
                                    'data_classification': assessment.data_classification,
                                    'needs_audit': assessment.needs_audit,
                                    'needs_review': assessment.needs_review,
                                    'file_size_mb': assessment.file_size_mb,
                                    'line_count': assessment.line_count,
                                    'directory': dir_name
                                }
                            except Exception as e:
                                error_msg = f"Error processing file assessment for {dir_name}: {str(e)}"
                                screening_results['errors'].append(error_msg)
                                print(f"  WARNING: {error_msg}")
                        
                        # Aggregate totals
                        total_files += dir_results['total_files_scanned']
                        total_risk_score += dir_results['total_risk_score']
                        all_compliance_requirements.update(dir_results['compliance_requirements'])
                        all_security_concerns.extend(dir_results['security_concerns'])
                        all_privacy_concerns.extend(dir_results['privacy_concerns'])
                        
                        print(f"  Files: {dir_results['total_files_scanned']}")
                        print(f"  Risk Score: {dir_results['total_risk_score']:.2f}")
                        print(f"  High Risk: {dir_results['risk_level_breakdown']['HIGH']}")
                        print(f"  Needs Audit: {len(dir_results['files_needing_audit'])}")
                    else:
                        print(f"\nSkipping {dir_name}/ (not found)")
                        
                except Exception as e:
                    error_msg = f"Error scanning directory {dir_name}: {str(e)}"
                    screening_results['errors'].append(error_msg)
                    print(f"  ERROR: {error_msg}")
                    continue
        
            # Generate regulatory overview
            try:
                print(f"\nGenerating regulatory overview...")
                regulatory_overview = self.tagger.generate_regulatory_overview({
                    'total_files_scanned': total_files,
                    'file_type_breakdown': self._aggregate_file_types(screening_results['directory_summaries']),
                    'risk_level_breakdown': self._aggregate_risk_levels(screening_results['directory_summaries']),
                    'data_classification_breakdown': self._aggregate_data_classifications(screening_results['directory_summaries']),
                    'total_risk_score': total_risk_score,
                    'files_needing_audit': self._aggregate_audit_files(screening_results['file_assessments']),
                    'files_needing_review': self._aggregate_review_files(screening_results['file_assessments']),
                    'compliance_requirements': list(all_compliance_requirements),
                    'security_concerns': all_security_concerns,
                    'privacy_concerns': all_privacy_concerns
                })
                
                screening_results['regulatory_overview'] = regulatory_overview
            except Exception as e:
                error_msg = f"Error generating regulatory overview: {str(e)}"
                screening_results['errors'].append(error_msg)
                print(f"  ERROR: {error_msg}")
                screening_results['regulatory_overview'] = {'error': error_msg}
        
        # Generate risk summary
        screening_results['risk_summary'] = {
            'total_files_assessed': total_files,
            'overall_risk_score': total_risk_score,
            'average_risk_score': total_risk_score / total_files if total_files > 0 else 0,
            'high_risk_files': len([f for f in screening_results['file_assessments'].values() if f['risk_level'] == 'HIGH']),
            'medium_risk_files': len([f for f in screening_results['file_assessments'].values() if f['risk_level'] == 'MEDIUM']),
            'low_risk_files': len([f for f in screening_results['file_assessments'].values() if f['risk_level'] == 'LOW']),
            'untagged_files': len([f for f in screening_results['file_assessments'].values() if f['risk_level'] == 'UNTAGGED']),
            'files_needing_audit': len([f for f in screening_results['file_assessments'].values() if f['needs_audit']]),
            'files_needing_review': len([f for f in screening_results['file_assessments'].values() if f['needs_review']])
        }
        
        # Generate compliance status
        screening_results['compliance_status'] = {
            'identified_frameworks': list(all_compliance_requirements),
            'framework_count': len(all_compliance_requirements),
            'security_concerns_count': len(all_security_concerns),
            'privacy_concerns_count': len(all_privacy_concerns),
            'compliance_readiness': regulatory_overview['executive_summary']['compliance_readiness'],
            'data_protection_status': regulatory_overview['executive_summary']['data_protection_status']
        }
        
        # Generate production readiness assessment
        screening_results['production_readiness'] = {
            'overall_risk_level': regulatory_overview['executive_summary']['overall_risk_level'],
            'approved_for_production': self._assess_production_readiness(screening_results),
            'critical_issues': self._identify_critical_issues(screening_results),
            'recommendations': regulatory_overview['recommendations']
        }
        
            # Save to single JSON file
            try:
                print(f"\nSaving results to: {self.output_file}")
                with open(self.output_file, 'w') as f:
                    json.dump(screening_results, f, indent=2, default=str)
                print(f"  SUCCESS: Results saved to {self.output_file}")
            except Exception as e:
                error_msg = f"Error saving results to {self.output_file}: {str(e)}"
                screening_results['errors'].append(error_msg)
                print(f"  ERROR: {error_msg}")
            
            # Print summary
            try:
                self._print_screening_summary(screening_results)
            except Exception as e:
                error_msg = f"Error printing summary: {str(e)}"
                screening_results['errors'].append(error_msg)
                print(f"  ERROR: {error_msg}")
            
            return screening_results
            
        except Exception as e:
            error_msg = f"Critical error in run_full_screening: {str(e)}"
            print(f"CRITICAL ERROR: {error_msg}")
            return {
                'screening_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'screening_date': self.screening_date,
                    'base_path': str(self.base_path),
                    'service_version': '1.0.0',
                    'assessor': 'RiskScreeningService',
                    'status': 'FAILED'
                },
                'errors': [error_msg],
                'file_assessments': {},
                'directory_summaries': {},
                'regulatory_overview': {'error': error_msg},
                'risk_summary': {'error': error_msg},
                'compliance_status': {'error': error_msg},
                'production_readiness': {'error': error_msg}
            }
    
    def _scan_directory_with_retry(self, dir_path: Path, dir_name: str) -> Dict:
        """Scan directory with retry logic and timeout."""
        for attempt in range(self.max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{self.max_retries} for {dir_name}")
                
                # Use timeout for directory scanning
                with timeout(self.timeout_seconds):
                    # Limit the number of files to prevent blocking
                    limited_results = self._scan_directory_limited(dir_path)
                    return limited_results
                    
            except TimeoutError as e:
                error_msg = f"Timeout scanning {dir_name} (attempt {attempt + 1}): {str(e)}"
                print(f"  WARNING: {error_msg}")
                if attempt == self.max_retries - 1:
                    # Return empty results on final failure
                    return {
                        'total_files_scanned': 0,
                        'file_type_breakdown': {},
                        'risk_level_breakdown': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNTAGGED': 0},
                        'data_classification_breakdown': {'PUBLIC': 0, 'INTERNAL': 0, 'CONFIDENTIAL': 0, 'RESTRICTED': 0},
                        'total_risk_score': 0.0,
                        'files_needing_audit': [],
                        'files_needing_review': [],
                        'compliance_requirements': [],
                        'security_concerns': [],
                        'privacy_concerns': [],
                        'detailed_assessments': [],
                        'error': error_msg
                    }
                time.sleep(1)  # Wait before retry
                
            except Exception as e:
                error_msg = f"Error scanning {dir_name} (attempt {attempt + 1}): {str(e)}"
                print(f"  WARNING: {error_msg}")
                if attempt == self.max_retries - 1:
                    return {
                        'total_files_scanned': 0,
                        'file_type_breakdown': {},
                        'risk_level_breakdown': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNTAGGED': 0},
                        'data_classification_breakdown': {'PUBLIC': 0, 'INTERNAL': 0, 'CONFIDENTIAL': 0, 'RESTRICTED': 0},
                        'total_risk_score': 0.0,
                        'files_needing_audit': [],
                        'files_needing_review': [],
                        'compliance_requirements': [],
                        'security_concerns': [],
                        'privacy_concerns': [],
                        'detailed_assessments': [],
                        'error': error_msg
                    }
                time.sleep(1)  # Wait before retry
    
    def _scan_directory_limited(self, dir_path: Path) -> Dict:
        """Scan directory with file count limits to prevent blocking."""
        try:
            # Get all files first
            all_files = []
            supported_extensions = ['.py', '.md', '.txt', '.html', '.csv', '.json', '.yaml', '.yml', '.ini', '.cfg']
            
            for file_path in dir_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    all_files.append(file_path)
                    if len(all_files) >= self.max_files_per_directory:
                        print(f"    Limiting to {self.max_files_per_directory} files to prevent timeout")
                        break
            
            print(f"    Found {len(all_files)} files to scan")
            
            # Process files in batches
            results = {
                'total_files_scanned': 0,
                'file_type_breakdown': {},
                'risk_level_breakdown': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNTAGGED': 0},
                'data_classification_breakdown': {'PUBLIC': 0, 'INTERNAL': 0, 'CONFIDENTIAL': 0, 'RESTRICTED': 0},
                'total_risk_score': 0.0,
                'files_needing_audit': [],
                'files_needing_review': [],
                'compliance_requirements': set(),
                'security_concerns': [],
                'privacy_concerns': [],
                'detailed_assessments': []
            }
            
            # Process files with timeout per file
            for i, file_path in enumerate(all_files):
                try:
                    if i % 10 == 0:  # Progress indicator
                        print(f"    Processing file {i+1}/{len(all_files)}")
                    
                    # Quick file assessment with timeout
                    with timeout(5):  # 5 seconds per file
                        assessment = self.tagger.assess_file_risk(file_path)
                        results['detailed_assessments'].append(assessment)
                        results['total_files_scanned'] += 1
                        
                        # Update breakdowns
                        file_type = assessment.file_type
                        results['file_type_breakdown'][file_type] = results['file_type_breakdown'].get(file_type, 0) + 1
                        results['risk_level_breakdown'][assessment.risk_level] += 1
                        results['data_classification_breakdown'][assessment.data_classification] += 1
                        results['total_risk_score'] += assessment.risk_score
                        
                        if assessment.needs_audit:
                            results['files_needing_audit'].append(assessment.file_path)
                        if assessment.needs_review:
                            results['files_needing_review'].append(assessment.file_path)
                        
                        results['compliance_requirements'].update(assessment.compliance_requirements)
                        results['security_concerns'].extend(assessment.security_concerns)
                        results['privacy_concerns'].extend(assessment.privacy_concerns)
                        
                except TimeoutError:
                    print(f"    Timeout on file: {file_path.name}")
                    continue
                except Exception as e:
                    print(f"    Error on file {file_path.name}: {str(e)}")
                    continue
            
            # Convert set to list
            results['compliance_requirements'] = list(results['compliance_requirements'])
            
            return results
            
        except Exception as e:
            print(f"    Critical error in directory scan: {str(e)}")
            return {
                'total_files_scanned': 0,
                'file_type_breakdown': {},
                'risk_level_breakdown': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNTAGGED': 0},
                'data_classification_breakdown': {'PUBLIC': 0, 'INTERNAL': 0, 'CONFIDENTIAL': 0, 'RESTRICTED': 0},
                'total_risk_score': 0.0,
                'files_needing_audit': [],
                'files_needing_review': [],
                'compliance_requirements': [],
                'security_concerns': [],
                'privacy_concerns': [],
                'detailed_assessments': [],
                'error': str(e)
            }
    
    def _aggregate_file_types(self, directory_summaries: Dict) -> Dict[str, int]:
        """Aggregate file type counts across all directories."""
        try:
            aggregated = {}
            for dir_summary in directory_summaries.values():
                if 'file_types' in dir_summary:
                    for file_type, count in dir_summary['file_types'].items():
                        aggregated[file_type] = aggregated.get(file_type, 0) + count
            return aggregated
        except Exception as e:
            print(f"  WARNING: Error aggregating file types: {str(e)}")
            return {}
    
    def _aggregate_risk_levels(self, directory_summaries: Dict) -> Dict[str, int]:
        """Aggregate risk level counts across all directories."""
        try:
            aggregated = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNTAGGED': 0}
            for dir_summary in directory_summaries.values():
                if 'risk_levels' in dir_summary:
                    for risk_level, count in dir_summary['risk_levels'].items():
                        aggregated[risk_level] = aggregated.get(risk_level, 0) + count
            return aggregated
        except Exception as e:
            print(f"  WARNING: Error aggregating risk levels: {str(e)}")
            return {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNTAGGED': 0}
    
    def _aggregate_data_classifications(self, directory_summaries: Dict) -> Dict[str, int]:
        """Aggregate data classification counts across all directories."""
        try:
            aggregated = {'PUBLIC': 0, 'INTERNAL': 0, 'CONFIDENTIAL': 0, 'RESTRICTED': 0}
            for dir_summary in directory_summaries.values():
                if 'data_classifications' in dir_summary:
                    for classification, count in dir_summary['data_classifications'].items():
                        aggregated[classification] = aggregated.get(classification, 0) + count
            return aggregated
        except Exception as e:
            print(f"  WARNING: Error aggregating data classifications: {str(e)}")
            return {'PUBLIC': 0, 'INTERNAL': 0, 'CONFIDENTIAL': 0, 'RESTRICTED': 0}
    
    def _aggregate_audit_files(self, file_assessments: Dict) -> List[str]:
        """Get all files that need audit."""
        try:
            return [file_path for file_path, assessment in file_assessments.items() 
                    if assessment.get('needs_audit', False)]
        except Exception as e:
            print(f"  WARNING: Error aggregating audit files: {str(e)}")
            return []
    
    def _aggregate_review_files(self, file_assessments: Dict) -> List[str]:
        """Get all files that need review."""
        try:
            return [file_path for file_path, assessment in file_assessments.items() 
                    if assessment.get('needs_review', False)]
        except Exception as e:
            print(f"  WARNING: Error aggregating review files: {str(e)}")
            return []
    
    def _assess_production_readiness(self, screening_results: Dict) -> bool:
        """Assess if system is ready for production."""
        try:
            risk_summary = screening_results.get('risk_summary', {})
            production_readiness = screening_results.get('production_readiness', {})
            
            # Check critical issues
            if production_readiness.get('critical_issues'):
                return False
            
            # Check risk levels
            if risk_summary.get('high_risk_files', 0) > 5:
                return False
            
            # Check overall risk level
            if production_readiness.get('overall_risk_level') == 'HIGH':
                return False
            
            return True
        except Exception as e:
            print(f"  WARNING: Error assessing production readiness: {str(e)}")
            return False
    
    def _identify_critical_issues(self, screening_results: Dict) -> List[str]:
        """Identify critical issues that block production."""
        try:
            issues = []
            
            risk_summary = screening_results.get('risk_summary', {})
            compliance_status = screening_results.get('compliance_status', {})
            
            # High-risk files
            if risk_summary.get('high_risk_files', 0) > 10:
                issues.append(f"Too many high-risk files: {risk_summary.get('high_risk_files', 0)}")
            
            # Files needing audit
            if risk_summary.get('files_needing_audit', 0) > 20:
                issues.append(f"Too many files needing audit: {risk_summary.get('files_needing_audit', 0)}")
            
            # Security concerns
            if compliance_status.get('security_concerns_count', 0) > 0:
                issues.append(f"Security concerns identified: {compliance_status.get('security_concerns_count', 0)}")
            
            # No compliance frameworks
            if compliance_status.get('framework_count', 0) == 0:
                issues.append("No compliance frameworks identified")
            
            return issues
        except Exception as e:
            print(f"  WARNING: Error identifying critical issues: {str(e)}")
            return [f"Error identifying critical issues: {str(e)}"]
    
    def _print_screening_summary(self, screening_results: Dict):
        """Print screening summary."""
        try:
            print("\n" + "=" * 60)
            print("RISK SCREENING SUMMARY")
            print("=" * 60)
            
            screening_metadata = screening_results.get('screening_metadata', {})
            risk_summary = screening_results.get('risk_summary', {})
            compliance_status = screening_results.get('compliance_status', {})
            production_readiness = screening_results.get('production_readiness', {})
            
            print(f"Screening Date: {screening_metadata.get('screening_date', 'UNKNOWN')}")
            print(f"Total Files Assessed: {risk_summary.get('total_files_assessed', 0)}")
            print(f"Overall Risk Score: {risk_summary.get('overall_risk_score', 0):.2f}")
            print(f"Average Risk Score: {risk_summary.get('average_risk_score', 0):.3f}")
            
            print(f"\nRisk Breakdown:")
            print(f"  High Risk: {risk_summary.get('high_risk_files', 0)}")
            print(f"  Medium Risk: {risk_summary.get('medium_risk_files', 0)}")
            print(f"  Low Risk: {risk_summary.get('low_risk_files', 0)}")
            print(f"  Untagged: {risk_summary.get('untagged_files', 0)}")
            
            print(f"\nCompliance Status:")
            print(f"  Frameworks Identified: {compliance_status.get('framework_count', 0)}")
            print(f"  Security Concerns: {compliance_status.get('security_concerns_count', 0)}")
            print(f"  Privacy Concerns: {compliance_status.get('privacy_concerns_count', 0)}")
            print(f"  Compliance Readiness: {compliance_status.get('compliance_readiness', 'UNKNOWN')}")
            
            print(f"\nProduction Readiness:")
            print(f"  Overall Risk Level: {production_readiness.get('overall_risk_level', 'UNKNOWN')}")
            print(f"  Approved for Production: {'YES' if production_readiness.get('approved_for_production', False) else 'NO'}")
            
            critical_issues = production_readiness.get('critical_issues', [])
            if critical_issues:
                print(f"  Critical Issues:")
                for issue in critical_issues:
                    print(f"    - {issue}")
            
            # Show errors if any
            errors = screening_results.get('errors', [])
            if errors:
                print(f"\nErrors Encountered: {len(errors)}")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"  - {error}")
                if len(errors) > 5:
                    print(f"  ... and {len(errors) - 5} more errors")
            
            print(f"\nOutput File: {self.output_file}")
        except Exception as e:
            print(f"  ERROR: Failed to print screening summary: {str(e)}")

def main():
    """Main screening function."""
    base_path = Path("C:/Users/marti/AI-Scoring")
    
    if not base_path.exists():
        print(f"ERROR: Base path not found: {base_path}")
        return 1
    
    # Create service with timeout and retry limits
    service = RiskScreeningService(
        base_path=base_path,
        max_retries=3,
        timeout_seconds=30
    )
    
    print(f"Starting risk screening with:")
    print(f"  Max retries: {service.max_retries}")
    print(f"  Timeout per directory: {service.timeout_seconds}s")
    print(f"  Max files per directory: {service.max_files_per_directory}")
    
    try:
        results = service.run_full_screening()
        return 0 if results.get('production_readiness', {}).get('approved_for_production', False) else 1
    except KeyboardInterrupt:
        print("\nScreening interrupted by user")
        return 1
    except Exception as e:
        print(f"\nCritical error in main: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
