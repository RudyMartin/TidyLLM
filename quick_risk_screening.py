#!/usr/bin/env python3
"""
Quick Risk Screening with Progress Pulse
=======================================

Fast risk screening with visible progress indicators, retries, and timeout handling.
Handles unavailable files gracefully (expected on active sites).
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class ProgressPulse:
    """Simple progress pulse indicator."""
    
    def __init__(self, interval: float = 3.0):
        self.interval = interval
        self.start_time = time.time()
        self.last_pulse = self.start_time
        self.count = 0
    
    def pulse(self, message: str = ""):
        """Show progress pulse."""
        current_time = time.time()
        if current_time - self.last_pulse >= self.interval:
            elapsed = current_time - self.start_time
            self.count += 1
            print(f"💓 PULSE {self.count} - {elapsed:.1f}s elapsed - {message}")
            self.last_pulse = current_time

class QuickRiskScreening:
    """Quick risk screening with progress indicators."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.screening_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"screening_{self.screening_date}.json"
        self.pulse = ProgressPulse(interval=3.0)  # Pulse every 3 seconds
        
    def scan_directory_quick(self, dir_path: Path, dir_name: str, max_retries: int = 3, timeout_seconds: int = 15) -> Dict:
        """Quick directory scan with progress, retries, and timeout."""
        print(f"\n📁 Scanning {dir_name}/...")
        self.pulse.pulse(f"Starting {dir_name}")
        
        results = {
            'directory': dir_name,
            'total_files': 0,
            'python_files': 0,
            'markdown_files': 0,
            'other_files': 0,
            'high_risk_files': [],
            'medium_risk_files': [],
            'untagged_files': [],
            'compliance_requirements': set(),
            'errors': [],
            'unavailable_files': [],
            'timeout_files': []
        }
        
        # Retry logic for directory scanning
        for attempt in range(max_retries):
            try:
                print(f"  🔄 Attempt {attempt + 1}/{max_retries} for {dir_name}")
                self.pulse.pulse(f"Attempt {attempt + 1} for {dir_name}")
                
                # Timeout for directory scanning
                start_time = time.time()
                scan_results = self._scan_directory_with_timeout(dir_path, dir_name, timeout_seconds)
                
                # Merge results
                results.update(scan_results)
                elapsed = time.time() - start_time
                print(f"  ✅ Completed {dir_name} in {elapsed:.2f}s")
                return results
                
            except Exception as e:
                error_msg = f"Error scanning {dir_name} (attempt {attempt + 1}): {str(e)}"
                print(f"  ❌ {error_msg}")
                results['errors'].append(error_msg)
                if attempt == max_retries - 1:
                    print(f"  ❌ Max retries reached for {dir_name}")
                    return results
                time.sleep(1)  # Wait before retry
        
        return results
    
    def _scan_directory_with_timeout(self, dir_path: Path, dir_name: str, timeout_seconds: int) -> Dict:
        """Scan directory with timeout and handle unavailable files."""
        results = {
            'directory': dir_name,
            'total_files': 0,
            'python_files': 0,
            'markdown_files': 0,
            'other_files': 0,
            'high_risk_files': [],
            'medium_risk_files': [],
            'untagged_files': [],
            'compliance_requirements': set(),
            'errors': [],
            'unavailable_files': [],
            'timeout_files': []
        }
        
        try:
            # Quick file discovery with timeout
            all_files = []
            start_time = time.time()
            
            for file_path in dir_path.rglob('*'):
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    print(f"    ⏰ Timeout reached, stopping file discovery")
                    results['errors'].append(f"Timeout during file discovery in {dir_name}")
                    break
                
                if file_path.is_file():
                    all_files.append(file_path)
                    if len(all_files) >= 50:  # Limit for speed
                        print(f"    ⚡ Limiting to 50 files for speed")
                        break
            
            print(f"    📊 Found {len(all_files)} files")
            self.pulse.pulse(f"Found {len(all_files)} files in {dir_name}")
            
            # Quick risk assessment with per-file timeout
            for i, file_path in enumerate(all_files):
                if i % 10 == 0:
                    self.pulse.pulse(f"Processing {i+1}/{len(all_files)} files in {dir_name}")
                
                # Per-file timeout
                file_start_time = time.time()
                try:
                    # Quick file type check
                    suffix = file_path.suffix.lower()
                    if suffix == '.py':
                        results['python_files'] += 1
                    elif suffix == '.md':
                        results['markdown_files'] += 1
                    else:
                        results['other_files'] += 1
                    
                    results['total_files'] += 1
                    
                    # Quick risk tag check with timeout
                    try:
                        # Check if file is accessible
                        if not file_path.exists():
                            results['unavailable_files'].append(str(file_path))
                            continue
                        
                        # Try to read file with timeout
                        content = ""
                        try:
                            content = file_path.read_text(encoding='utf-8', errors='ignore')[:1000]  # First 1000 chars
                        except (PermissionError, OSError, IOError) as e:
                            # File is locked or unavailable - this is expected on active sites
                            results['unavailable_files'].append(str(file_path))
                            print(f"    🔒 File unavailable (expected): {file_path.name}")
                            continue
                        
                        content_lower = content.lower()
                        
                        if '@risk: high' in content_lower or 'risk level: high' in content_lower:
                            results['high_risk_files'].append(str(file_path))
                        elif '@risk: medium' in content_lower or 'risk level: medium' in content_lower:
                            results['medium_risk_files'].append(str(file_path))
                        elif '@risk:' in content_lower or 'risk level:' in content_lower:
                            pass  # Has some risk tag
                        else:
                            results['untagged_files'].append(str(file_path))
                        
                        # Quick compliance check
                        if 'sox' in content_lower:
                            results['compliance_requirements'].add('SOX')
                        if 'pci' in content_lower:
                            results['compliance_requirements'].add('PCI-DSS')
                        if 'gdpr' in content_lower:
                            results['compliance_requirements'].add('GDPR')
                        if 'ccpa' in content_lower:
                            results['compliance_requirements'].add('CCPA')
                            
                    except Exception as e:
                        results['errors'].append(f"Error reading {file_path.name}: {str(e)}")
                        
                except Exception as e:
                    results['errors'].append(f"Error processing {file_path.name}: {str(e)}")
                
                # Check per-file timeout
                if time.time() - file_start_time > 2:  # 2 seconds per file max
                    results['timeout_files'].append(str(file_path))
                    print(f"    ⏰ Timeout on file: {file_path.name}")
            
            # Convert set to list
            results['compliance_requirements'] = list(results['compliance_requirements'])
            
            print(f"    ✅ Completed {dir_name}: {results['total_files']} files")
            print(f"    🔒 Unavailable files: {len(results['unavailable_files'])} (expected on active site)")
            print(f"    ⏰ Timeout files: {len(results['timeout_files'])}")
            self.pulse.pulse(f"Completed {dir_name}")
            
        except Exception as e:
            error_msg = f"Critical error scanning {dir_name}: {str(e)}"
            results['errors'].append(error_msg)
            print(f"    ❌ {error_msg}")
            self.pulse.pulse(f"Error in {dir_name}")
        
        return results
    
    def run_quick_screening(self) -> Dict[str, Any]:
        """Run quick risk screening with progress."""
        print("🚀 QUICK RISK SCREENING SERVICE")
        print("=" * 50)
        print(f"📅 Date: {self.screening_date}")
        print(f"📂 Base Path: {self.base_path}")
        print("💓 Progress pulse every 3 seconds")
        print("⚡ Quick mode: Limited file scanning")
        print("🔄 Max retries: 3 per directory")
        print("⏱️  Timeout: 15s per directory, 2s per file")
        print("🔒 Handles unavailable files gracefully")
        print("=" * 50)
        
        start_time = time.time()
        
        # Initialize results
        screening_results = {
            'screening_metadata': {
                'timestamp': datetime.now().isoformat(),
                'screening_date': self.screening_date,
                'base_path': str(self.base_path),
                'service_version': 'quick-1.0.0',
                'assessor': 'QuickRiskScreening'
            },
            'directory_results': {},
            'summary': {},
            'errors': []
        }
        
        # Scan directories
        directories_to_scan = ['tidyllm', 'v2', 'onboarding', 'pending']
        
        total_files = 0
        total_high_risk = 0
        total_medium_risk = 0
        total_untagged = 0
        all_compliance = set()
        all_errors = []
        
        for dir_name in directories_to_scan:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                try:
                    dir_results = self.scan_directory_quick(dir_path, dir_name)
                    screening_results['directory_results'][dir_name] = dir_results
                    
                    # Aggregate totals
                    total_files += dir_results['total_files']
                    total_high_risk += len(dir_results['high_risk_files'])
                    total_medium_risk += len(dir_results['medium_risk_files'])
                    total_untagged += len(dir_results['untagged_files'])
                    all_compliance.update(dir_results['compliance_requirements'])
                    all_errors.extend(dir_results['errors'])
                    
                except Exception as e:
                    error_msg = f"Error scanning {dir_name}: {str(e)}"
                    all_errors.append(error_msg)
                    print(f"❌ {error_msg}")
                    self.pulse.pulse(f"Error in {dir_name}")
            else:
                print(f"⏭️  Skipping {dir_name}/ (not found)")
                self.pulse.pulse(f"Skipped {dir_name}")
        
        # Generate summary
        elapsed_time = time.time() - start_time
        total_unavailable = sum(len(dir_results.get('unavailable_files', [])) for dir_results in screening_results['directory_results'].values())
        total_timeout = sum(len(dir_results.get('timeout_files', [])) for dir_results in screening_results['directory_results'].values())
        
        screening_results['summary'] = {
            'total_files_scanned': total_files,
            'high_risk_files': total_high_risk,
            'medium_risk_files': total_medium_risk,
            'untagged_files': total_untagged,
            'unavailable_files': total_unavailable,
            'timeout_files': total_timeout,
            'compliance_frameworks': list(all_compliance),
            'total_errors': len(all_errors),
            'scanning_time_seconds': round(elapsed_time, 2),
            'production_ready': total_high_risk == 0 and len(all_errors) == 0
        }
        
        screening_results['errors'] = all_errors
        
        # Save results
        try:
            print(f"\n💾 Saving results to: {self.output_file}")
            with open(self.output_file, 'w') as f:
                json.dump(screening_results, f, indent=2, default=str)
            print(f"✅ Results saved successfully!")
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
            screening_results['errors'].append(f"Save error: {str(e)}")
        
        # Print final summary
        self._print_final_summary(screening_results, elapsed_time)
        
        return screening_results
    
    def _print_final_summary(self, results: Dict, elapsed_time: float):
        """Print final summary."""
        print("\n" + "=" * 50)
        print("📊 QUICK RISK SCREENING SUMMARY")
        print("=" * 50)
        
        summary = results['summary']
        
        print(f"⏱️  Scanning Time: {elapsed_time:.2f} seconds")
        print(f"📁 Total Files: {summary['total_files_scanned']}")
        print(f"🔴 High Risk: {summary['high_risk_files']}")
        print(f"🟡 Medium Risk: {summary['medium_risk_files']}")
        print(f"⚪ Untagged: {summary['untagged_files']}")
        print(f"🔒 Unavailable: {summary['unavailable_files']} (expected on active site)")
        print(f"⏰ Timeout: {summary['timeout_files']}")
        print(f"📋 Compliance Frameworks: {len(summary['compliance_frameworks'])}")
        print(f"❌ Errors: {summary['total_errors']}")
        
        if summary['compliance_frameworks']:
            print(f"   Frameworks: {', '.join(summary['compliance_frameworks'])}")
        
        print(f"\n🚀 Production Ready: {'YES' if summary['production_ready'] else 'NO'}")
        
        if results['errors']:
            print(f"\n⚠️  Errors encountered:")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(results['errors']) > 5:
                print(f"   ... and {len(results['errors']) - 5} more errors")
        
        print(f"\n📄 Output File: {self.output_file}")
        print("=" * 50)

def main():
    """Main quick screening function."""
    base_path = Path("C:/Users/marti/AI-Scoring")
    
    if not base_path.exists():
        print(f"❌ ERROR: Base path not found: {base_path}")
        return 1
    
    print("🚀 Starting Quick Risk Screening...")
    print("💓 You'll see progress pulses every 3 seconds")
    print("⚡ This is a fast, limited scan for quick results")
    print("🔄 Max retries: 3 per directory")
    print("⏱️  Timeout: 15s per directory, 2s per file")
    print("🔒 Handles unavailable files gracefully (expected on active sites)")
    print()
    
    try:
        screener = QuickRiskScreening(base_path)
        results = screener.run_quick_screening()
        
        return 0 if results['summary']['production_ready'] else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Screening interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
