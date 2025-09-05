#!/usr/bin/env python3
"""
Generate a comprehensive state-of-the-code report
Analyzes module structure, import paths, and identifies missing modules
"""

import os
import sys
import json
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

def analyze_test_imports() -> Dict[str, List[str]]:
    """Analyze all import statements in test files."""
    test_dir = Path("tests")
    import_analysis = {
        "test_files": [],
        "import_statements": [],
        "missing_modules": [],
        "valid_modules": [],
        "import_errors": []
    }
    
    if not test_dir.exists():
        return import_analysis
    
    for test_file in test_dir.glob("test_*.py"):
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            import_analysis["test_files"].append(str(test_file))
            
            # Extract import statements
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line.startswith('from ') or line.startswith('import '):
                    import_analysis["import_statements"].append({
                        "file": str(test_file),
                        "line": line_num,
                        "statement": line
                    })
                    
                    # Check if module exists
                    if line.startswith('from '):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            module_path = parts[1]
                            module_name = module_path.split('.')[0]
                            
                            # Check various possible locations
                            possible_paths = [
                                f"src/{module_name}",
                                f"src/backend/{module_name}",
                                f"src/{module_name}.py",
                                f"src/backend/{module_name}.py"
                            ]
                            
                            module_exists = any(Path(path).exists() for path in possible_paths)
                            
                            if module_exists:
                                import_analysis["valid_modules"].append({
                                    "module": module_name,
                                    "path": module_path,
                                    "file": str(test_file),
                                    "line": line_num
                                })
                            else:
                                import_analysis["missing_modules"].append({
                                    "module": module_name,
                                    "path": module_path,
                                    "file": str(test_file),
                                    "line": line_num,
                                    "statement": line
                                })
                                
        except Exception as e:
            import_analysis["import_errors"].append({
                "file": str(test_file),
                "error": str(e)
            })
    
    return import_analysis

def analyze_module_structure() -> Dict[str, any]:
    """Analyze the current module structure."""
    src_dir = Path("src")
    structure = {
        "src_exists": src_dir.exists(),
        "backend_exists": (src_dir / "backend").exists(),
        "modules": [],
        "missing_directories": [],
        "file_count": 0,
        "python_files": 0
    }
    
    if not src_dir.exists():
        structure["missing_directories"].append("src")
        return structure
    
    # Analyze src directory
    for item in src_dir.rglob("*"):
        if item.is_file():
            structure["file_count"] += 1
            if item.suffix == ".py":
                structure["python_files"] += 1
                structure["modules"].append({
                    "path": str(item.relative_to(src_dir)),
                    "size": item.stat().st_size,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
    
    # Check for expected directories
    expected_dirs = ["backend", "utils", "config"]
    for expected_dir in expected_dirs:
        if not (src_dir / expected_dir).exists():
            structure["missing_directories"].append(expected_dir)
    
    return structure

def check_import_paths() -> Dict[str, any]:
    """Check if import paths are consistent."""
    path_analysis = {
        "consistent_paths": [],
        "inconsistent_paths": [],
        "recommendations": []
    }
    
    # Common import patterns to check
    import_patterns = [
        ("personas", "backend.personas"),
        ("core", "backend.core"),
        ("mcp", "backend.mcp"),
        ("config", "backend.config")
    ]
    
    test_dir = Path("tests")
    if test_dir.exists():
        for test_file in test_dir.glob("test_*.py"):
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                
                for old_pattern, new_pattern in import_patterns:
                    if f"from {old_pattern}." in content:
                        path_analysis["inconsistent_paths"].append({
                            "file": str(test_file),
                            "old_pattern": old_pattern,
                            "new_pattern": new_pattern,
                            "recommendation": f"Update to use {new_pattern}"
                        })
                    elif f"from {new_pattern}." in content:
                        path_analysis["consistent_paths"].append({
                            "file": str(test_file),
                            "pattern": new_pattern
                        })
                        
            except Exception as e:
                continue
    
    return path_analysis

def generate_fix_recommendations(import_analysis: Dict, path_analysis: Dict) -> List[str]:
    """Generate recommendations for fixing import issues."""
    recommendations = []
    
    # Missing modules
    if import_analysis["missing_modules"]:
        recommendations.append("🔴 CRITICAL: Missing modules detected")
        for missing in import_analysis["missing_modules"]:
            recommendations.append(f"   - {missing['file']}:{missing['line']} - {missing['module']}")
        recommendations.append("   → Fix: Update import paths or create missing modules")
    
    # Inconsistent paths
    if path_analysis["inconsistent_paths"]:
        recommendations.append("🟡 WARNING: Inconsistent import paths detected")
        for inconsistent in path_analysis["inconsistent_paths"]:
            recommendations.append(f"   - {inconsistent['file']}: {inconsistent['recommendation']}")
        recommendations.append("   → Fix: Standardize import paths")
    
    # No issues
    if not import_analysis["missing_modules"] and not path_analysis["inconsistent_paths"]:
        recommendations.append("✅ All import paths are consistent")
    
    return recommendations

def main():
    """Generate comprehensive state-of-the-code report."""
    print("🔍 Generating State-of-the-Code Report...")
    print("=" * 50)
    
    # Analyze different aspects
    import_analysis = analyze_test_imports()
    module_structure = analyze_module_structure()
    path_analysis = check_import_paths()
    
    # Generate recommendations
    recommendations = generate_fix_recommendations(import_analysis, path_analysis)
    
    # Create comprehensive report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_test_files": len(import_analysis["test_files"]),
            "total_imports": len(import_analysis["import_statements"]),
            "missing_modules": len(import_analysis["missing_modules"]),
            "valid_modules": len(import_analysis["valid_modules"]),
            "import_errors": len(import_analysis["import_errors"]),
            "consistent_paths": len(path_analysis["consistent_paths"]),
            "inconsistent_paths": len(path_analysis["inconsistent_paths"])
        },
        "import_analysis": import_analysis,
        "module_structure": module_structure,
        "path_analysis": path_analysis,
        "recommendations": recommendations,
        "status": "healthy" if not import_analysis["missing_modules"] else "needs_attention"
    }
    
    # Print summary
    print(f"📊 Summary:")
    print(f"   Test Files: {report['summary']['total_test_files']}")
    print(f"   Total Imports: {report['summary']['total_imports']}")
    print(f"   Missing Modules: {report['summary']['missing_modules']}")
    print(f"   Valid Modules: {report['summary']['valid_modules']}")
    print(f"   Import Errors: {report['summary']['import_errors']}")
    print(f"   Consistent Paths: {report['summary']['consistent_paths']}")
    print(f"   Inconsistent Paths: {report['summary']['inconsistent_paths']}")
    print()
    
    # Print recommendations
    print("💡 Recommendations:")
    for rec in recommendations:
        print(f"   {rec}")
    print()
    
    # Save report
    output_dir = Path("data/tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = output_dir / f"code_state_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save to latest
    latest_file = output_dir / "latest_code_state_report.json"
    with open(latest_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📁 Report saved to:")
    print(f"   {report_file}")
    print(f"   {latest_file}")
    
    # Return status code
    if report['summary']['missing_modules'] > 0:
        print("\n❌ Issues detected - review recommendations above")
        sys.exit(1)
    else:
        print("\n✅ Code state is healthy")
        sys.exit(0)

if __name__ == "__main__":
    main()

