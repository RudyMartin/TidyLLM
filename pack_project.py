#!/usr/bin/env python
"""
TidyLLM Project Packing Script

This script organizes the TidyLLM project into smaller, manageable zip files
for easier distribution and deployment.

Usage:
    python pack_project.py [options]

Options:
    --output-dir DIR     Output directory for zip files (default: ./packages)
    --max-size SIZE      Maximum size per package in MB (default: 50)
    --exclude PATTERN    Exclude files matching pattern (can be used multiple times)
    --include-empty      Include empty directories
    --dry-run           Show what would be packed without creating files
    --verbose           Show detailed progress
    --help              Show this help message
"""

import os
import sys
import zipfile
import argparse
import shutil
import json
from datetime import datetime

# Python 2/3 compatibility
try:
    from pathlib import Path
except ImportError:
    # Python 2 fallback
    class Path(object):
        def __init__(self, path):
            self.path = os.path.abspath(path)
        
        def __str__(self):
            return self.path
        
        def __div__(self, other):
            return Path(os.path.join(self.path, other))
        
        def is_file(self):
            return os.path.isfile(self.path)
        
        def is_dir(self):
            return os.path.isdir(self.path)
        
        def mkdir(self, parents=False, exist_ok=False):
            if parents:
                os.makedirs(self.path, exist_ok=exist_ok)
            else:
                os.mkdir(self.path)
        
        def stat(self):
            return os.stat(self.path)
        
        def relative_to(self, other):
            return Path(os.path.relpath(self.path, other.path))
        
        def resolve(self):
            return Path(os.path.abspath(self.path))


class ProjectPacker:
    def __init__(self, project_root, output_dir="./packages", max_size_mb=50):
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # Define package structure
        self.packages = {
            "01_core_tidyllm": {
                "description": "Core TidyLLM package and essential files",
                "paths": [
                    "tidyllm/",
                    "setup.py",
                    "pyproject.toml", 
                    "requirements.txt",
                    "MANIFEST.in",
                    "README.md",
                    "INSTALLATION.md",
                    "CLI_DOCUMENTATION.md",
                    "qa_processor.py",
                    "qa_test_runner.py"
                ],
                "estimated_size": "~2MB"
            },
            "02_knowledge_base": {
                "description": "Knowledge base with PDFs and documents",
                "paths": [
                    "knowledge_base/"
                ],
                "estimated_size": "~39MB"
            },
            "03_scripts_demos": {
                "description": "Scripts, demos, and automation tools",
                "paths": [
                    "scripts/",
                    "drop_zones/",
                    "prompts/"
                ],
                "estimated_size": "~5.5MB"
            },
            "04_educational_libs": {
                "description": "Educational ML libraries (tidyllm-sentence, tlm)",
                "paths": [
                    "tidyllm-sentence/",
                    "tlm/"
                ],
                "estimated_size": "~500KB"
            },
            "05_tests_docs": {
                "description": "Tests and ecosystem documentation",
                "paths": [
                    "tests/",
                    "paper_repository/",
                    "ECOSYSTEM_*.md",
                    "PACKAGE_SUCCESS_SUMMARY.md",
                    "Under-the-hood-with-flow.md"
                ],
                "estimated_size": "~1.5MB"
            },
            "06_demo_files": {
                "description": "Demo scripts and flow documentation",
                "paths": [
                    "1-enterprise.py",
                    "2-developer.py", 
                    "3-demo.py",
                    "flow_demo.py",
                    "FLOW_README.md"
                ],
                "estimated_size": "~100KB"
            }
        }
        
        # Files to exclude
        self.exclude_patterns = {
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            ".git/",
            ".gitignore",
            ".DS_Store",
            "*.log",
            "*.tmp",
            ".pytest_cache/",
            "*.egg-info/",
            "dist/",
            "build/",
            ".venv/",
            "venv/",
            "env/",
            ".env"
        }

    def get_file_size(self, file_path):
        """Get file size in bytes."""
        try:
            return os.path.getsize(str(file_path))
        except (OSError, IOError):
            return 0

    def should_exclude(self, path, exclude_patterns):
        """Check if path should be excluded."""
        path_str = str(path)
        for pattern in exclude_patterns:
            if pattern.endswith("/"):
                if pattern.rstrip("/") in path_str:
                    return True
            elif pattern.startswith("*"):
                if os.path.basename(path_str).endswith(pattern[1:]):
                    return True
            else:
                if pattern in path_str:
                    return True
        return False

    def collect_files(self, paths, exclude_patterns):
        """Collect all files matching the given paths."""
        files = []
        
        for path_str in paths:
            path = Path(os.path.join(str(self.project_root), path_str))
            
            if path.is_file():
                if not self.should_exclude(path, exclude_patterns):
                    files.append(path)
            elif path.is_dir():
                for root, dirs, filenames in os.walk(str(path)):
                    root_path = Path(root)
                    
                    # Remove excluded directories from dirs list to prevent walking into them
                    dirs[:] = [d for d in dirs if not self.should_exclude(Path(os.path.join(root, d)), exclude_patterns)]
                    
                    for filename in filenames:
                        file_path = Path(os.path.join(root, filename))
                        if not self.should_exclude(file_path, exclude_patterns):
                            files.append(file_path)
        
        return files

    def calculate_package_size(self, files):
        """Calculate total size of files in bytes."""
        total_size = 0
        for file_path in files:
            total_size += self.get_file_size(file_path)
        return total_size

    def create_zip_package(self, package_name, files, description, 
                          dry_run=False, verbose=False):
        """Create a zip package with the given files."""
        package_info = {
            "name": package_name,
            "description": description,
            "file_count": len(files),
            "total_size_bytes": self.calculate_package_size(files),
            "total_size_mb": round(self.calculate_package_size(files) / (1024.0 * 1024.0), 2),
            "created": datetime.now().isoformat(),
            "files": []
        }
        
        if dry_run:
            if verbose:
                print("  [DRY RUN] Would create {}.zip with {} files".format(package_name, len(files)))
            return package_info
        
        # Create output directory
        if not os.path.exists(str(self.output_dir)):
            os.makedirs(str(self.output_dir))
        
        zip_path = Path(os.path.join(str(self.output_dir), "{}.zip".format(package_name)))
        
        if verbose:
            print("  Creating {}...".format(os.path.basename(str(zip_path))))
        
        with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                try:
                    # Calculate relative path from project root
                    arcname = os.path.relpath(str(file_path), str(self.project_root))
                    zipf.write(str(file_path), arcname)
                    
                    file_info = {
                        "path": arcname,
                        "size_bytes": self.get_file_size(file_path)
                    }
                    package_info["files"].append(file_info)
                    
                    if verbose:
                        print("    Added: {}".format(arcname))
                        
                except Exception as e:
                    print("    Warning: Could not add {}: {}".format(file_path, e))
        
        package_info["zip_file"] = str(zip_path)
        package_info["zip_size_mb"] = round(os.path.getsize(str(zip_path)) / (1024.0 * 1024.0), 2)
        
        return package_info

    def pack_all(self, exclude_patterns=None, dry_run=False, verbose=False):
        """Pack all packages."""
        if exclude_patterns is None:
            exclude_patterns = self.exclude_patterns
        
        print("Packing TidyLLM project from: {}".format(self.project_root))
        print("Output directory: {}".format(self.output_dir))
        print("Max package size: {}MB".format(self.max_size_mb))
        print("Exclude patterns: {}".format(exclude_patterns))
        print()
        
        all_packages = {}
        total_files = 0
        total_size = 0
        
        for package_name, package_config in self.packages.items():
            print("Processing {}: {}".format(package_name, package_config['description']))
            
            # Collect files for this package
            files = self.collect_files(package_config["paths"], exclude_patterns)
            
            if not files:
                print("  No files found for {}".format(package_name))
                continue
            
            # Create package
            package_info = self.create_zip_package(
                package_name, files, package_config["description"], 
                dry_run, verbose
            )
            
            all_packages[package_name] = package_info
            total_files += package_info["file_count"]
            total_size += package_info["total_size_bytes"]
            
            print("  Files: {}, Size: {}MB".format(package_info['file_count'], package_info['total_size_mb']))
            
            # Check size limit
            if package_info["total_size_mb"] > self.max_size_mb:
                print("  WARNING: Package exceeds size limit of {}MB".format(self.max_size_mb))
            
            print()
        
        # Create summary
        summary = {
            "project_root": str(self.project_root),
            "output_dir": str(self.output_dir),
            "packing_date": datetime.now().isoformat(),
            "total_packages": len(all_packages),
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024.0 * 1024.0), 2),
            "packages": all_packages
        }
        
        # Save summary
        if not dry_run:
            summary_path = Path(os.path.join(str(self.output_dir), "packing_summary.json"))
            with open(str(summary_path), 'w') as f:
                json.dump(summary, f, indent=2)
            print("Packing summary saved to: {}".format(summary_path))
        
        print("\nPacking complete!")
        print("Total packages: {}".format(len(all_packages)))
        print("Total files: {}".format(total_files))
        print("Total size: {}MB".format(round(total_size / (1024.0 * 1024.0), 2)))
        
        return summary

    def create_unpack_script(self, dry_run=False):
        """Create a script to unpack all packages."""
        unpack_script = '''#!/bin/bash
# TidyLLM Project Unpacking Script
# Generated by pack_project.py

set -e

echo "TidyLLM Project Unpacker"
echo "========================"

# Check if packages directory exists
if [ ! -d "packages" ]; then
    echo "Error: packages directory not found!"
    echo "Please run this script from the directory containing the packages folder."
    exit 1
fi

# Create project directory
PROJECT_DIR="TidyLLM_unpacked"
echo "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Unpack all zip files in order
echo "Unpacking packages..."
for zip_file in ../packages/*.zip; do
    if [ -f "$zip_file" ]; then
        echo "  Unpacking $(basename "$zip_file")..."
        unzip -q "$zip_file"
    fi
done

echo "Unpacking complete!"
echo "Project restored to: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_DIR"
echo "2. pip install -e ."
echo "3. python qa_processor.py --setup"
'''
        
        if not dry_run:
            script_path = Path(os.path.join(str(self.output_dir), "unpack.sh"))
            with open(str(script_path), 'w') as f:
                f.write(unpack_script)
            os.chmod(str(script_path), 0755)
            print("Unpack script created: {}".format(script_path))
        
        return unpack_script


def main():
    parser = argparse.ArgumentParser(
        description="Pack TidyLLM project into smaller zip files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--output-dir", default="./packages",
                       help="Output directory for zip files (default: ./packages)")
    parser.add_argument("--max-size", type=int, default=50,
                       help="Maximum size per package in MB (default: 50)")
    parser.add_argument("--exclude", action="append", default=[],
                       help="Exclude files matching pattern (can be used multiple times)")
    parser.add_argument("--include-empty", action="store_true",
                       help="Include empty directories")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be packed without creating files")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed progress")
    parser.add_argument("--project-root", default=".",
                       help="Project root directory (default: current directory)")
    
    args = parser.parse_args()
    
    # Create packer
    packer = ProjectPacker(
        project_root=args.project_root,
        output_dir=args.output_dir,
        max_size_mb=args.max_size
    )
    
    # Add custom exclude patterns
    exclude_patterns = packer.exclude_patterns.copy()
    exclude_patterns.update(args.exclude)
    
    # Pack all packages
    summary = packer.pack_all(
        exclude_patterns=exclude_patterns,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Create unpack script
    if not args.dry_run:
        packer.create_unpack_script(dry_run=args.dry_run)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())