#!/usr/bin/env python3
"""
TidyLLM Project Packing Script (Python 3)

- Splits project into logical packages (config below).
- Enforces --max-size by splitting large packages into parts _part01, _part02, ...
- Supports glob-style excludes (e.g., **/__pycache__/**, *.pyc, .venv/**).
- Optional inclusion of empty directories via --include-empty.
- Dry-run and verbose modes for planning.

Usage:
    python pack_project.py [options]
"""

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

class ProjectPacker:
    def __init__(self, project_root, output_dir="./packages", max_size_mb=50, include_empty=False):
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.include_empty = include_empty

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
                    "qa_test_runner.py",
                ],
            },
            "02_knowledge_base": {
                "description": "Knowledge base with PDFs and documents",
                "paths": ["knowledge_base/"],
            },
            "03_scripts_demos": {
                "description": "Scripts, demos, and automation tools",
                "paths": ["scripts/", "drop_zones/", "prompts/"],
            },
            "04_educational_libs": {
                "description": "Educational ML libraries (tidyllm-sentence, tlm)",
                "paths": ["tidyllm-sentence/", "tlm/"],
            },
            "05_tests_docs": {
                "description": "Tests and ecosystem documentation",
                "paths": [
                    "tests/",
                    "paper_repository/",
                    "ECOSYSTEM_*.md",
                    "PACKAGE_SUCCESS_SUMMARY.md",
                    "Under-the-hood-with-flow.md",
                ],
            },
            "06_demo_files": {
                "description": "Demo scripts and flow documentation",
                "paths": [
                    "1-enterprise.py",
                    "2-developer.py",
                    "3-demo.py",
                    "flow_demo.py",
                    "FLOW_README.md",
                ],
            },
        }

        # Default exclusions (glob patterns; matched against POSIX-style relative paths)
        self.default_excludes = {
            "**/__pycache__/**",
            "**/*.pyc",
            "**/*.pyo",
            ".git/**",
            ".gitignore",
            ".DS_Store",
            "**/*.log",
            "**/*.tmp",
            ".pytest_cache/**",
            "**/*.egg-info/**",
            "dist/**",
            "build/**",
            ".venv/**",
            "venv/**",
            "env/**",
            ".env",
        }

    def rel(self, p: Path) -> str:
        return p.relative_to(self.project_root).as_posix()

    def match_any(self, rel_path: str, patterns) -> bool:
        return any(fnmatch(rel_path, pat) for pat in patterns)

    def collect_files(self, paths, exclude_patterns):
        files = []
        # Normalize to POSIX-rel for matching
        for p_str in paths:
            # Support globs for top-level includes too (e.g., ECOSYSTEM_*.md)
            for match in self.project_root.glob(p_str):
                if match.is_file():
                    rp = self.rel(match)
                    if not self.match_any(rp, exclude_patterns):
                        files.append(match)
                elif match.is_dir():
                    # Walk directory
                    for root, dirs, filenames in os.walk(match):
                        root_path = Path(root)
                        rel_root = self.rel(root_path)
                        # Filter dirs in-place to avoid walking excluded ones
                        keep_dirs = []
                        for d in dirs:
                            d_rel = f"{rel_root}/{d}" if rel_root else d
                            if not self.match_any(d_rel + "/**", exclude_patterns):
                                keep_dirs.append(d)
                        dirs[:] = keep_dirs

                        # Optionally include empty directories
                        if self.include_empty and not filenames:
                            # Represent empty dir by adding a trailing slash name
                            # ZipFormat: add a directory entry with no data later
                            pass  # We add directory entries when writing

                        for fname in filenames:
                            fpath = root_path / fname
                            rp = self.rel(fpath)
                            if not self.match_any(rp, exclude_patterns):
                                files.append(fpath)
        # De-dup and sort for stability
        files = sorted(set(files))
        return files

    def file_size(self, p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    def chunk_into_parts(self, files):
        """Greedy size-based bin packing into parts <= max_size_bytes."""
        parts = []
        current = []
        current_size = 0

        # Sort by size desc to reduce parts (simple first-fit-decreasing)
        files_sorted = sorted(files, key=self.file_size, reverse=True)

        for f in files_sorted:
            sz = self.file_size(f)
            # If a single file exceeds limit, place it alone (still creates a part)
            if sz > self.max_size_bytes and not current:
                parts.append([f])
                continue

            if current_size + sz <= self.max_size_bytes:
                current.append(f)
                current_size += sz
            else:
                if current:
                    parts.append(current)
                current = [f]
                current_size = sz

        if current:
            parts.append(current)

        return parts

    def write_zip(self, zip_path: Path, files, dry_run=False, verbose=False):
        info = {
            "zip_file": zip_path.as_posix(),
            "file_count": len(files),
            "files": [],
            "total_size_bytes": 0,
        }

        if dry_run:
            if verbose:
                print(f"  [DRY RUN] Would create {zip_path.name} with {len(files)} files")
            for f in files:
                size = self.file_size(f)
                info["files"].append({"path": self.rel(f), "size_bytes": size})
                info["total_size_bytes"] += size
            info["zip_size_mb"] = round(info["total_size_bytes"] / (1024 * 1024), 2)
            return info

        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  Creating {zip_path.name} ...")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            # Track directories to add explicit entries for empty dirs (optional)
            dir_entries = set()

            for f in files:
                arcname = self.rel(f)
                # Collect parent directories for possible empty-dir entries
                parent = Path(arcname).parent
                while parent and parent.as_posix() != ".":
                    dir_entries.add(parent.as_posix() + "/")
                    parent = parent.parent

                try:
                    z.write(f, arcname)
                    sz = self.file_size(f)
                    info["files"].append({"path": arcname, "size_bytes": sz})
                    info["total_size_bytes"] += sz
                    if verbose:
                        print(f"    Added: {arcname}")
                except Exception as e:
                    print(f"    Warning: Could not add {arcname}: {e}")

            # Optionally ensure empty directories exist in the archive
            if self.include_empty:
                for d in sorted(dir_entries):
                    # If no member starts with this dir, add a directory entry
                    if not any(m.filename.startswith(d) for m in z.infolist()):
                        z.writestr(d, "")  # directory entry

        info["zip_size_mb"] = round(Path(zip_path).stat().st_size / (1024 * 1024), 2)
        return info

    def pack_package(self, name, desc, files, dry_run=False, verbose=False):
        size_bytes = sum(self.file_size(f) for f in files)
        size_mb = round(size_bytes / (1024 * 1024), 2)

        if verbose:
            print(f"  Files collected: {len(files)}, total ~{size_mb} MB")

        parts = self.chunk_into_parts(files) if self.max_size_bytes > 0 else [files]

        summaries = []
        if len(parts) == 1:
            zip_path = self.output_dir / f"{name}.zip"
            summaries.append(self.write_zip(zip_path, parts[0], dry_run, verbose))
        else:
            if verbose:
                print(f"  Splitting into {len(parts)} parts to respect {self.max_size_mb} MB limit")
            for i, part in enumerate(parts, start=1):
                zip_path = self.output_dir / f"{name}_part{i:02d}.zip"
                summaries.append(self.write_zip(zip_path, part, dry_run, verbose))

        # Aggregate
        total_bytes = sum(s["total_size_bytes"] for s in summaries)
        return {
            "name": name,
            "description": desc,
            "created": datetime.now().isoformat(),
            "parts": summaries,
            "file_count": sum(s["file_count"] for s in summaries),
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        }

    def pack_all(self, exclude_patterns=None, dry_run=False, verbose=False):
        excludes = set(self.default_excludes)
        if exclude_patterns:
            excludes.update(exclude_patterns)

        print(f"Packing TidyLLM project from: {self.project_root}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max package size: {self.max_size_mb} MB")
        print(f"Include empty dirs: {self.include_empty}")
        print(f"Exclude patterns: {sorted(excludes)}\n")

        all_packages = {}
        total_files = 0
        total_size_mb = 0.0

        for pkg_name, cfg in self.packages.items():
            print(f"Processing {pkg_name}: {cfg['description']}")
            files = self.collect_files(cfg["paths"], excludes)
            if not files:
                print(f"  No files found for {pkg_name}\n")
                continue

            pkg_summary = self.pack_package(pkg_name, cfg["description"], files, dry_run, verbose)
            all_packages[pkg_name] = pkg_summary
            total_files += pkg_summary["file_count"]
            total_size_mb += pkg_summary["total_size_mb"]

            # Quick line
            parts_n = len(pkg_summary["parts"])
            label = f"{pkg_summary['total_size_mb']} MB in {parts_n} part(s)"
            print(f"  ✅ Packed: {label}\n")

        summary = {
            "project_root": self.project_root.as_posix(),
            "output_dir": self.output_dir.as_posix(),
            "packing_date": datetime.now().isoformat(),
            "total_packages": len(all_packages),
            "total_files": total_files,
            "total_size_mb": round(total_size_mb, 2),
            "packages": all_packages,
        }

        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = self.output_dir / "packing_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Packing summary saved to: {summary_path}")

        print("\nPacking complete!")
        print(f"Total packages: {len(all_packages)}")
        print(f"Total files: {total_files}")
        print(f"Total size: {round(total_size_mb, 2)} MB")

        return summary

    def create_unpack_script(self, dry_run=False):
        script = """#!/bin/bash
# TidyLLM Project Unpacking Script
# Generated by pack_project.py

set -euo pipefail

echo "TidyLLM Project Unpacker"
echo "========================"

PKG_DIR="packages"
if [ ! -d "$PKG_DIR" ]; then
  echo "Error: packages directory not found!"
  echo "Run this script from the directory containing the 'packages' folder."
  exit 1
fi

PROJECT_DIR="TidyLLM_unpacked"
echo "Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "Unpacking packages..."
shopt -s nullglob
for zip_file in ../$PKG_DIR/*.zip; do
  echo "  Unpacking $(basename "$zip_file")..."
  unzip -q "$zip_file"
done

echo "Unpacking complete!"
echo "Project restored to: $PROJECT_DIR"
echo
echo "Next steps:"
echo "1) cd $PROJECT_DIR"
echo "2) pip install -e ."
echo "3) python qa_processor.py --setup || true"
"""
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "unpack.sh"
            path.write_text(script, encoding="utf-8")
            path.chmod(0o755)
            print(f"Unpack script created: {path}")
        return script


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack TidyLLM project into zip packages with size-aware splitting."
    )
    parser.add_argument("--output-dir", default="./packages", help="Output directory (default: ./packages)")
    parser.add_argument("--max-size", type=int, default=50, help="Max size per package in MB (default: 50)")
    parser.add_argument("--exclude", action="append", default=[], help="Glob exclude pattern (repeatable)")
    parser.add_argument("--include-empty", action="store_true", help="Include empty directories")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not write archives")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress")
    parser.add_argument("--project-root", default=".", help="Project root (default: .)")
    return parser.parse_args()


def main():
    args = parse_args()
    packer = ProjectPacker(
        project_root=args.project_root,
        output_dir=args.output_dir,
        max_size_mb=args.max_size,
        include_empty=args.include_empty,
    )

    # Merge excludes (respect user-provided)
    summary = packer.pack_all(
        exclude_patterns=set(args.exclude),
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    if not args.dry_run:
        packer.create_unpack_script(dry_run=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
