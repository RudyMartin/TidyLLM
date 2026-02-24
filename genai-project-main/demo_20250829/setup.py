#!/usr/bin/env python3
"""
Setup script that creates project directory with custom name
Usage: python setup.py [directory_name]
"""

import sys
import os
import shutil
from pathlib import Path


def setup_project(project_name=None):
    """Setup project in specified directory"""
    
    # Get project name from command line or use default
    if project_name is None:
        if len(sys.argv) > 1:
            project_name = sys.argv[1]
        else:
            project_name = input("Enter project directory name [my_project_dir]: ").strip()
            if not project_name:
                project_name = "my_project_dir"
    
    current_dir = Path.cwd()
    target_dir = current_dir / project_name
    
    print(f"Setting up project in: {target_dir}")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to copy
    files = [
        'config.yaml',
        'simple_postgres_connection.py',
        'requirements_simple.txt', 
        'SIMPLE_DEPLOYMENT.md'
    ]
    
    # Copy files
    for file in files:
        src = current_dir / file
        if src.exists():
            dst = target_dir / file
            shutil.copy2(src, dst)
            print(f"✅ Copied {file}")
    
    print(f"\n🎉 Project setup complete!")
    print(f"📁 Location: {target_dir}")
    print(f"\n📋 Next steps:")
    print(f"1. cd {project_name}")
    print(f"2. Edit config.yaml with your PostgreSQL credentials")
    print(f"3. pip install -r requirements_simple.txt")
    print(f"4. python simple_postgres_connection.py")


if __name__ == "__main__":
    setup_project()