#!/usr/bin/env python3
"""
Mermaid to PNG Converter Utility

Converts Mermaid diagrams from markdown files to PNG images using mermaid-cli.

Requirements:
    - Node.js installed
    - npm install -g @mermaid-js/mermaid-cli

Usage:
    python mermaid_to_png.py architecture.md
    python mermaid_to_png.py --input architecture.md --output diagram.png
    python mermaid_to_png.py --help
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
import tempfile
import os


def extract_mermaid_from_markdown(md_file):
    """Extract mermaid code blocks from markdown file."""
    content = Path(md_file).read_text(encoding='utf-8')
    
    # Find mermaid code blocks
    mermaid_pattern = r'```mermaid\s*\n(.*?)\n```'
    matches = re.findall(mermaid_pattern, content, re.DOTALL)
    
    if not matches:
        print(f"No mermaid diagrams found in {md_file}")
        return None
    
    if len(matches) > 1:
        print(f"Found {len(matches)} mermaid diagrams. Using the first one.")
    
    return matches[0].strip()


def check_mermaid_cli():
    """Check if mermaid-cli is installed."""
    try:
        subprocess.run(['mmdc', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_mermaid_cli():
    """Install mermaid-cli via npm."""
    print("Installing @mermaid-js/mermaid-cli...")
    try:
        subprocess.run(['npm', 'install', '-g', '@mermaid-js/mermaid-cli'], check=True)
        return True
    except subprocess.CalledProcessError:
        print("Failed to install mermaid-cli. Please install manually:")
        print("npm install -g @mermaid-js/mermaid-cli")
        return False


def convert_mermaid_to_png(mermaid_content, output_path, width=1600, height=1200):
    """Convert mermaid content to high-resolution PNG using mermaid-cli."""
    
    # Create temporary mermaid file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
        f.write(mermaid_content)
        temp_mermaid = f.name
    
    try:
        # Convert to high-resolution PNG
        cmd = [
            'mmdc', 
            '-i', temp_mermaid, 
            '-o', str(output_path), 
            '-t', 'neutral', 
            '-b', 'white',
            '-w', str(width),      # High resolution width
            '-H', str(height),     # High resolution height
            '--scale', '2'         # 2x scale factor for even higher quality
        ]
        subprocess.run(cmd, check=True)
        print(f"Successfully converted to high-resolution PNG: {output_path} ({width}x{height}, 2x scale)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting mermaid to PNG: {e}")
        return False
        
    finally:
        # Clean up temporary file
        os.unlink(temp_mermaid)


def main():
    parser = argparse.ArgumentParser(description='Convert Mermaid diagrams to PNG')
    parser.add_argument('input', nargs='?', help='Input markdown file containing mermaid diagram')
    parser.add_argument('-i', '--input', dest='input_file', help='Input markdown file')
    parser.add_argument('-o', '--output', help='Output PNG file (default: same name as input with .png extension)')
    parser.add_argument('--install', action='store_true', help='Install mermaid-cli if not present')
    parser.add_argument('--width', type=int, default=1600, help='Output width in pixels (default: 1600)')
    parser.add_argument('--height', type=int, default=1200, help='Output height in pixels (default: 1200)')
    
    args = parser.parse_args()
    
    # Handle input file
    input_file = args.input or args.input_file
    if not input_file:
        parser.print_help()
        return 1
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Input file not found: {input_file}")
        return 1
    
    # Check mermaid-cli
    if not check_mermaid_cli():
        if args.install:
            if not install_mermaid_cli():
                return 1
        else:
            print("mermaid-cli not found. Install it with:")
            print("npm install -g @mermaid-js/mermaid-cli")
            print("Or run with --install flag")
            return 1
    
    # Extract mermaid content
    mermaid_content = extract_mermaid_from_markdown(input_file)
    if not mermaid_content:
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.png')
    
    # Convert to PNG
    if convert_mermaid_to_png(mermaid_content, output_path, args.width, args.height):
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())