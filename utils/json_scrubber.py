#!/usr/bin/env python3
"""
TidyLLM JSON Scrubber Utility
============================

A utility to convert Unicode characters to ASCII equivalents in JSON files
and fix common JSON structure issues to ensure compatibility across different
systems and encodings, particularly for corporate environments.

This addresses the critical issue where Unicode characters in JSON files
cause parsing errors in various tools and environments, and provides
safe automatic fallback for JSON loading operations.
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Standard Unicode to ASCII mappings
UNICODE_TO_ASCII = {
    # Status indicators
    '✅': '[OK]',
    '❌': '[X]',
    '⚠️': '[WARNING]',
    '⏳': '[PENDING]',
    '❗': '[!]',
    '❓': '[?]',

    # Arrows
    '→': '->',
    '←': '<-',
    '↑': '^',
    '↓': 'v',
    '⇒': '=>',
    '⇐': '<=',

    # Bullets and markers
    '•': '-',
    '▪': '-',
    '▫': '-',
    '◦': 'o',
    '‣': '-',
    '⁃': '-',

    # Stars and ratings
    '★': '*',
    '☆': '*',
    '⭐': '*',
    '✨': '*',
    '💫': '*',

    # Shapes
    '●': '*',
    '○': '*',
    '■': '[#]',
    '□': '[ ]',
    '▲': '^',
    '▼': 'v',
    '◆': '<>',
    '◇': '<>',

    # Emojis - Tools and Objects
    '🔧': '[TOOL]',
    '📋': '[DOC]',
    # '🚀': '[ROCKET]',  # Removed for Windows compatibility
    '📊': '[CHART]',
    '📈': '[UP]',
    '📉': '[DOWN]',
    '📥': '[DOWNLOAD]',
    '📤': '[UPLOAD]',
    '🎉': '[CELEBRATION]',
    '💡': '[IDEA]',
    '🔍': '[SEARCH]',
    '💻': '[COMPUTER]',
    '📱': '[PHONE]',
    '🌐': '[WEB]',
    '🔄': '[REFRESH]',
    '🔑': '[KEY]',
    '🔒': '[LOCK]',
    '🔓': '[UNLOCK]',
    '🛡️': '[SHIELD]',
    '📝': '[NOTE]',
    '📁': '[FOLDER]',
    '📂': '[OPEN-FOLDER]',
    '🗂️': '[FILES]',
    '🗃️': '[FILE-BOX]',

    # Emojis - Status and Actions
    '🎯': '[TARGET]',
    '🏆': '[TROPHY]',
    '🚨': '[ALERT]',
    '⚡': '[LIGHTNING]',
    '🔥': '[FIRE]',
    '💰': '[MONEY]',
    '💎': '[GEM]',
    '⚙️': '[GEAR]',
    '🧩': '[PUZZLE]',
    '🎪': '[CIRCUS]',

    # Mathematical symbols
    '≈': '~=',
    '≠': '!=',
    '≤': '<=',
    '≥': '>=',
    '±': '+/-',
    '∞': 'inf',
    '∑': 'sum',
    '∆': 'delta',
    '∇': 'grad',

    # Quotation marks
    '"': '"',
    '"': '"',
    ''': "'",
    ''': "'",
    '«': '<<',
    '»': '>>',

    # Dashes and hyphens
    '–': '-',
    '—': '--',
    '―': '---',
    '‒': '-',

    # Other common symbols
    '…': '...',
    '§': 'section',
    '¶': 'paragraph',
    '†': '+',
    '‡': '++',
    '©': '(c)',
    '®': '(r)',
    '™': '(tm)',
    '℃': 'C',
    '℉': 'F',
    '€': 'EUR',
    '£': 'GBP',
    '¥': 'JPY',
    '¢': 'cent',
}

class JSONScrubber:
    """JSON Unicode to ASCII conversion and structure fixing utility for TidyLLM"""

    def __init__(self, custom_mappings: Optional[Dict[str, str]] = None):
        """
        Initialize the Unicode Scrubber

        Args:
            custom_mappings: Additional Unicode to ASCII mappings
        """
        self.mappings = UNICODE_TO_ASCII.copy()
        if custom_mappings:
            self.mappings.update(custom_mappings)

    def scrub_text(self, text: str, preserve_whitespace: bool = True, fix_json_structure: bool = False) -> Tuple[str, List[str]]:
        """
        Convert Unicode characters in text to ASCII equivalents

        Args:
            text: Input text with potential Unicode characters
            preserve_whitespace: Whether to preserve Unicode whitespace characters
            fix_json_structure: Whether to attempt basic JSON structure fixes

        Returns:
            Tuple of (cleaned_text, list_of_replacements_made)
        """
        replacements_made = []
        cleaned_text = text

        # Apply mapped replacements
        for unicode_char, ascii_replacement in self.mappings.items():
            if unicode_char in cleaned_text:
                count = cleaned_text.count(unicode_char)
                cleaned_text = cleaned_text.replace(unicode_char, ascii_replacement)
                replacements_made.append(f'{unicode_char} -> {ascii_replacement} ({count} times)')

        # Handle any remaining non-ASCII characters
        if not preserve_whitespace:
            # Replace any remaining non-ASCII characters
            remaining_unicode = re.findall(r'[^\x00-\x7F]', cleaned_text)
            if remaining_unicode:
                unique_chars = list(set(remaining_unicode))
                for char in unique_chars:
                    count = cleaned_text.count(char)
                    cleaned_text = cleaned_text.replace(char, '[UNICODE]')
                    replacements_made.append(f'{char} -> [UNICODE] ({count} times)')
        else:
            # Only replace non-whitespace, non-ASCII characters
            remaining_unicode = re.findall(r'[^\x00-\x7F\s]', cleaned_text)
            if remaining_unicode:
                unique_chars = list(set(remaining_unicode))
                for char in unique_chars:
                    count = cleaned_text.count(char)
                    cleaned_text = cleaned_text.replace(char, '[UNICODE]')
                    replacements_made.append(f'{char} -> [UNICODE] ({count} times)')

        # Apply JSON structure fixes if requested
        if fix_json_structure:
            cleaned_text, structure_fixes = self._fix_json_structure(cleaned_text)
            replacements_made.extend(structure_fixes)

        return cleaned_text, replacements_made

    def _fix_json_structure(self, text: str) -> Tuple[str, List[str]]:
        """
        Attempt to fix common JSON structure issues (SAFE fixes only)

        Args:
            text: JSON text that may have structure issues

        Returns:
            Tuple of (fixed_text, list_of_fixes_applied)
        """
        fixes_applied = []
        fixed_text = text

        # Fix 1: Remove dangling quotes followed by newlines (common Unicode cleanup artifact)
        dangling_quote_pattern = r'"\s*\n\s*\n\s*"([a-zA-Z_])'
        matches = re.findall(dangling_quote_pattern, fixed_text)
        if matches:
            fixed_text = re.sub(dangling_quote_pattern, r'"\1', fixed_text)
            fixes_applied.append(f'Fixed {len(matches)} dangling quote(s) with newlines')

        # Fix 2: Remove trailing commas before closing brackets/braces (common JSON issue)
        trailing_comma_pattern = r',\s*([}\]])'
        matches = re.findall(trailing_comma_pattern, fixed_text)
        if matches:
            fixed_text = re.sub(trailing_comma_pattern, r'\1', fixed_text)
            fixes_applied.append(f'Removed {len(matches)} trailing comma(s)')

        # Fix 3: Fix incomplete key-value pairs (dangling quotes)
        incomplete_kv_pattern = r'"\s*\n\s*\n\s*"([^"]*)":'
        matches = re.findall(incomplete_kv_pattern, fixed_text)
        if matches:
            # Only fix if we can be confident about the structure
            for match in matches:
                if match and match.replace('_', '').replace('-', '').isalpha():
                    old_pattern = f'"\\s*\\n\\s*\\n\\s*"{re.escape(match)}":'
                    new_replacement = f'"{match}":'
                    if re.search(old_pattern, fixed_text):
                        fixed_text = re.sub(old_pattern, new_replacement, fixed_text)
                        fixes_applied.append(f'Fixed incomplete key "{match}"')

        # Fix 4: Normalize excessive whitespace between JSON elements
        excessive_whitespace_pattern = r'\n\s*\n\s*\n+'
        if re.search(excessive_whitespace_pattern, fixed_text):
            fixed_text = re.sub(excessive_whitespace_pattern, '\n\n', fixed_text)
            fixes_applied.append('Normalized excessive whitespace')

        # Fix 5: Convert JSON boolean literals to Python boolean literals (for mixed JSON/Python files)
        # This prevents "name 'true' is not defined" errors when JSON booleans are used in Python context
        boolean_fixes = 0
        if ': true' in fixed_text:
            count = fixed_text.count(': true')
            fixed_text = fixed_text.replace(': true', ': True')
            boolean_fixes += count
            fixes_applied.append(f'Converted {count} JSON "true" to Python "True"')

        if ': false' in fixed_text:
            count = fixed_text.count(': false')
            fixed_text = fixed_text.replace(': false', ': False')
            boolean_fixes += count
            fixes_applied.append(f'Converted {count} JSON "false" to Python "False"')

        # Fix 6: Remove control characters that are invalid in JSON
        control_chars = re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', fixed_text)
        if control_chars:
            for char in set(control_chars):
                fixed_text = fixed_text.replace(char, '')
            fixes_applied.append(f'Removed {len(control_chars)} invalid control character(s)')

        return fixed_text, fixes_applied

    def scrub_json_file(self, file_path: str, backup: bool = True) -> Dict[str, any]:
        """
        Scrub a JSON file to remove Unicode characters

        Args:
            file_path: Path to the JSON file
            backup: Whether to create a backup before modifying

        Returns:
            Dictionary with operation results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return {'success': False, 'error': f'File not found: {file_path}'}

        # Create backup if requested
        if backup:
            backup_path = file_path.with_suffix(f'{file_path.suffix}.unicode_backup')
            backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')

        try:
            # Read and process the file
            original_content = file_path.read_text(encoding='utf-8')
            cleaned_content, replacements = self.scrub_text(original_content)

            # Verify it's valid JSON after cleaning
            json.loads(cleaned_content)

            # Write back the cleaned content
            file_path.write_text(cleaned_content, encoding='utf-8')

            return {
                'success': True,
                'file_path': str(file_path),
                'backup_created': backup_path if backup else None,
                'replacements_made': replacements,
                'original_size': len(original_content),
                'cleaned_size': len(cleaned_content),
                'bytes_changed': len(original_content) - len(cleaned_content)
            }

        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'Invalid JSON after cleaning: {e}',
                'file_path': str(file_path)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing file: {e}',
                'file_path': str(file_path)
            }

    def scrub_directory(self, directory_path: str, pattern: str = "*.json", recursive: bool = True) -> List[Dict[str, any]]:
        """
        Scrub all matching files in a directory

        Args:
            directory_path: Path to the directory
            pattern: File pattern to match (e.g., "*.json", "*.txt")
            recursive: Whether to search subdirectories

        Returns:
            List of operation results for each file
        """
        directory_path = Path(directory_path)

        if not directory_path.is_dir():
            return [{'success': False, 'error': f'Directory not found: {directory_path}'}]

        # Find matching files
        if recursive:
            files = list(directory_path.rglob(pattern))
        else:
            files = list(directory_path.glob(pattern))

        results = []
        for file_path in files:
            if pattern.endswith('.json'):
                result = self.scrub_json_file(str(file_path))
            else:
                # For non-JSON files, just do text scrubbing
                try:
                    original_content = file_path.read_text(encoding='utf-8')
                    cleaned_content, replacements = self.scrub_text(original_content)
                    file_path.write_text(cleaned_content, encoding='utf-8')

                    result = {
                        'success': True,
                        'file_path': str(file_path),
                        'replacements_made': replacements,
                        'original_size': len(original_content),
                        'cleaned_size': len(cleaned_content)
                    }
                except Exception as e:
                    result = {
                        'success': False,
                        'error': str(e),
                        'file_path': str(file_path)
                    }

            results.append(result)

        return results

    def add_custom_mapping(self, unicode_char: str, ascii_replacement: str):
        """Add a custom Unicode to ASCII mapping"""
        self.mappings[unicode_char] = ascii_replacement

    def validate_ascii_only(self, file_path: str) -> Dict[str, any]:
        """
        Validate that a file contains only ASCII characters

        Args:
            file_path: Path to the file to validate

        Returns:
            Validation result dictionary
        """
        file_path = Path(file_path)

        try:
            content = file_path.read_text(encoding='utf-8')

            # Find all non-ASCII characters
            non_ascii_chars = re.findall(r'[^\x00-\x7F]', content)

            if not non_ascii_chars:
                return {
                    'is_ascii_only': True,
                    'file_path': str(file_path),
                    'message': 'File contains only ASCII characters'
                }
            else:
                unique_chars = list(set(non_ascii_chars))
                char_counts = {char: content.count(char) for char in unique_chars}

                return {
                    'is_ascii_only': False,
                    'file_path': str(file_path),
                    'non_ascii_characters': char_counts,
                    'total_non_ascii': len(non_ascii_chars),
                    'unique_non_ascii': len(unique_chars)
                }

        except Exception as e:
            return {
                'is_ascii_only': False,
                'error': str(e),
                'file_path': str(file_path)
            }

def convert_json_to_python_values(data):
    """
    Convert JSON standard values to Python equivalents.

    Converts:
    - "false" string to False boolean
    - "true" string to True boolean
    - "null" string to None
    - "none" string to None (case insensitive)

    Args:
        data: JSON data (dict, list, or primitive value)

    Returns:
        Data with converted boolean and null values
    """
    if isinstance(data, dict):
        return {key: convert_json_to_python_values(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_json_to_python_values(item) for item in data]
    elif isinstance(data, str):
        # Convert string representations to Python values
        lower_data = data.lower()
        if lower_data == "false":
            return False
        elif lower_data == "true":
            return True
        elif lower_data in ["null", "none"]:
            return None
        else:
            return data
    else:
        return data

def safe_load_json_with_scrubbing(file_path: str, auto_fix: bool = True) -> Dict[str, any]:
    """
    Safely load JSON file, automatically falling back to Unicode scrubbing if needed

    Args:
        file_path: Path to JSON file
        auto_fix: Whether to automatically apply structure fixes if basic scrubbing fails

    Returns:
        Dictionary with 'data' (if successful) and metadata about the operation
    """
    try:
        # First, try to load normally
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            'success': True,
            'data': data,
            'scrubbing_required': False,
            'fixes_applied': []
        }
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        original_error = str(e)

        # Fallback to scrubbing
        scrubber = JSONScrubber()

        try:
            # First try: Just Unicode replacement
            content = Path(file_path).read_text(encoding='utf-8')
            cleaned_content, replacements = scrubber.scrub_text(content, fix_json_structure=False)
            data = json.loads(cleaned_content)

            return {
                'success': True,
                'data': data,
                'scrubbing_required': True,
                'fixes_applied': replacements,
                'original_error': original_error
            }
        except json.JSONDecodeError:
            if auto_fix:
                try:
                    # Second try: Unicode replacement + structure fixes
                    cleaned_content, replacements = scrubber.scrub_text(content, fix_json_structure=True)
                    data = json.loads(cleaned_content)

                    return {
                        'success': True,
                        'data': data,
                        'scrubbing_required': True,
                        'structure_fixes_required': True,
                        'fixes_applied': replacements,
                        'original_error': original_error
                    }
                except json.JSONDecodeError as final_error:
                    return {
                        'success': False,
                        'error': f'Could not fix JSON even with scrubbing: {final_error}',
                        'original_error': original_error,
                        'fixes_attempted': replacements
                    }
            else:
                return {
                    'success': False,
                    'error': f'JSON invalid after Unicode scrubbing (auto-fix disabled)',
                    'original_error': original_error,
                    'fixes_attempted': replacements
                }

def clean_json_file(input_json: str, output_json: str = None, fix_structure: bool = False) -> Dict[str, any]:
    """
    Simple function to clean a JSON file of Unicode characters

    Args:
        input_json: Path to input JSON file
        output_json: Path to output clean JSON file (defaults to input_json if not provided)
        fix_structure: Whether to attempt JSON structure fixes

    Returns:
        Results dictionary
    """
    scrubber = JSONScrubber()

    if output_json is None:
        output_json = input_json

    # Read input
    input_path = Path(input_json)
    if not input_path.exists():
        return {'success': False, 'error': f'Input file not found: {input_json}'}

    try:
        original_content = input_path.read_text(encoding='utf-8')
        cleaned_content, replacements = scrubber.scrub_text(original_content, fix_json_structure=fix_structure)

        # Verify valid JSON
        json.loads(cleaned_content)

        # Write output
        output_path = Path(output_json)
        output_path.write_text(cleaned_content, encoding='utf-8')

        return {
            'success': True,
            'input_file': input_json,
            'output_file': output_json,
            'replacements_made': replacements,
            'bytes_saved': len(original_content) - len(cleaned_content),
            'structure_fixes_applied': fix_structure
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'input_file': input_json
        }

def main():
    """Command line interface for the JSON Scrubber"""
    parser = argparse.ArgumentParser(description='TidyLLM JSON Scrubber - Clean and fix JSON files')
    parser.add_argument('input_json', help='Input JSON file path')
    parser.add_argument('output_json', nargs='?', help='Output clean JSON file path (optional)')
    parser.add_argument('--validate-only', action='store_true', help='Only validate ASCII compliance')
    parser.add_argument('--fix-structure', action='store_true', help='Attempt to fix JSON structure issues')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    scrubber = JSONScrubber()

    if args.validate_only:
        # Validation mode
        result = scrubber.validate_ascii_only(args.input_json)
        print(f"Validation result for {args.input_json}:")
        print(f"  ASCII only: {result['is_ascii_only']}")
        if not result['is_ascii_only'] and 'non_ascii_characters' in result:
            print(f"  Non-ASCII characters found: {result['total_non_ascii']}")
            if args.verbose:
                for char, count in result['non_ascii_characters'].items():
                    print(f"    '{char}': {count} occurrences")
    else:
        # Clean JSON file
        result = clean_json_file(args.input_json, args.output_json, fix_structure=args.fix_structure)

        print(f"Unicode Scrubber Results:")
        print(f"  Input: {result.get('input_file', args.input_json)}")
        print(f"  Output: {result.get('output_file', args.output_json or args.input_json)}")
        print(f"  Success: {result['success']}")
        if result.get('structure_fixes_applied'):
            print(f"  Structure fixes: ENABLED")

        if result['success']:
            replacements = result.get('replacements_made', [])
            print(f"  Replacements made: {len(replacements)}")
            print(f"  Bytes saved: {result.get('bytes_saved', 0)}")

            # Separate structure fixes from Unicode replacements for clarity
            unicode_fixes = [r for r in replacements if not r.startswith('Fixed') and not r.startswith('Removed') and not r.startswith('Normalized')]
            structure_fixes = [r for r in replacements if r.startswith('Fixed') or r.startswith('Removed') or r.startswith('Normalized')]

            if args.verbose and (unicode_fixes or structure_fixes):
                if unicode_fixes:
                    print("  Unicode replacements:")
                    for fix in unicode_fixes:
                        print(f"    {fix}")
                if structure_fixes:
                    print("  Structure fixes:")
                    for fix in structure_fixes:
                        print(f"    {fix}")
        else:
            print(f"  Error: {result['error']}")
            if args.fix_structure:
                print("  Tip: Structure fixing was enabled but couldn't resolve the JSON error")

if __name__ == "__main__":
    main()