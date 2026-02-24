#!/usr/bin/env python3
"""
Clean JSON I/O Utility for TidyLLM
=================================

Provides automatic JSON cleaning hooks for load and save operations using the
json_scrubber utility. This ensures all JSON files remain clean and ASCII-compliant
throughout the application lifecycle.

Usage:
    from tidyllm.utils.clean_json_io import load_json, save_json

    # Load JSON with automatic cleaning
    data = load_json("workflow.json")

    # Save JSON with automatic cleaning
    save_json(data, "workflow.json")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .json_scrubber import safe_load_json_with_scrubbing, JSONScrubber

logger = logging.getLogger(__name__)

class CleanJSONIO:
    """Clean JSON I/O handler with automatic scrubbing"""

    def __init__(self, auto_clean: bool = True, backup_on_write: bool = False):
        """
        Initialize Clean JSON I/O handler

        Args:
            auto_clean: Whether to automatically clean JSON on load/save
            backup_on_write: Whether to create backup files before writing
        """
        self.auto_clean = auto_clean
        self.backup_on_write = backup_on_write
        self.scrubber = JSONScrubber() if auto_clean else None

    def load_json(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Load JSON file with automatic cleaning if needed

        Args:
            file_path: Path to JSON file
            encoding: File encoding (default: utf-8)

        Returns:
            Loaded JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid even after cleaning
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        if self.auto_clean:
            # Use safe loading with automatic scrubbing
            result = safe_load_json_with_scrubbing(str(file_path))

            if result['success']:
                if result.get('scrubbing_required'):
                    logger.info(f"JSON file required cleaning: {file_path}")
                    if result.get('fixes_applied'):
                        logger.debug(f"Fixes applied: {result['fixes_applied']}")
                return result['data']
            else:
                raise ValueError(f"Failed to load JSON file {file_path}: {result['error']}")
        else:
            # Standard JSON loading
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)

    def save_json(self, data: Any, file_path: Union[str, Path],
                  encoding: str = 'utf-8', indent: int = 2,
                  ensure_ascii: bool = True) -> Dict[str, Any]:
        """
        Save JSON file with automatic cleaning

        Args:
            data: Data to save as JSON
            file_path: Path to save JSON file
            encoding: File encoding (default: utf-8)
            indent: JSON indentation (default: 2)
            ensure_ascii: Whether to ensure ASCII output (default: True)

        Returns:
            Save operation metadata
        """
        file_path = Path(file_path)

        # Create backup if requested
        backup_path = None
        if self.backup_on_write and file_path.exists():
            backup_path = file_path.with_suffix(f'{file_path.suffix}.bak')
            backup_path.write_text(file_path.read_text(encoding=encoding), encoding=encoding)
            logger.debug(f"Created backup: {backup_path}")

        try:
            # Convert to JSON string
            json_content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

            if self.auto_clean:
                # Clean the JSON content before saving
                cleaned_content, fixes = self.scrubber.scrub_text(
                    json_content, fix_json_structure=True
                )

                # Verify it's still valid JSON after cleaning
                json.loads(cleaned_content)

                # Write cleaned content
                file_path.write_text(cleaned_content, encoding=encoding)

                result = {
                    'success': True,
                    'file_path': str(file_path),
                    'backup_created': str(backup_path) if backup_path else None,
                    'cleaning_applied': len(fixes) > 0,
                    'fixes_applied': fixes,
                    'bytes_written': len(cleaned_content)
                }

                if fixes:
                    logger.info(f"Applied {len(fixes)} fixes while saving {file_path}")
                    logger.debug(f"Fixes: {fixes}")

                return result
            else:
                # Write without cleaning
                file_path.write_text(json_content, encoding=encoding)

                return {
                    'success': True,
                    'file_path': str(file_path),
                    'backup_created': str(backup_path) if backup_path else None,
                    'cleaning_applied': False,
                    'fixes_applied': [],
                    'bytes_written': len(json_content)
                }

        except Exception as e:
            # Restore backup if something went wrong
            if backup_path and backup_path.exists():
                file_path.write_text(backup_path.read_text(encoding=encoding), encoding=encoding)
                logger.error(f"Restored backup after save failure: {backup_path}")

            raise ValueError(f"Failed to save JSON file {file_path}: {e}")

# Global instance for convenient usage
_global_clean_io = CleanJSONIO(auto_clean=True, backup_on_write=False)

def load_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load JSON file with automatic cleaning

    Convenience function using global CleanJSONIO instance

    Args:
        file_path: Path to JSON file
        encoding: File encoding (default: utf-8)

    Returns:
        Loaded JSON data
    """
    return _global_clean_io.load_json(file_path, encoding)

def save_json(data: Any, file_path: Union[str, Path],
              encoding: str = 'utf-8', indent: int = 2,
              ensure_ascii: bool = True) -> Dict[str, Any]:
    """
    Save JSON file with automatic cleaning

    Convenience function using global CleanJSONIO instance

    Args:
        data: Data to save as JSON
        file_path: Path to save JSON file
        encoding: File encoding (default: utf-8)
        indent: JSON indentation (default: 2)
        ensure_ascii: Whether to ensure ASCII output (default: True)

    Returns:
        Save operation metadata
    """
    return _global_clean_io.save_json(data, file_path, encoding, indent, ensure_ascii)

def configure_clean_io(auto_clean: bool = True, backup_on_write: bool = False):
    """
    Configure global CleanJSONIO behavior

    Args:
        auto_clean: Whether to automatically clean JSON on load/save
        backup_on_write: Whether to create backup files before writing
    """
    global _global_clean_io
    _global_clean_io = CleanJSONIO(auto_clean=auto_clean, backup_on_write=backup_on_write)

# Context manager for temporary configuration
class clean_json_context:
    """Context manager for temporary CleanJSONIO configuration"""

    def __init__(self, auto_clean: bool = True, backup_on_write: bool = False):
        self.temp_config = (auto_clean, backup_on_write)
        self.original_io = None

    def __enter__(self):
        global _global_clean_io
        self.original_io = _global_clean_io
        _global_clean_io = CleanJSONIO(*self.temp_config)
        return _global_clean_io

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _global_clean_io
        _global_clean_io = self.original_io

# Example usage:
if __name__ == "__main__":
    # Basic usage
    data = {"test": "value", "unicode": "✅ clean"}
    save_json(data, "test.json")
    loaded_data = load_json("test.json")
    print(f"Loaded: {loaded_data}")

    # Context manager usage
    with clean_json_context(auto_clean=True, backup_on_write=True):
        save_json({"important": "data"}, "important.json")