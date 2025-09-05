"""
Debug Configuration System

Provides configurable debug and logging options to control system behavior
and performance impact. Allows fine-grained control over what gets logged
and when to optimize for production environments.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DebugConfig:
    """Debug configuration settings."""
    
    # Master debug flag - controls all debug features
    debug_full: bool = False
    
    # Logging controls
    write_local_logs: bool = False
    log_performance_metrics: bool = False
    log_dspy_errors: bool = False
    log_fallback_usage: bool = False
    log_context_errors: bool = False
    log_database_errors: bool = False
    log_import_errors: bool = False
    log_sme_analysis: bool = False
    
    # Performance controls
    enable_timing: bool = False
    detailed_context_logging: bool = False
    log_json_pretty_print: bool = False
    
    # Output controls
    console_debug: bool = False
    verbose_errors: bool = False
    
    def __post_init__(self):
        """Apply debug_full override if enabled."""
        if self.debug_full:
            self.write_local_logs = True
            self.log_performance_metrics = True
            self.log_dspy_errors = True
            self.log_fallback_usage = True
            self.log_context_errors = True
            self.log_database_errors = True
            self.log_import_errors = True
            self.log_sme_analysis = True
            self.enable_timing = True
            self.detailed_context_logging = True
            self.log_json_pretty_print = True
            self.console_debug = True
            self.verbose_errors = True
    
    @classmethod
    def from_env(cls) -> 'DebugConfig':
        """Create debug config from environment variables."""
        return cls(
            debug_full=os.getenv('DEBUG_FULL', 'false').lower() == 'true',
            write_local_logs=os.getenv('WRITE_LOCAL_LOGS', 'false').lower() == 'true',
            log_performance_metrics=os.getenv('LOG_PERFORMANCE_METRICS', 'false').lower() == 'true',
            log_dspy_errors=os.getenv('LOG_DSPY_ERRORS', 'false').lower() == 'true',
            log_fallback_usage=os.getenv('LOG_FALLBACK_USAGE', 'false').lower() == 'true',
            log_context_errors=os.getenv('LOG_CONTEXT_ERRORS', 'false').lower() == 'true',
            log_database_errors=os.getenv('LOG_DATABASE_ERRORS', 'false').lower() == 'true',
            log_import_errors=os.getenv('LOG_IMPORT_ERRORS', 'false').lower() == 'true',
            log_sme_analysis=os.getenv('LOG_SME_ANALYSIS', 'false').lower() == 'true',
            enable_timing=os.getenv('ENABLE_TIMING', 'false').lower() == 'true',
            detailed_context_logging=os.getenv('DETAILED_CONTEXT_LOGGING', 'false').lower() == 'true',
            log_json_pretty_print=os.getenv('LOG_JSON_PRETTY_PRINT', 'false').lower() == 'true',
            console_debug=os.getenv('CONSOLE_DEBUG', 'false').lower() == 'true',
            verbose_errors=os.getenv('VERBOSE_ERRORS', 'false').lower() == 'true'
        )
    
    @classmethod
    def production(cls) -> 'DebugConfig':
        """Create production-optimized config (minimal logging for critical errors only)."""
        return cls(
            debug_full=False,
            write_local_logs=True,  # Keep minimal logging for critical errors
            log_performance_metrics=False,  # HIGH OVERHEAD - disable in production
            log_dspy_errors=True,  # Important for troubleshooting
            log_fallback_usage=True,  # Important for monitoring system reliability
            log_context_errors=True,  # Critical errors
            log_database_errors=True,  # Critical errors
            log_import_errors=True,  # Critical errors  
            log_sme_analysis=False,  # HIGH OVERHEAD - skip detailed analysis logs
            enable_timing=False,  # Not needed in production
            detailed_context_logging=False,  # Reduces payload size
            log_json_pretty_print=False,  # Faster, more compact (saves ~5% overhead)
            console_debug=False,  # No console output needed
            verbose_errors=False  # Keep error messages concise
        )
    
    @classmethod
    def production_minimal(cls) -> 'DebugConfig':
        """Create ultra-minimal production config (errors only)."""
        return cls(
            debug_full=False,
            write_local_logs=True,
            log_performance_metrics=False,
            log_dspy_errors=True,  # Only critical DSPy errors
            log_fallback_usage=False,  # Disable to reduce overhead
            log_context_errors=True,  # Only critical context errors
            log_database_errors=True,  # Only critical database errors
            log_import_errors=True,  # Only critical import errors
            log_sme_analysis=False,  # Disable completely
            enable_timing=False,
            detailed_context_logging=False,
            log_json_pretty_print=False,
            console_debug=False,
            verbose_errors=False
        )
    
    @classmethod
    def development(cls) -> 'DebugConfig':
        """Create development-optimized config (full logging)."""
        return cls(debug_full=True)
    
    @classmethod
    def performance_test(cls) -> 'DebugConfig':
        """Create config for performance testing (no logging)."""
        return cls(
            debug_full=False,
            write_local_logs=False,
            log_performance_metrics=False,
            log_dspy_errors=False,
            log_fallback_usage=False,
            log_context_errors=False,
            log_database_errors=False,
            log_import_errors=False,
            log_sme_analysis=False,
            enable_timing=True,  # Keep timing for performance measurement
            detailed_context_logging=False,
            log_json_pretty_print=False,
            console_debug=False,
            verbose_errors=False
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'debug_full': self.debug_full,
            'write_local_logs': self.write_local_logs,
            'log_performance_metrics': self.log_performance_metrics,
            'log_dspy_errors': self.log_dspy_errors,
            'log_fallback_usage': self.log_fallback_usage,
            'log_context_errors': self.log_context_errors,
            'log_database_errors': self.log_database_errors,
            'log_import_errors': self.log_import_errors,
            'log_sme_analysis': self.log_sme_analysis,
            'enable_timing': self.enable_timing,
            'detailed_context_logging': self.detailed_context_logging,
            'log_json_pretty_print': self.log_json_pretty_print,
            'console_debug': self.console_debug,
            'verbose_errors': self.verbose_errors
        }
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return f"DebugConfig(debug_full={self.debug_full}, write_local_logs={self.write_local_logs})"


# Global debug config instance
_debug_config: Optional[DebugConfig] = None


def get_debug_config() -> DebugConfig:
    """Get the current debug configuration."""
    global _debug_config
    if _debug_config is None:
        _debug_config = DebugConfig.from_env()
    return _debug_config


def set_debug_config(config: DebugConfig) -> None:
    """Set the debug configuration."""
    global _debug_config
    _debug_config = config


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled."""
    return get_debug_config().debug_full


def should_log(log_type: str) -> bool:
    """Check if a specific log type should be logged."""
    config = get_debug_config()
    
    log_type_mapping = {
        'performance_metrics': config.log_performance_metrics,
        'dspy_errors': config.log_dspy_errors,
        'fallback_usage': config.log_fallback_usage,
        'context_errors': config.log_context_errors,
        'database_errors': config.log_database_errors,
        'import_errors': config.log_import_errors,
        'sme_analysis': config.log_sme_analysis
    }
    
    return log_type_mapping.get(log_type, False)


def get_json_indent() -> Optional[int]:
    """Get JSON indent for logging (None for compact, 2 for pretty)."""
    config = get_debug_config()
    return 2 if config.log_json_pretty_print else None
