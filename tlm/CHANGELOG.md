# Changelog

All notable changes to TLM (Transparent Learning Machines) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2025-09-09

### Added
- **Automatic type detection** for critical mathematical functions
- `_is_scalar()` helper function for consistent type checking across TLM
- **Enhanced `dot()` function**: Now intelligently handles vectors, matrices, and scalars like numpy.dot
- **Matrix support in mathematical functions**: All core functions now handle scalars, vectors, and matrices
- **Modern entropy-based seeding**: Replaced 1950s-era fixed seeding with entropy-based auto-initialization
- **Historical RNG algorithms**: Pure Python implementations of Park & Miller LCG, Marsaglia Xorshift, Box-Muller transform
- **Seed mixing functions**: Hash-based seed improvement for better distribution quality
- **Lazy initialization**: Automatic high-entropy seeding on first use of random functions
- **Matrix norms**: Frobenius norm support for matrices in `norm()` function

### Changed
- **`dot()` function**: Now automatically detects input types and routes between vector dot product and matrix multiplication
- **Random number generation**: Enhanced with modern seeding practices while preserving historical algorithms
- **All statistical functions**: Now gracefully handle scalar inputs where mathematically appropriate
- **Function signatures**: Relaxed type annotations to support automatic type detection

### Fixed
- **`argmax()` function**: Completely rewritten to fix conflict with TLM's `max()` function
  - Now supports scalars (→0), vectors (→index), and matrices (→index arrays)
- **`std()` function**: No longer fails on scalar inputs (returns 0.0)
- **`sum()` function**: Fixed recursive call conflicts and generator iteration issues with matrices
- **`transpose()` function**: Now converts vectors to column matrices instead of failing
- **`flatten()` function**: Handles scalar inputs by converting to single-element lists
- **`norm()` function**: Added matrix support using Frobenius norm, scalar support using absolute value
- **`var()` function**: Added scalar case returning 0.0 (no variance from self)

### Technical
- **Zero-dependency compliance**: Replaced all remaining external imports with pure Python implementations
- **Removed conflicts**: Fixed issues where TLM functions conflicted with Python built-ins
- **Enhanced error handling**: Better error messages for unsupported operations
- **Type detection pattern**: Established consistent pattern for future function enhancements
- **Pure math implementations**: All mathematical operations now use internal implementations

### Compatibility
- **Backward compatible**: All existing code continues to work unchanged
- **NumPy-like API**: Enhanced compatibility with NumPy conventions while maintaining pure Python
- **Enhanced robustness**: Functions now handle edge cases and mixed input types gracefully

## [1.0.1] - Previous Release
- Initial stable release of TLM
- Pure Python machine learning algorithms
- Zero external dependencies
- Basic mathematical operations and ML algorithms