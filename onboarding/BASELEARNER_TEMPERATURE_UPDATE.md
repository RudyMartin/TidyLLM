# BaseLearner Temperature Calculation Update

**Date:** 2025-01-10  
**Status:** ✅ Complete

## Summary

Updated `BaseLearner` to use the **current** temperature formula: **τ = 1/α_ω** (NOT τ = 1/α).

## What Changed

### Old Formula (Deprecated)
```python
tau = 1.0 / alpha  # ❌ NO LONGER USED
```

### New Formula (Current)
```python
# Step 1: Compute alpha_omega (distribution-adjusted quality)
alpha_omega = omega * alpha + (1 - omega) * alpha_prior
# Where alpha_prior = 0.5 (conservative prior for OOD content)

# Step 2: Compute temperature from alpha_omega
tau = 1.0 / alpha_omega  # ✅ CURRENT FORMULA
```

## Formula Details

### α_ω (Alpha Omega) - Distribution-Adjusted Quality

**Formula:**
```
α_ω = ω·α + (1-ω)·α_prior
```

**Where:**
- **α** = Quality score (from `compute_quality()`)
- **ω** = Reliability score (from `compute_omega()`)
  - ω → 1: In-distribution, reliable
  - ω → 0: Out-of-distribution, unreliable
- **α_prior** = 0.5 (conservative prior for OOD content)

**Intuition:**
- When ω = 1 (in-distribution): α_ω = α (trust the quality score)
- When ω = 0 (out-of-distribution): α_ω = 0.5 (use conservative prior)
- When 0 < ω < 1: Blend between quality and prior

### τ (Tau) - Temperature

**Formula:**
```
τ = 1/α_ω
```

**Modes (applied to α_ω):**
- `linear`:  τ = 1/α_ω
- `power`:   τ = 1/α_ω^k (k=2)
- `log`:     τ = 1/log(1 + 10·α_ω)
- `sigmoid`: Bounded smooth mapping

## Changes Made to BaseLearner

### 1. Added `compute_omega()` Method
```python
def compute_omega(self, **kwargs) -> float:
    """
    Compute reliability score ω (omega) from input.
    
    Default implementation returns 1.0 (fully in-distribution).
    Override for domain-specific OOD detection.
    """
    return 1.0  # Default: assume in-distribution
```

### 2. Added `compute_alpha_omega()` Method
```python
def compute_alpha_omega(self, alpha: float, omega: Optional[float] = None, **kwargs) -> float:
    """
    Compute distribution-adjusted quality α_ω.
    
    Formula: α_ω = ω·α + (1-ω)·α_prior
    """
    if omega is None:
        omega = self.compute_omega(**kwargs)
    
    alpha_prior = 0.5
    alpha_omega = omega * alpha + (1 - omega) * alpha_prior
    
    return max(alpha_omega, 0.01)
```

### 3. Updated `compute_temperature()` Method
```python
def compute_temperature(self, alpha_omega: float) -> float:
    """
    Compute temperature from alpha_omega (NOT alpha).
    
    Current formula: τ = 1/α_ω
    """
    # ... applies modes to alpha_omega, not alpha
```

### 4. Updated `predict()` and `learn()` Methods
Both now compute:
1. `alpha = self.compute_quality(**kwargs)`
2. `omega = self.compute_omega(**kwargs)`
3. `alpha_omega = self.compute_alpha_omega(alpha, omega, **kwargs)`
4. `tau = self.compute_temperature(alpha_omega)`  # ✅ Uses alpha_omega

## Updated Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      BaseLearner                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │   SDM   │  │Hopfield │  │   EWC   │  │ Replay  │        │
│  │(a priori)│  │(patterns)│  │(protect)│  │(online) │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│       └────────────┴─────┬──────┴────────────┘              │
│                          │                                   │
│              α_ω = ω·α + (1-ω)·α_prior                       │
│                     τ = 1/α_ω                                │
│              (temperature couples all)                       │
└──────────────────────────────────────────────────────────────┘
```

## Backward Compatibility

- **Default `compute_omega()`** returns 1.0, so existing code continues to work
- When ω = 1.0: α_ω = α, so behavior is identical to old formula
- Subclasses can override `compute_omega()` for domain-specific OOD detection

## Files Updated

- ✅ `yrsn/src/yrsn/core/memory/base_learner.py` - Updated temperature calculation
- ✅ `yrsn/src/yrsn/core/memory/ewc.py` - Updated comment

## Verification

```python
from yrsn.core.memory import BaseLearner
# ✅ BaseLearner updated to use alpha_omega!
```

## References

- Formula source: `yrsn-context/src/yrsn_context/core/types.py:58-66`
- Current usage: `yrsn/demo_000_core/01_benchmark_mock.ipynb:340-345`
- Current usage: `yrsn/demo_001_memristor/04_benchmark_mock.ipynb:284-287`

