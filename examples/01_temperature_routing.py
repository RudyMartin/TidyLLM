"""
Example 1: Temperature-Based Routing

Demonstrates how temperature controls reasoning mode in TensorLogicService
"""

import sys
import os
# Add sibling packages to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, 'tlm'))
sys.path.insert(0, os.path.join(parent_dir, 'tidyllm-sentence'))

from reasoning import TensorLogicService, ReasoningMode, TemperatureRouter

print("=" * 60)
print("TidyLLM: Temperature-Based Routing")
print("=" * 60)

# Example 1: Understanding the Router
print("\n1. Temperature Router Basics")
print("-" * 40)

router = TemperatureRouter(symbolic_threshold=0.05, hybrid_threshold=0.5)

test_temps = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]

print("Temperature → Reasoning Mode mapping:\n")
for temp in test_temps:
    mode = router.get_mode(temp)
    weights = router.get_weights(temp)
    print(f"T={temp:4.2f} → {mode.value:12s} ", end="")
    print(f"(symbolic: {weights['symbolic']:.2f}, analogical: {weights['analogical']:.2f})")

# Example 2: Symbolic Reasoning (T≈0)
print("\n\n2. Symbolic Reasoning (T≈0)")
print("-" * 40)
print("Certifiable, rule-based, deterministic\n")

service = TensorLogicService()

query = "Is data validation required?"

result = service.infer(query, temperature=0.0)

print(f"Query: {query}")
print(f"Temperature: 0.0")
print(f"Mode: {result['reasoning_mode']}")
print(f"Certifiable: {result['certifiable']}")
print(f"Confidence: {result['confidence']}")
print(f"Answer: {result['answer']}")

# Example 3: Analogical Reasoning (T≥0.5)
print("\n\n3. Analogical Reasoning (T≥0.5)")
print("-" * 40)
print("Case-based, similarity-driven, probabilistic\n")

cases = [
    "Data validation is required for MVS compliance",
    "Schema checks must be performed",
    "Code review ensures quality"
]

service_with_cases = TensorLogicService(case_base=cases)

result = service_with_cases.infer(query, temperature=0.7)

print(f"Query: {query}")
print(f"Temperature: 0.7")
print(f"Mode: {result['reasoning_mode']}")
print(f"Certifiable: {result['certifiable']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Answer: {result['answer']}")

# Example 4: Hybrid Reasoning (0<T<0.5)
print("\n\n4. Hybrid Reasoning (0<T<0.5)")
print("-" * 40)
print("Weighted combination of symbolic + analogical\n")

result = service_with_cases.infer(query, temperature=0.3)

print(f"Query: {query}")
print(f"Temperature: 0.3")
print(f"Mode: {result['reasoning_mode']}")
print(f"Certifiable: {result['certifiable']}")
print(f"Confidence: {result['confidence']:.3f}")

# Show component details
if 'components' in result and 'weights' in result['components']:
    weights = result['components']['weights']
    print(f"\nMixing weights:")
    print(f"  Symbolic:    {weights['symbolic']:.2f}")
    print(f"  Analogical:  {weights['analogical']:.2f}")

    if 'symbolic' in result['components']:
        sym = result['components']['symbolic']
        ana = result['components']['analogical']
        print(f"\nComponent confidences:")
        print(f"  Symbolic:    {sym['confidence']:.3f}")
        print(f"  Analogical:  {ana['confidence']:.3f}")

# Example 5: Temperature Sweep
print("\n\n5. Temperature Sweep Analysis")
print("-" * 40)
print("Observe how mode and confidence change with temperature\n")

query = "What validation is needed?"
temperatures = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

print(f"Query: {query}\n")
print(f"{'Temp':>6} {'Mode':>12} {'Cert':>5} {'Confidence':>10}")
print("-" * 40)

for temp in temperatures:
    result = service_with_cases.infer(query, temperature=temp)
    cert_str = "Yes" if result['certifiable'] else "No"
    print(f"{temp:6.1f} {result['reasoning_mode']:>12} {cert_str:>5} {result['confidence']:>10.3f}")

# Example 6: Custom Thresholds
print("\n\n6. Custom Temperature Thresholds")
print("-" * 40)

# Wider hybrid range
router_wide = TemperatureRouter(symbolic_threshold=0.1, hybrid_threshold=0.7)

# Narrow hybrid range
router_narrow = TemperatureRouter(symbolic_threshold=0.02, hybrid_threshold=0.3)

test_temp = 0.2

print(f"Testing temperature: {test_temp}\n")

mode_wide = router_wide.get_mode(test_temp)
weights_wide = router_wide.get_weights(test_temp)

mode_narrow = router_narrow.get_mode(test_temp)
weights_narrow = router_narrow.get_weights(test_temp)

print("Wide hybrid range (0.1 - 0.7):")
print(f"  Mode: {mode_wide.value}")
print(f"  Weights: symbolic={weights_wide['symbolic']:.2f}, analogical={weights_wide['analogical']:.2f}")

print("\nNarrow hybrid range (0.02 - 0.3):")
print(f"  Mode: {mode_narrow.value}")
print(f"  Weights: symbolic={weights_narrow['symbolic']:.2f}, analogical={weights_narrow['analogical']:.2f}")

print("\n" + "=" * 60)
print("Temperature routing examples completed!")
print("=" * 60)
