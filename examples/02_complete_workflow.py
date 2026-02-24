"""
Example 2: Complete Tensor Logic Workflow

Demonstrates end-to-end usage of the TensorLogicService for a compliance checking scenario
"""

import sys
import os
# Add sibling packages to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, 'tlm'))
sys.path.insert(0, os.path.join(parent_dir, 'tidyllm-sentence'))

from reasoning import create_reasoner

print("=" * 60)
print("TidyLLM: Complete Tensor Logic Workflow")
print("=" * 60)

# Example 1: Compliance Checking Scenario
print("\n1. Compliance Checking Scenario")
print("-" * 40)

# Define compliance rules (for symbolic reasoning)
rules = [
    {"rule": "data_validation_required", "condition": "all_data_sources"},
    {"rule": "schema_validation_required", "condition": "before_processing"},
]

# Define compliance cases (for analogical reasoning)
cases = [
    "Data validation is required for all MVS compliance workflows",
    "Schema validation must be performed before data processing",
    "All input data must undergo quality checks",
    "Validation errors must be logged and reported",
    "Data sources require authentication and authorization",
    "Processing pipelines must include error handling",
    "Compliance reports must be generated for all transactions",
    "Audit trails are mandatory for regulatory compliance"
]

# Create reasoner with both symbolic rules and analogical cases
reasoner = create_reasoner(rules=rules, cases=cases, embedding_method='lsa')

print("Knowledge Base:")
print(f"  Symbolic rules: {len(rules)}")
print(f"  Analogical cases: {len(cases)}")
print(f"  Embedding method: LSA")

# Example 2: Symbolic Mode (T=0) - Certifiable Answers
print("\n\n2. Symbolic Mode (T=0.0)")
print("-" * 40)
print("Use for: Formal verification, rule checking\n")

queries = [
    "Is data validation required?",
    "Must schemas be validated?",
    "Are compliance reports needed?"
]

for query in queries:
    result = reasoner.infer(query, temperature=0.0)
    print(f"Q: {query}")
    print(f"   Mode: {result['reasoning_mode']}, Certifiable: {result['certifiable']}")
    print(f"   → {result['answer'][:60]}...")
    print()

# Example 3: Analogical Mode (T=0.7) - Case-Based Answers
print("\n3. Analogical Mode (T=0.7)")
print("-" * 40)
print("Use for: Finding similar cases, precedent-based reasoning\n")

queries = [
    "What validation steps are needed?",
    "How to handle input data?",
    "What about error handling?"
]

for query in queries:
    result = reasoner.infer(query, temperature=0.7)
    print(f"Q: {query}")
    print(f"   Mode: {result['reasoning_mode']}, Confidence: {result['confidence']:.3f}")
    print(f"   → {result['answer'][:60]}...")
    print()

# Example 4: Hybrid Mode (T=0.3) - Combined Reasoning
print("\n4. Hybrid Mode (T=0.3)")
print("-" * 40)
print("Use for: Balanced reasoning, combining rules + cases\n")

query = "What are the data quality requirements?"

result = reasoner.infer(query, temperature=0.3)

print(f"Query: {query}")
print(f"Mode: {result['reasoning_mode']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"\nAnswer: {result['answer'][:100]}...")

if 'components' in result and 'weights' in result['components']:
    weights = result['components']['weights']
    print(f"\nReasoning breakdown:")
    print(f"  {weights['symbolic']*100:.0f}% Symbolic (rules)")
    print(f"  {weights['analogical']*100:.0f}% Analogical (cases)")

# Example 5: Decision Making with Temperature Sweep
print("\n\n5. Decision Making with Temperature Sweep")
print("-" * 40)
print("Analyze the same query at different temperatures\n")

query = "Is validation mandatory for compliance?"

print(f"Query: {query}\n")
print(f"{'Temp':>6} {'Mode':>12} {'Cert':>5} {'Conf':>6} {'Answer Preview':>40}")
print("-" * 80)

for temp in [0.0, 0.2, 0.4, 0.6, 0.8]:
    result = reasoner.infer(query, temperature=temp)
    cert = "✓" if result['certifiable'] else "✗"
    answer_preview = result['answer'][:35]
    print(f"{temp:6.1f} {result['reasoning_mode']:>12} {cert:>5} {result['confidence']:>6.3f} {answer_preview:>40}")

# Example 6: Practical Use Case - Batch Compliance Checking
print("\n\n6. Batch Compliance Checking")
print("-" * 40)

compliance_questions = [
    ("Is data validation required?", 0.0),           # Strict rule check
    ("What validation methods exist?", 0.7),         # Find similar cases
    ("How comprehensive must validation be?", 0.3),  # Hybrid approach
]

print("Checking multiple compliance requirements:\n")

for i, (question, temp) in enumerate(compliance_questions, 1):
    result = reasoner.infer(question, temperature=temp)

    print(f"{i}. {question}")
    print(f"   Temperature: {temp} ({result['reasoning_mode']})")
    print(f"   Certifiable: {result['certifiable']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Answer: {result['answer'][:70]}...")
    print()

# Example 7: Using Evidence and Components
print("\n7. Examining Evidence and Components")
print("-" * 40)

query = "What are the compliance requirements?"
result = reasoner.infer(query, temperature=0.3)

print(f"Query: {query}")
print(f"Mode: {result['reasoning_mode']}\n")

print("Evidence sources:")
for evidence in result['evidence']:
    print(f"  • {evidence}")

if 'components' in result and result['components']:
    print("\nReasoning components:")
    for component_name, component_data in result['components'].items():
        if isinstance(component_data, dict) and 'confidence' in component_data:
            print(f"  {component_name}: confidence={component_data['confidence']:.3f}")

# Example 8: Best Practices
print("\n\n8. Best Practices Summary")
print("-" * 40)
print("""
Temperature Selection Guide:

T = 0.0 (Symbolic)
  ✓ Formal verification needed
  ✓ Rule-based decision required
  ✓ Certifiable answer important
  ✓ Binary yes/no questions
  Example: "Is X required by regulation Y?"

T = 0.2-0.4 (Hybrid)
  ✓ Need both rules and examples
  ✓ Nuanced interpretation required
  ✓ Combining multiple evidence sources
  ✓ Risk assessment scenarios
  Example: "How should X be implemented?"

T = 0.5-0.8 (Analogical)
  ✓ Looking for similar cases
  ✓ Precedent-based reasoning
  ✓ No exact rule exists
  ✓ Exploring options
  Example: "What approaches have others used for X?"

General Guidelines:
  • Start with T=0 for strict compliance checks
  • Use T=0.3 for balanced reasoning
  • Use T=0.7 for exploratory case finding
  • Run temperature sweeps for critical decisions
  • Pre-compute embeddings for repeated queries
""")

print("=" * 60)
print("Complete workflow examples finished!")
print("=" * 60)
