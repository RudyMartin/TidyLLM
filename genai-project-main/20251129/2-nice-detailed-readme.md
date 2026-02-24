YRSN Context: Adaptive Context Engineering Library
A Python library for advanced context management inspired by hardware architecture innovations and neural continuous thought machines.

Overview
YRSN Context implements sophisticated context engineering strategies with both classical and neural approaches:

Classical Components (Hardware-Inspired)
Drawing inspiration from Nature Electronics research:

3D Integrated Hybrid Circuits - Layered architecture with interface quality management
RRAM-based Matrix Solving - Iterative refinement and bit-slicing techniques
Four main paradigms for context management:

Iterative Refinement: Fast approximate retrieval → residual computation → precise refinement
Bit-Slicing: Multi-granularity context representation (core/detail/background)
Hierarchical Decomposition: BlockAMC-inspired systematic breakdown of large contexts
Layered Architecture: Stack management where foundation quality propagates upward
Neural Components (Optional - Requires PyTorch)
Drawing inspiration from Sakana AI's Continuous Thought Machine research:

CTM Context Retriever: Neural model with adaptive computation time
Neuron-Level Models: Private-weight neurons learning temporal patterns
Neural Synchronization: Correlation-based representations (S^t = Z^t · Z^t^T)
Adaptive Loss: Different samples use different amounts of computation
NoisyNet Exploration: Learnable exploration for discovering context patterns
See Neural Context Documentation for details.

Installation
# Basic installation (classical components only)
pip install yrsn-context

# With optional dependencies
pip install yrsn-context[embeddings]  # sentence-transformers
pip install yrsn-context[search]      # faiss for vector search
pip install yrsn-context[llm]         # OpenAI/Anthropic clients
pip install yrsn-context[neural]      # PyTorch-based neural components
pip install yrsn-context[full]        # everything (classical + neural + all extras)
pip install yrsn-context[all]         # everything except neural
Note: Neural components require PyTorch 2.0+. Classical components work without PyTorch.

Development Installation
git clone https://github.com/RudyMartin/yrsn-electronics.git
cd yrsn-electronics
pip install -e ".[dev]"
Quick Start
1. Iterative Context Refinement
Inspired by the HP-INV algorithm - iteratively refine context with fast coarse retrieval and precise refinement.

from yrsn_context import IterativeContextEngine, CoarseRetriever, FineRetriever, ResidualAnalyzer

# Set up retrievers (with your own implementations)
coarse = CoarseRetriever(index=your_index, embedding_model=your_model)
fine = FineRetriever(index=your_index, embedding_model=your_model, reranker=your_reranker)
residual = ResidualAnalyzer(llm_client=your_llm)

# Create engine
engine = IterativeContextEngine(
    coarse_retriever=coarse,
    fine_retriever=fine,
    residual_analyzer=residual,
    max_iterations=5,
    precision_threshold=0.95
)

# Retrieve with iterative refinement
result = engine.retrieve("What are the latest trends in AI?")

print(f"Found {len(result.current_context)} context blocks")
print(f"Precision: {result.precision_achieved:.2f}")
print(f"Iterations: {result.iteration}")
2. Bit-Sliced Context
Multi-granularity representation where context is decomposed into importance levels.

from yrsn_context import SlicedContext, GranularityDecomposer

# Create sliced context
sliced = SlicedContext()

# Add content at different granularity levels
sliced.add_slice(level=0, content="Core fact: AI uses neural networks", source="doc1")
sliced.add_slice(level=1, content="Example: CNN for image classification", source="doc1")
sliced.add_slice(level=2, content="Historical context: perceptrons from 1950s", source="doc2")

# Combine with appropriate precision
formatted = sliced.combine(target_precision=12)
print(formatted)
3. Hierarchical Decomposition
BlockAMC-inspired decomposition for large documents.

from yrsn_context import HierarchicalContextDecomposer, BlockAMCProcessor

# Decompose large document
decomposer = HierarchicalContextDecomposer(base_block_size=1000, max_levels=3)
blocks = decomposer.decompose(content=large_document, query=user_query)

# Process blocks systematically
processor = BlockAMCProcessor(
    block_processor=your_processor,
    propagator=your_propagator
)
results = processor.process_hierarchical(blocks, query=user_query)
4. Layered Context Stack
Stack management with interface quality tracking.

from yrsn_context import LayeredContextStack

# Create stack
stack = LayeredContextStack(max_layers=6)

# Add layers (foundation first!)
stack.add_layer(
    level=0,
    name="core_knowledge",
    content=["Python is a programming language", "AI uses machine learning"],
    quality_score=0.98  # High quality for foundation
)

stack.add_layer(
    level=1,
    name="domain_context",
    content=["RAG systems retrieve relevant documents"],
    quality_score=0.90
)

# Get effective context (accounts for interface quality)
context = stack.get_effective_context(target_layer=1, include_lower=True)
print(f"Stack quality: {stack.compute_stack_quality():.2f}")
5. Neural Context Retrieval (Optional)
Requires: pip install yrsn-context[neural]

Continuous Thought Machine-inspired neural model with adaptive computation.

import torch
from yrsn_context.neural import CTMContextRetriever

# Create neural model
model = CTMContextRetriever(
    context_dim=128,              # Embedding dimension
    n_internal_ticks=5,           # "Thought" steps
    history_length=10,            # Temporal history length
    n_classes=10,                 # Output classes
    use_noisy_retrieval=True,     # Enable exploration
    use_iterative_refinement=True # Enable HP-INV refinement
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for context, targets in train_loader:
    outputs = model(context, targets)
    loss = outputs['loss']
    loss.backward()
    optimizer.step()

    # Adaptive computation: different samples use different ticks
    print(f"Avg ticks used: {outputs['loss_info']['mean_t1']:.1f}")

# Inference
model.eval()
with torch.no_grad():
    outputs = model(context_batch)
    predictions = outputs['final_logits'].argmax(dim=-1)
See Neural Context Tutorial for complete training guide.

Architecture
yrsn-context/
├── core/           # Base classes and types
├── iterative/      # Iterative refinement (HP-INV inspired)
├── bitslicing/     # Multi-granularity representation
├── hierarchical/   # BlockAMC decomposition
├── layered/        # Layered stack with interface quality
├── neural/         # Neural components (optional, requires PyTorch)
│   ├── ctm/        # Continuous Thought Machine components
│   ├── refinement/ # Neural HP-INV and NoisyNet
│   └── training/   # Training utilities (SnAp, Budget)
└── utils/          # Utility functions
Key Concepts
Iterative Refinement
Analogous to solving Ax = b iteratively:

Initial approximation (coarse retrieval)
Compute residual (what's missing)
Refine (precise retrieval on gaps)
Update and repeat until convergence
Bit-Slicing
Like decomposing a matrix: A = 2⁰A₀ + 2³A₁ + 2⁶A₂ + ...

Context slices:

Level 0 (2⁰): Core facts, direct answers
Level 1 (2⁻³): Supporting details
Level 2 (2⁻⁶): Background context
Level 3 (2⁻⁹): Tangential information
Hierarchical Decomposition
BlockAMC algorithm for large matrices → context blocks:

Partition into independent blocks (diagonal)
Identify dependencies (off-diagonal)
Process systematically
Combine results
Layered Architecture
3D circuit stacking insights → context layers:

Foundation quality propagates upward
Interface quality affects upper layers
Complementary approaches (fast + precise)
Paper Inspirations
This library is inspired by research from Nature Electronics:

Three-Dimensional Integrated Hybrid Complementary Circuits

Interface roughness hierarchy
Complementary balancing
Low thermal budget processing
Precise Analogue Matrix Solving with RRAM

Iterative refinement (LP-INV + HP-MVM)
Bit-slicing decomposition
BlockAMC for scalability
Examples
See the examples/ directory for complete working examples:

examples/basic_iterative.py - Iterative refinement
examples/bitsliced_retrieval.py - Multi-granularity context
examples/hierarchical_decomposition.py - Large document processing
examples/neural_ctm_demo.py - Neural context retrieval demo (requires PyTorch)
Documentation
Classical Components:

Architecture Overview
Tutorial 01: Getting Started
Tutorial 02: Iterative Retrieval
Tutorial 03: Production RAG
Hardware Analogy Guide
Neural Components:

Neural Context Documentation - Complete reference
Tutorial 04: Neural Retrieval - Training guide
Research Foundations - CTM, HP-INV, NoisyNet
Guides:

When to Use What
Comparison Guide
FAQ
Requirements
Core (Classical Components):

Python >= 3.8
NumPy >= 1.20.0
Optional:

sentence-transformers >= 2.2.0 (for embeddings)
faiss-cpu >= 1.7.4 (for vector search)
openai >= 1.0.0 or anthropic >= 0.18.0 (for LLM integration)
torch >= 2.0.0 (for neural components)
Development
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
MIT License - see LICENSE file for details.

Citation
If you use this library in your research, please cite:

@software{yrsn_context,
  title = {YRSN Context: Adaptive Context Engineering Library},
  author = {YRSN Electronics},
  year = {2025},
  url = {https://github.com/RudyMartin/yrsn-electronics}
}
Acknowledgments
This library draws inspiration from multiple research areas:

Hardware Architecture (Classical Components):

Nature Electronics research on 3D integrated circuits and RRAM-based computing
Demonstrates how physical computing principles inform software design patterns
Neural Continuous Thought (Neural Components):

Sakana AI's Continuous Thought Machine (CTM) research
HP-INV analog computing from Nature Electronics
DeepMind's NoisyNet exploration
Efficient gradient tracking (SnAp, RTRL variants)
The hybrid approach demonstrates that hardware-software co-design principles can be successfully extended to neural architectures.

About
Inspirations for Context Engineering

Resources
 Readme
License
 MIT license
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Contributors
2
@claude
claude Claude
@RudyMartin
RudyMartin Rudy Martin
Languages
HTML
56.3%
 
Python
43.7%
Suggested workflows
Based on your tech stack
SLSA Generic generator logo
SLSA Generic generator
Generate SLSA3 provenance for your existing release workflows
Jekyll using Docker image logo
Jekyll using Docker image
Package a Jekyll site using the jekyll/builder Docker image.
Python package logo
Python package
Create and test a Python package on multiple Python versions.
More workflows
Footer
