# YRSN vs DSPy (with GEPA & MIvPro2): Fundamental Differences

## Executive Summary

**YRSN and DSPy solve different problems at different layers:**

| Framework | Problem | When | Layer |
|-----------|---------|------|-------|
| **DSPy** | Prompt optimization | Compile-time | Prompt engineering |
| **GEPA** | Reflective reasoning | Runtime | Reasoning quality |
| **MIvPro2** | Multi-prompt optimization | Compile-time | Prompt selection |
| **YRSN** | Context quality decomposition | Runtime | Context engineering |

**Key Insight:** YRSN is **NOT** like DSPy. They are **complementary**:
- **DSPy** optimizes **what prompts to use** (compile-time)
- **YRSN** optimizes **what context to use** (runtime)

---

## 1. What Each Framework Does

### DSPy (Programmatic Prompt Optimization)

**Purpose:** Optimize prompts at compile-time using programmatic signatures

**How it works:**
```
Define Signature → Compile → Optimize Prompts → Deploy
```

**Key Features:**
- Compile-time optimization
- Signature-based prompt engineering
- Automatic few-shot example selection
- BootstrapFewShot, MIPROv2, COPRO optimizers

**Example:**
```python
import dspy

class Summarize(dspy.Signature):
    """Summarize the given text."""
    text = dspy.InputField()
    summary = dspy.OutputField()

# Compile-time: DSPy optimizes the prompt
summarize = dspy.ChainOfThought(Summarize)
```

### GEPA (Reflective Reasoning)

**Purpose:** Reflective reasoning for quality improvement

**Note:** GEPA appears to be mentioned in passing in the codebase. It seems to be a reasoning quality framework, but detailed implementation is not found in the current codebase.

### MIvPro2 (Multi-Prompt Optimization)

**Purpose:** Multi-prompt optimization at compile-time (DSPy optimizer)

**How it works:**
- Evaluates multiple prompt variations
- Selects best-performing prompts
- Optimizes prompt selection strategy

**Example (from function_list.md):**
```python
# DSPy MIPROv2 for report improvement
generate_improved_report = dspy.optimize.MIPROv2(
    signature=ImproveReport,
    metric=quality_score
)
```

### YRSN (Context Quality Decomposition)

**Purpose:** Decompose context into R/S/N components and manage quality at runtime

**How it works:**
```
Context → Decompose (Y = R + S + N) → Quality Score (α) → Temperature (τ) → Routing
```

**Key Features:**
- Runtime context analysis
- Three-way decomposition (R/S/N)
- Quality-driven temperature coupling (τ = 1/α)
- Collapse detection and remediation
- Adaptive memory systems (SDM, Hopfield, EWC, Replay)

**Example:**
```python
from yrsn.core import robust_pca, detect_collapse

# Runtime: YRSN analyzes context quality
result = robust_pca(context_embeddings)
R, S, N = result.to_yrsn()
quality = R / (R + S + N)  # α
temperature = 1 / quality   # τ
```

---

## 2. Fundamental Differences

### Layer of Operation

| Framework | Layer | When | What It Controls |
|-----------|-------|------|------------------|
| **DSPy** | Prompt layer | Compile-time | Prompt text, examples |
| **GEPA** | Reasoning layer | Runtime | Reasoning quality |
| **MIvPro2** | Prompt selection | Compile-time | Which prompts to use |
| **YRSN** | Context layer | Runtime | What context to use |

### Optimization Timing

**DSPy/MIvPro2:**
- **Compile-time:** Optimize prompts before deployment
- **Static:** Same prompts for all queries
- **One-time:** Optimize once, use many times

**YRSN:**
- **Runtime:** Analyze context for each query
- **Dynamic:** Adapts to each context's quality
- **Continuous:** Learns from feedback continuously

### What They Measure

| Framework | Measures | Output |
|-----------|----------|--------|
| **DSPy** | Prompt effectiveness | Optimized prompt text |
| **MIvPro2** | Multi-prompt performance | Best prompt selection |
| **YRSN** | Context quality (R/S/N) | Quality score (α), temperature (τ) |

---

## 3. How They Work Together

### Complementary Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    COMPILE-TIME                          │
│  DSPy + MIvPro2: Optimize prompts                        │
│  └─> Best prompt signatures                             │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    RUNTIME                              │
│  YRSN: Analyze context quality                          │
│  └─> R/S/N decomposition → α → τ                       │
│                                                          │
│  GEPA: Reflective reasoning                             │
│  └─> Quality improvement                               │
└─────────────────────────────────────────────────────────┘
```

### Integration Example

```python
# COMPILE-TIME: DSPy optimizes prompts
import dspy
from dspy.optimize import MIPROv2

class AnalyzeDocument(dspy.Signature):
    """Analyze document quality."""
    document = dspy.InputField()
    analysis = dspy.OutputField()

# Optimize prompts at compile-time
analyzer = MIPROv2(AnalyzeDocument, metric=quality_metric)
optimized_prompts = analyzer.compile()

# RUNTIME: YRSN manages context quality
from yrsn.core import robust_pca
from yrsn.memory import BaseLearner

# Analyze context before using optimized prompts
context_quality = robust_pca(document_embeddings)
R, S, N = context_quality.to_yrsn()
alpha = R / (R + S + N)

# Route based on quality
if alpha > 0.7:
    # High quality: Use optimized prompt
    result = analyzer(document=document)
else:
    # Low quality: Use conservative prompt or human review
    result = conservative_analyzer(document=document)
```

---

## 4. Key Quote from Documentation

From `yrsn-context/docs/How-BaseLearner-Works.md`:

> **"DSPy is NOT learning"** - DSPy optimizes prompts at compile-time. BaseLearner learns at runtime from feedback.

**Translation:**
- **DSPy:** Static optimization (compile-time)
- **YRSN BaseLearner:** Dynamic learning (runtime)

---

## 5. Comparison Table

| Aspect | DSPy | MIvPro2 | GEPA | YRSN |
|--------|------|---------|------|------|
| **Optimization Timing** | Compile-time | Compile-time | Runtime | Runtime |
| **What It Optimizes** | Prompt text | Prompt selection | Reasoning quality | Context quality |
| **Learning** | Static (one-time) | Static (one-time) | Dynamic | Dynamic |
| **Decomposition** | ❌ | ❌ | ❌ | ✅ R/S/N |
| **Quality Metrics** | Prompt effectiveness | Multi-prompt performance | Reasoning quality | Context quality (α) |
| **Temperature Control** | ❌ | ❌ | ❌ | ✅ τ = 1/α |
| **Collapse Detection** | ❌ | ❌ | ❌ | ✅ 10 types |
| **Memory Systems** | ❌ | ❌ | ❌ | ✅ SDM, Hopfield, EWC, Replay |
| **Adaptive Routing** | ❌ | ❌ | ❌ | ✅ Quality-based |

---

## 6. When to Use Each

### Use DSPy/MIvPro2 When:
- ✅ You need to optimize prompts before deployment
- ✅ You have a fixed set of tasks
- ✅ You want automatic few-shot example selection
- ✅ You need compile-time prompt optimization

### Use YRSN When:
- ✅ You need runtime context quality analysis
- ✅ You have dynamic, varying contexts
- ✅ You need to detect context collapse
- ✅ You want quality-driven routing
- ✅ You need adaptive memory systems

### Use Both Together When:
- ✅ You want optimized prompts (DSPy) + quality-aware routing (YRSN)
- ✅ You need compile-time optimization + runtime adaptation
- ✅ You have complex systems requiring both prompt and context management

---

## 7. Real-World Analogy

**DSPy/MIvPro2:**
- Like a **recipe book** - optimized instructions (prompts) written once
- You follow the recipe (prompt) the same way every time

**YRSN:**
- Like a **quality inspector** - checks ingredients (context) before cooking
- Adapts the recipe based on ingredient quality

**Together:**
- Use optimized recipes (DSPy) + quality-check ingredients (YRSN) = Best results

---

## 8. Code Examples

### DSPy Only (Compile-Time)
```python
import dspy

class Summarize(dspy.Signature):
    text = dspy.InputField()
    summary = dspy.OutputField()

# Optimize once at compile-time
summarizer = dspy.ChainOfThought(Summarize)
summarizer = dspy.optimize.BootstrapFewShot(summarizer)

# Use same optimized prompt for all queries
result = summarizer(text="Long document...")
```

### YRSN Only (Runtime)
```python
from yrsn.core import robust_pca

# Analyze context quality at runtime
context = get_context_for_query(query)
result = robust_pca(context)
R, S, N = result.to_yrsn()
alpha = R / (R + S + N)

# Route based on quality
if alpha > 0.7:
    use_strong_model()
else:
    use_conservative_model()
```

### Combined (Best of Both)
```python
# COMPILE-TIME: Optimize prompts
import dspy
summarizer = dspy.optimize.MIPROv2(Summarize)

# RUNTIME: Check context quality
from yrsn.core import robust_pca
context_quality = robust_pca(context)
alpha = context_quality.quality_score()

# Use optimized prompt + quality-aware routing
if alpha > 0.7:
    result = summarizer(text=context)  # High quality: use optimized prompt
else:
    result = conservative_summarizer(text=context)  # Low quality: use safe prompt
```

---

## 9. Key Takeaways

1. **YRSN ≠ DSPy:**
   - DSPy: Prompt optimization (compile-time)
   - YRSN: Context quality management (runtime)

2. **They are Complementary:**
   - DSPy optimizes **what prompts to use**
   - YRSN optimizes **what context to use**

3. **Different Layers:**
   - DSPy: Prompt layer
   - YRSN: Context layer

4. **Different Timing:**
   - DSPy: Compile-time (static)
   - YRSN: Runtime (dynamic)

5. **Use Both for Best Results:**
   - Optimized prompts (DSPy) + Quality-aware routing (YRSN)

---

## 10. References in Codebase

- **DSPy Integration:** `yrsn-research/docs/talks/multiagent/00-MultiAgent-Talk-Index.md`
- **BaseLearner (YRSN):** `yrsn-context/docs/How-BaseLearner-Works.md`
- **MIvPro2 Usage:** `genai-project/function_list.md` (line 96)
- **GEPA Mention:** `yrsn-context/2025-12-11-CollapseConversation.md` (line 5242)

---

*Generated: 2025-09-10*
*Based on codebase analysis and documentation review*
