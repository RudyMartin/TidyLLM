### **🔍 Breakdown: Where Each DSPy Component is Used in the Script**
Here’s **exactly where each DSPy component** is being used:

---

### **✅ 1. DSPy Signatures (`dspy.Signature`)**
✅ **Used for structured query & retrieval**  
📌 **Located in lines:** **74-86**  
These define the **input/output structure** for our DSPy models.
```python
# DSPy Signatures
class RationaleSignature(dspy.Signature):  # 📌 Line 74
    """Generate rationale for a query based on retrieved documents."""
    query: str = dspy.InputField()
    sources: list = dspy.InputField()
    rationale: str = dspy.OutputField()  # 📌 Defines output

class ChainOfThoughtQASignature(dspy.Signature):  # 📌 Line 81
    """Answer questions using Chain of Thought."""
    query: str = dspy.InputField()
    context: str = dspy.InputField()
    reasoning_steps: str = dspy.OutputField()
    final_answer: str = dspy.OutputField()  # 📌 Defines output
```

---

### **✅ 2. DSPy Modules (`dspy.Module`)**
✅ **Defines `RationaleModel` & `ChainOfThoughtQAModel`**  
📌 **Located in lines:** **89-103**  
These **implement logic for DSPy models** that process retrieved documents and generate answers.
```python
# DSPy Modules
class RationaleModel(dspy.Module):  # 📌 Line 89
    def __init__(self):
        super().__init__()

    @dspy.predict(RationaleSignature)  # 📌 Uses the Signature for structure
    def __call__(self, query, sources):
        return RationaleSignature(query=query, sources=sources)

class ChainOfThoughtQAModel(dspy.Module):  # 📌 Line 97
    def __init__(self):
        super().__init__()

    @dspy.predict(ChainOfThoughtQASignature)  # 📌 Uses the Signature for structure
    def __call__(self, query, context):
        return ChainOfThoughtQASignature(query=query, context=context)
```

---

### **✅ 3. DSPy Compilers (`dspy.Compile`)**
✅ **NEW - Uses `BootstrapFewShot` to improve reasoning**  
📌 **Located in line:** **106**  
This **compiles our Chain of Thought model** using few-shot learning.
```python
# **🔹 NEW: DSPy Compiler Using `BootstrapFewShot`**
teleprompter = BootstrapFewShot(metric=None)  # 📌 Line 106
compiled_qa = teleprompter.compile(ChainOfThoughtQAModel())  # 📌 Compiles CoT Model
```
This **optimizes how the model generates answers** based on training examples.

---

### **✅ 4. DSPy Prompters (`dspy.teleprompt.Teleprompter`)**
✅ **NEW - Uses DSPy’s prompting capabilities**  
📌 **Located in lines:** **150-156**  
This **actually applies DSPy’s reasoning to the search results.**
```python
    # **NEW: Use DSPy Prompter for Enhanced Rationale**
    retrieved_texts = [f"Document {i}: [Relevant Text]" for i in search_results["indices"]]
    rationale_model = RationaleModel()  # 📌 Uses DSPy for explaining relevance
    rationale_output = rationale_model(query=query_text, sources=retrieved_texts)

    qa_response = compiled_qa(query=query_text, context="\n".join(retrieved_texts))  # 📌 Uses DSPy’s trained CoT model
```
- **`rationale_model`** → Explains why retrieved documents are relevant.  
- **`compiled_qa`** → Uses **few-shot learning** to refine the final AI-generated answer.  

---

### **🚀 Summary of DSPy Usage in Code**
| **DSPy Component** | **Where It's Used in Code** |
|-------------------|---------------------------|
| **Signatures (`dspy.Signature`)** | **Lines 74-86** (Defines structured input/output) |
| **Modules (`dspy.Module`)** | **Lines 89-103** (Implements AI models for retrieval & reasoning) |
| **Compilers (`dspy.Compile`)** | **Line 106** (Optimizes CoT model using `BootstrapFewShot`) |
| **Prompters (`dspy.teleprompt.Teleprompter`)** | **Lines 150-156** (Applies DSPy reasoning to search results) |

