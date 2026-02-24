Great question. The clean way to give your workflow “advanced functions” (tools) without bloating core logic is to add a **resource layer** with a tiny interface + registry, and let steps opt-in to tools through dependency injection. Here’s a tight, production-friendly pattern that fits your S3-first, DSPy/RAG setup.

# 1) Define a tiny Resource interface + registry

```python
# resources/base.py
from typing import Protocol, Any, Dict, Callable
from dataclasses import dataclass, field
import time, json, hashlib

class Resource(Protocol):
    name: str
    version: str
    def call(self, **kwargs) -> Any: ...

@dataclass
class ResourceRegistry:
    _r: Dict[str, Resource] = field(default_factory=dict)
    _policy: Dict[str, Any] = field(default_factory=dict)  # allowlist, limits

    def register(self, resource: Resource):
        self._r[f"{resource.name}:{resource.version}"] = resource

    def use(self, name: str, **kwargs):
        # simple resolver: latest version or explicit "name:ver"
        key = name if ":" in name else max(
            (k for k in self._r if k.startswith(name + ":")),
            default=None
        )
        if key is None: raise KeyError(f"Resource not found: {name}")
        self._enforce_policy(name, kwargs)
        result = self._r[key].call(**kwargs)
        self._audit(name=key, kwargs=kwargs, result=result)
        return result

    def _enforce_policy(self, name: str, kwargs: dict):
        # check allowlist, cost/model caps, argument guards, etc.
        pass

    def _audit(self, name: str, kwargs: dict, result: Any):
        # write compact call log (hash inputs/outputs) to S3
        payload = {
            "ts": time.time(),
            "resource": name,
            "args_hash": hashlib.sha256(
                json.dumps(kwargs, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "result_type": type(result).__name__,
        }
        # s3_utils.put_json("s3://.../logs/resource_calls/", payload)  # your helper
```

# 2) Wrap your “advanced functions” as resources

Examples that match your stack.

```python
# resources/rag_search.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RAGSearch:
    name: str = "rag_search"
    version: str = "1.0.0"
    vector_manager=None  # inject your VectorManager

    def call(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.vector_manager.search(query=query, k=k)

# resources/pdf_extract.py
@dataclass
class PDFExtract:
    name: str = "pdf_extract"
    version: str = "1.0.0"
    extractor=None  # your extraction_helper wrapper

    def call(self, s3_uri: str) -> str:
        return self.extractor.extract_text(s3_uri)

# resources/llm_complete.py (Bedrock/Claude/Sonnet etc.)
@dataclass
class LLMComplete:
    name: str = "llm_complete"
    version: str = "1.0.0"
    client=None  # your Bedrock gateway
    model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    def call(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.2):
        return self.client.complete(model_id=self.model_id,
                                    prompt=prompt,
                                    max_tokens=max_tokens,
                                    temperature=temperature)
```

Register once at app start:

```python
registry = ResourceRegistry()
registry.register(RAGSearch(vector_manager=my_vm))
registry.register(PDFExtract(extractor=my_extractor))
registry.register(LLMComplete(client=my_bedrock))
```

# 3) Use tools inside your workflow via dependency injection

Your workflow functions should receive a `use` callable (i.e., `registry.use`) rather than importing helpers directly. Keeps steps light and testable.

```python
# workflow/mvr_scope.py
def ReviewExecution(ctx: dict, use: Callable):
    # 1) Pull context with RAG (lightweight)
    chunks = use("rag_search", query=ctx["question"], k=6)
    # 2) Build prompt with retrieved evidence
    evidence = "\n\n".join(c["text"] for c in chunks)
    prompt = f"""You are a QA reviewer. Use the evidence below.
Evidence:
{evidence}

Task: {ctx['task']}
Return a concise JSON with fields: decision, rationale, cited_chunk_ids.
"""
    # 3) Call LLM
    return use("llm_complete", prompt=prompt, max_tokens=1200)
```

Wire at runtime:

```python
def run_workflow(message: str, ctx: dict):
    return ReviewExecution(ctx | {"question": message}, use=registry.use)
```

# 4) Keep it safe & controllable with a **Capabilities Manifest**

Put an allowlist + limits in JSON/YAML so you can toggle tools per workflow without code edits.

```yaml
# configs/capabilities.yaml
workflows:
  mvr_scope:
    allow:
      - rag_search
      - llm_complete
    limits:
      llm_complete:
        max_tokens: 16000
        models_allowed:
          - anthropic.claude-3-5-sonnet-20240620-v1:0
      rag_search:
        k_max: 10
```

Load this into `registry._policy` and enforce in `_enforce_policy`.

# 5) Versioning & rollout

* **Semantic versions** on resources (`rag_search:1.0.0 → 1.1.0`).
* Allow per-workflow pinning: `use("rag_search:1.0.0")` for deterministic runs.
* Keep a **changelog** in S3; expose a “Revert” dropdown in Streamlit (you already planned rollbacks).

# 6) Caching (keeps it “light”)

* Add an **S3 memo cache** to expensive resources:

  * Key: `resource_name/version/sha256(args)`
  * TTL or “pin to run\_id” mode.
* Cache is especially helpful for `pdf_extract`, `rag_search` (query, k), and long LLM prompts.

# 7) Observability & cost control

* **Audit log per resource call** (already stubbed).
* Track tokens/cost in `llm_complete` and write per-run totals to S3.
* Optional: “dry-run mode” that returns **plan only** (which tools would be invoked).

# 8) DSPy integration (optional but nice)

If you use DSPy signatures, bind a **tool shim** so the module can call `use()` under the hood:

```python
import dspy

class EvidenceAnswer(dspy.Signature):
    """answer question with citations"""
    question: str
    answer: str

class QAWithTools(dspy.Module):
    def __init__(self, use):
        super().__init__()
        self.use = use
        self.gen = dspy.Predict(EvidenceAnswer)

    def forward(self, question):
        chunks = self.use("rag_search", query=question, k=5)
        ctx = "\n".join(c["text"] for c in chunks)
        return self.gen(question=f"{question}\n\nEvidence:\n{ctx}")
```

# 9) Streamlit toggles (simple governance)

Expose a sidebar **Tools** panel:

* Checkboxes for allowed tools
* Model selector (from allowlist)
* Sliders: max tokens, k
* A “Preview Plan” button to print which resources will be called

# 10) Minimal end-to-end example

```python
ctx = {"task": "Score MVR clarity on Sections 1–3 with citations"}
result = run_workflow("How clear is the Methods section?", ctx)
print(result)
```

---

### Why this works

* **Lightweight by default**: Steps only call tools you allow; no heavy pre-processing.
* **RAG-first**: Retrieval happens just-in-time, not globally.
* **Swappable**: You can replace `rag_search` with pgVector later without touching workflow code.
* **Governable**: One manifest gates models, limits, and tool access per workflow.
* **Auditable**: Every tool call is logged with hashes; easy to trace outputs to inputs.

If you want, I can bundle a tiny starter module (`resources/`, `workflow/`, `configs/`) you can drop into your app.
