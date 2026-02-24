Cool, let’s turn that into something you can actually drop into your stack **without** dragging in all of Snorkel.

I’ll give you:

1. A **minimal LF framework** (registry + apply + simple label combiner).
2. A **JSON/CSV schema** for semantic supervision examples.
3. A **concrete compliance example** (risk-doc LFs).
4. How to **pipe the results into DSPy / GEPA**.

---

## 1. Minimal Snorkel-style LF framework (no Snorkel)

### 1.1. Core label constants

```python
# labels.py
ABSTAIN = -1

OK = 0
WEAK = 1
HIGH_RISK = 2
```

You can tailor that enum to whatever target you’re training (e.g., {NO_ISSUE, FLAG, BLOCK}).

---

### 1.2. Labeling function type + simple decorator

Each LF takes a **record** (dict, pydantic model, etc.) and returns one of your labels or `ABSTAIN`.

```python
# lf_core.py
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass

from labels import ABSTAIN

LF = Callable[[Dict[str, Any]], int]

@dataclass
class LabelingFunctionMeta:
    name: str
    func: LF
    weight: float = 1.0        # optional confidence weight
    description: str = ""      # optional docstring-style summary

def labeling_function(name: Optional[str] = None, weight: float = 1.0, description: str = ""):
    """
    Tiny decorator to tag functions as LFs with metadata.
    """
    def wrapper(fn: LF) -> LabelingFunctionMeta:
        lf_name = name or fn.__name__
        return LabelingFunctionMeta(
            name=lf_name,
            func=fn,
            weight=weight,
            description=description or (fn.__doc__ or "").strip()
        )
    return wrapper
```

Usage:

```python
from lf_core import labeling_function
from labels import ABSTAIN, HIGH_RISK

@labeling_function(weight=1.5, description="Flags missing model purpose section.")
def lf_missing_model_purpose(record):
    text = record["text"].lower()
    if "model purpose" not in text and "intended use" not in text:
        return HIGH_RISK
    return ABSTAIN
```

The decorator returns a `LabelingFunctionMeta` object.

---

### 1.3. LF registry + apply

```python
# lf_registry.py
from typing import Dict, Any, List
from dataclasses import dataclass, field

from lf_core import LabelingFunctionMeta
from labels import ABSTAIN

@dataclass
class LFRegistry:
    lfs: List[LabelingFunctionMeta] = field(default_factory=list)

    def register(self, lf_meta: LabelingFunctionMeta):
        self.lfs.append(lf_meta)

    def apply(self, records: List[Dict[str, Any]]):
        """
        Returns:
          votes: List[List[int]] where votes[i][j] is label from LF j on record i
        """
        votes = []
        for record in records:
            row = []
            for lf_meta in self.lfs:
                try:
                    label = lf_meta.func(record)
                except Exception:
                    label = ABSTAIN
                row.append(label)
            votes.append(row)
        return votes
```

Registering:

```python
# lfs_compliance.py
from lf_core import labeling_function
from labels import ABSTAIN, OK, WEAK, HIGH_RISK

@labeling_function(weight=1.2)
def lf_has_sr11_7_ref(record):
    """If SR 11-7 is referenced, we consider this at least WEAK (not OK by default)."""
    text = record["text"].lower()
    if "sr 11-7" in text or "sr 11 7" in text:
        return WEAK
    return ABSTAIN

@labeling_function(weight=1.5)
def lf_mentions_material_limitations(record):
    """High risk if 'material limitation' appears without mitigation."""
    text = record["text"].lower()
    if "material limitation" in text and "mitigat" not in text:
        return HIGH_RISK
    return ABSTAIN

@labeling_function(weight=0.8)
def lf_generic_positive_language(record):
    """OK if the section contains strong assurance language."""
    text = record["text"].lower()
    if "independent validation completed" in text or "no material findings" in text:
        return OK
    return ABSTAIN
```

Wire them into the registry:

```python
# build_registry.py
from lf_registry import LFRegistry
from lfs_compliance import (
    lf_has_sr11_7_ref,
    lf_mentions_material_limitations,
    lf_generic_positive_language,
)

registry = LFRegistry()
for lf in [lf_has_sr11_7_ref, lf_mentions_material_limitations, lf_generic_positive_language]:
    registry.register(lf)
```

---

### 1.4. Simple label combiner (majority + weights)

#### Majority vote (unweighted)

```python
# label_combiner.py
from collections import Counter
from typing import List
from labels import ABSTAIN

def majority_vote_row(votes_row: List[int]) -> int:
    """
    votes_row: list of LF outputs for a single record.
    Returns a single label or ABSTAIN if nobody voted.
    """
    non_abstain = [v for v in votes_row if v != ABSTAIN]
    if not non_abstain:
        return ABSTAIN
    counts = Counter(non_abstain)
    return counts.most_common(1)[0][0]
```

#### Weighted vote + confidence score

```python
from typing import List
from lf_core import LabelingFunctionMeta

def weighted_vote_row(votes_row: List[int], lfs: List[LabelingFunctionMeta]):
    """
    Returns (label, confidence)
    confidence is normalized sum of supporting weights.
    """
    label_scores = {}
    total_weight = 0.0

    for v, lf_meta in zip(votes_row, lfs):
        if v == ABSTAIN:
            continue
        w = lf_meta.weight
        total_weight += w
        label_scores[v] = label_scores.get(v, 0.0) + w

    if total_weight == 0.0:
        return ABSTAIN, 0.0

    # pick label with max score
    best_label = max(label_scores.items(), key=lambda x: x[1])[0]
    confidence = label_scores[best_label] / total_weight
    return best_label, confidence
```

Now, given `records` and `registry`:

```python
# run_lfs.py
from build_registry import registry
from lf_registry import LFRegistry
from label_combiner import majority_vote_row, weighted_vote_row

def label_records(records):
    votes = registry.apply(records)  # [n_examples x n_lfs]

    outputs = []
    for record, row in zip(records, votes):
        label, conf = weighted_vote_row(row, registry.lfs)
        outputs.append({
            "record": record,
            "label": label,
            "confidence": conf,
            "lf_votes": row,
        })
    return outputs
```

---

## 2. JSON/CSV schema for semantic supervision examples

You can persist the results as **semantic supervision examples**.

### 2.1. JSON example (per record)

```json
{
  "id": "section_00123",
  "text": "This model is intended to support the credit approval process for SME loans...",
  "metadata": {
    "doc_id": "mvp_2025_001",
    "section_name": "Model Purpose",
    "page": 5
  },
  "label": 0,
  "label_name": "OK",
  "confidence": 0.92,
  "source_lfs": [
    {
      "name": "lf_has_sr11_7_ref",
      "vote": -1
    },
    {
      "name": "lf_mentions_material_limitations",
      "vote": -1
    },
    {
      "name": "lf_generic_positive_language",
      "vote": 0
    }
  ]
}
```

### 2.2. CSV-friendly minimal version

Columns:

* `id`
* `text`
* `label`
* `confidence`
* `lf_votes_json` (stringified JSON of LF names → votes)
* optional: `doc_id`, `section_name`, `page`, etc.

Example row:

```csv
id,text,label,confidence,lf_votes_json,doc_id,section_name,page
section_00123,"This model is intended to support...",0,0.92,"{""lf_generic_positive_language"":0}",mvp_2025_001,Model Purpose,5
```

---

## 3. Concrete compliance LFs (how this helps you)

Pick a simple target task, e.g.:

> Classify each section into {OK, WEAK, HIGH_RISK} for “governance & documentation quality”.

Example LFs:

```python
@labeling_function(weight=1.4)
def lf_no_validation_plan(record):
    text = record["text"].lower()
    if "validation" not in text and "independent review" not in text:
        return HIGH_RISK
    return ABSTAIN

@labeling_function(weight=1.3)
def lf_has_validation_but_no_timing(record):
    text = record["text"].lower()
    if "validation" in text and "annual" not in text and "frequency" not in text:
        return WEAK
    return ABSTAIN

@labeling_function(weight=1.1)
def lf_has_roles_and_responsibilities(record):
    text = record["text"].lower()
    if "roles and responsibilities" in text or "r&r" in text:
        return OK
    return ABSTAIN
```

Run over 5k–50k sections from your model inventories → you now have a weakly labeled dataset with confidences.

---

## 4. Feeding this into DSPy / GEPA

Assume you’ve saved JSON like the example above.

### 4.1. Build a DSPy Dataset

Rough DSPy-style pseudo-code (you’ll tune to your signatures):

```python
import json
import dspy

from labels import OK, WEAK, HIGH_RISK

def load_semantic_examples(path, min_conf=0.8):
    examples = []
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            if rec["confidence"] < min_conf:
                continue
            label = rec["label"]
            label_name = {OK: "OK", WEAK: "WEAK", HIGH_RISK: "HIGH_RISK"}[label]
            examples.append(dspy.Example(
                section_text=rec["text"],
                risk_label=label_name
            ).with_inputs("section_text"))
    return examples

dataset = load_semantic_examples("semantic_supervision.jsonl")
```

---

### 4.2. A simple DSPy module for risk labeling

```python
class RiskLabelSignature(dspy.Signature):
    """Classify the risk quality of this section."""
    section_text = dspy.InputField()
    risk_label = dspy.OutputField(desc="One of: OK, WEAK, HIGH_RISK")

class RiskTierClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.lm = dspy.ChainOfThought(RiskLabelSignature)

    def forward(self, section_text):
        return self.lm(section_text=section_text)
```

---

### 4.3. Plug into GEPA (reflective optimizer)

(Lightweight sketch, not exact API):

```python
from dspy.gepa import GEPAOptimizer  # adjust import to your version

gepa = GEPAOptimizer(
    metric="accuracy",          # you can define your own
    n_generations=5,
    population_size=8,
)

optimized_classifier = gepa.optimize(
    module=RiskTierClassifier(),
    trainset=dataset[:200],    # or a subset
    evalset=dataset[200:400],  # again, just an example split
)
```

Now your pipeline is:

1. **LF framework** → weak labels (`semantic_supervision.jsonl`).
2. **Filter by high confidence** → strong examples.
3. **Turn into `dspy.Example`** → training data.
4. **GEPA or other DSPy optimizer** → tuned classifier / extractor.
5. Optionally log:

   * Which LFs generated each example,
   * Confidence scores,
   * GEPA generations + metric improvements → into your existing FAISS / S3 logging.

---

If you’d like, next step I can:

* Wrap all of this into a single **`semantic_supervision_helper.py`** with:

  * `register_lfs()`
  * `run_lfs_and_export_jsonl()`
  * `load_semantic_dataset_for_dspy()`

so you can literally drop one file into `concepts/` or `helpers/` and wire it into VectorQA / TidyUMA.
