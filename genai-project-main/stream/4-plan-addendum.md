# Plan Addendum — Map Agents to Your QA Framework (Planning Mode)

> Using **Bedrock models**, **multi-pass (Text → Visual)**, **DSPy signatures**, and your provided **config.json**.
> I’m interpreting your “`<.075`” as **`< 0.75`** for the vision-escalation gate. If you literally meant **0.075**, we can lower it later.

---

## A) Locked Settings

* **Vision trigger:** escalate when `text_confidence < 0.75`.
* **Agent → QA area mapping (yours):**

  ```python
  agent_to_qa_mapping = {
      "DataAgent": "Data Quality/Content Control",
      "LogicAgent": "Validation Review and Testing",
      "DocsAgent": "Style and Formatting",
      "PerfAgent": "Validation Review and Testing",
      "RiskAgent": "Validation Processes",
      "GovAgent": "Governance and Compliance"
  }
  ```
* **Dynamic topic & outputs:** read from **config.json** (your file) at runtime.

---

## B) Agent ↔ Category ↔ Criteria (v1 target)

Primary ownership kept simple; a second agent may “assist” if useful.

| Agent          | Category (config.id)           | Criteria (primary)       | Visual when…                                                   |
| -------------- | ------------------------------ | ------------------------ | -------------------------------------------------------------- |
| **DataAgent**  | `data_quality_content_control` | `data_001..004`          | Tables/metrics referenced but low text confidence (<0.75)      |
| **GovAgent**   | `governance_compliance`        | `gov_001..004`           | Signatures/approvals present but not reliably parsed from text |
| **RiskAgent**  | `validation_processes`         | `val_001..004`           | Methodology diagrams/tables drive findings and text is weak    |
| **LogicAgent** | `validation_review_testing`    | `review_001, review_003` | Test procedures described in images/flowcharts                 |
| **PerfAgent**  | `validation_review_testing`    | `review_002, review_004` | Result tables/plots carry the evidence                         |
| **DocsAgent**  | `style_formatting`             | `style_001..002`         | Only if formatting proof needs figure/table glimpses (rare)    |

> One category (Validation Review & Testing) has **two agents** to keep cognitive load low: **LogicAgent** focuses on procedures/independence, **PerfAgent** on results/verification.

---

## C) How **topic focus** works (from `config.json`)

* At runtime, we read a **topic list** (e.g., `["governance_compliance","validation_processes"]`) from config or UI.
* Coordinator runs **only** agents mapped to those categories; others are skipped.
* Suggested addition to your config (minimal, backward-compatible):

```json
"settings": {
  "...": "...",
  "topic_focus": ["governance_compliance", "validation_processes"],
  "vision_trigger_threshold": 0.75
},
"outputs": {
  "formats": ["PDF", "JSON", "Markdown"],
  "include_evidence_thumbnails": true
}
```

(If `topic_focus` is absent, we default to **all required** categories.)

---

## D) Multi-Pass Gate (simple)

1. **Text pass** (cheap Bedrock text model) → per-criterion `text_confidence ∈ [0,1]`.
2. If (criterion **requires images** by nature **OR** `text_confidence < 0.75`) **AND** we have a page/region crop → run **Visual pass** (vision model) on those **few** regions only.
3. **Merge**: prefer visual for signatures/tables; otherwise weighted tie-break.

No OCR text stored; we keep **page/bbox + crop URI** as evidence.

---

## E) Scoring & Aggregation (use your `scoring_rules`)

* For each **criterion** we produce:

  * `status: pass|fail|needs_review`
  * `confidence: 0–1` (LLM probability-like score)
  * `score`: map to your **scale**:

    * `boolean`: pass=100 / fail=0
    * `percentage_100`: use model’s suggested % (bounded by evidence); if only pass/fail is known, map pass=100, fail=0
* Category score = your **weighted\_average** over criteria (`weight` in config).
* Overall score = your **composite\_calculation** (already in config).
* We honor `allow_partial_scoring: false` → if a **required** criterion lacks evidence, mark **needs\_review** (0) and surface in the report.

---

## F) Wiring to Your Workflow Steps

* **Step 1 — VST Selection**: seeds **topic\_focus** and category weights (from template).
* **Step 2 — Input Gathering**: ingest PDFs; build **manifest + crops**; extract **text layer**.
* **Step 3 — Scoring**: run agents only for **topic\_focus** categories; store per-criterion findings.
* **Step 4 — HealthCheck Logic**: merge findings, compute category/overall scores.
* **Step 5 — QA Findings**: generate recommendations tied to **criteria IDs**.
* **Step 6 — Finalization**: output per `outputs.formats` (PDF/JSON/MD) with thumbnails/links.

All step names/labels remain exactly as your `workflow_steps` define.

---

## G) Minimal Finding Object (planning contract)

```json
{
  "run_id": "uuid",
  "category_id": "governance_compliance",
  "criterion_id": "gov_004",
  "agent": "GovAgent",
  "status": "pass|fail|needs_review",
  "confidence": 0.83,
  "score": 100,
  "evidence": [
    {"type":"text","chunk_id":"c-12","quote":"..."},
    {"type":"vision","page":5,"bbox":[120,240,380,140],"crop_uri":"s3://.../page-005/sign-01.png"}
  ],
  "model_versions": {"text":"bedrock:…","vision":"bedrock:…"},
  "hashes": {"doc":"sha256:…","standards":"sha256:…"}
}
```

---

## H) What we’re deferring (explicit “Later”)

* Heavy stats (distribution tests), algorithm/code execution, recomputing backtests or scenarios.
* Deep style linting.
* Multi-tenant RBAC and advanced workflow automation.

---

## I) Next Actions (tight)

1. I’ll hard-code the **agent ↔ category ↔ criteria** map above (v1).
2. Add `settings.topic_focus` and `settings.vision_trigger_threshold=0.75` to your config.
3. Prepare 6–8 PDFs with **tables + signatures** to validate gates.
4. Wire the **Text→Visual** gate and per-criterion scoring exactly to your `scoring_rules`.

If you want, I can also produce a one-pager “**Config Keys We Read**” so DS/QA knows what toggles what.
