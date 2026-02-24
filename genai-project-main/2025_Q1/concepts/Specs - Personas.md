# 📊 VectorQA – Persona × Functionality Matrix (Release Spec)

This document outlines the VectorQA tool's key features, release tiers, and how each feature maps to user personas across five core functional areas.

---

## 🎭 Personas

- **Model Developer**: Creates model documentation and iterates through pre-review
- **Validator / Auditor**: Reviews documents for completeness and policy adherence
- **Risk Officer / PM**: Oversees program success and ensures project momentum
- **Governance Executive**: Needs summary signoff visibility, emerging risk trends
- **LLM Dev / QA**: Maintains performance and explainability of AI/DSPy agents

---

## 🧩 Functional Areas (App Sections)

1. 📄 **Model Pre-Review**
2. 📊 **Inventory & Dashboards**
3. 🧠 **LLM Self-Evaluation**
4. 🔍 **Semantic Search & Compare**
5. 🛠 **Admin / Diagnostics**

---

## 🔢 Release Tiers

| Tier | Description                         | Audience                     | Example Features                             |
|------|-------------------------------------|-------------------------------|----------------------------------------------|
| R1   | **Core MVP**                        | Developers, Auditors, QA     | Chunk, Comment, Search, Submit               |
| R2   | **Admin + Reporting Maturity**     | PMs, Risk Owners              | Dashboards, Cleanup, Index Monitoring        |
| R3   | **LLM Audit + Governance Loop**    | Execs, DSPy/QA teams          | Agent Metrics, Traceback, Self-Audit         |

---

## 🧠 Persona × Function Matrix

| Persona / Role         | 📄 Model Pre-Review       | 📊 Inventory & Dash         | 🧠 LLM Self-Eval           | 🔍 Semantic Search        | 🛠 Admin / Diagnostics     |
|------------------------|----------------------------|------------------------------|-----------------------------|----------------------------|----------------------------|
| **Model Developer**    | ✅ R1: Submit Draft MDP    | ⚪ R2: Review Status         | ⚪ R3: Trace Pipeline        | ✅ R1: Guided Search Tool   | ⚪ R2: View Stored Chunks   |
|                        | ✅ R1: Section Feedback    |                              |                             | ✅ R1: Compare Embeddings   | ⚪ R2: Clean Up My Files    |
| **Validator / Auditor**| ✅ R1: Review Sections     | ✅ R1: Topic Coverage Map    | ⚪ R3: LLM Agent Breakdown   | ✅ R1: Search Logs by Tag   | ⚪ R2: Embedding Index View |
|                        | ✅ R1: Comment Mode        | ✅ R1: Review Completion     |                             |                            |                            |
| **Risk Officer / PM**  | ⚪ R2: Review Status       | ✅ R1: Completion Dashboard  | ⚪ R3: Agent Accuracy Trends | ⚪ R2: Topic Issue Heatmap  | ⚪ R2: Config Snapshot      |
|                        |                            | ✅ R1: Emerging Risks        |                             |                            |                            |
| **Governance Executive**| ⚪ R3: Signoff Summary     | ✅ R1: Emerging Risks        | ⚪ R3: Trust Scoring         | ⚪ R3: Trend Highlights      | ❌ Hidden                   |
|                        |                            | ✅ R1: Topic Map             |                             |                            |                            |
| **LLM Dev / QA**       | ⚪ R3: Upload QA Truths    | ⚪ R2: Eval Summary          | ✅ R1: Agent Metrics         | ✅ R1: Embedding Compare     | ✅ R1: Logs & Config Tools  |
|                        |                            | ⚪ R2: Eval Coverage Stats   | ✅ R1: DSPy Trace Explorer   | ✅ R1: Audit Agent Output    | ✅ R1: Index Health Tools   |

---

## 📁 Functional Area Examples

### 📄 Model Pre-Review
- Upload + Chunk Model Dev Plans
- Section-by-section reviewer mode
- Comments + status per section

### 📊 Inventory & Dashboards
- Heatmaps by topic or team
- Completion % by document or org
- Risk flags and clustering

### 🧠 LLM Self-Evaluation
- Agent-by-agent accuracy breakdown
- DSPy optimizer comparison
- Self-submission as a model plan

### 🔍 Semantic Search
- Vector search by section text
- Compare different model outputs
- Guided Q&A with LLM embedding

### 🛠 Admin / Diagnostics
- FAISS + pgVector index health
- Chunk viewers + cleanup tools
- Config toggles and model list

---

## 🔚 Next Steps

This matrix can drive:
- UI Filtering by Persona (e.g., select your role on the Overview page)
- Release Roadmapping (build R1, plan R2+R3)
- Stakeholder Presentations (focused demos)

---

💡 *Updated: April 2025 — Contact: Rudy Martin, Next Shift Consulting*

