Here are the two scripts you asked for. They’re **standard-library only** (no extra installs), self-documenting, and designed to run from your repo root. They implement a simple **watch → ingest → extract → embed → index → analysis → report** pipeline that kicks off when a new `REVxxxxx` package appears in `qa_drop/`.

---

## `qa_controls_setup.py`

**(creates folders, a config file, and a README with tour instructions)**

````python
# PLACE IN: qa_controls_setup.py
# PURPOSE: One-time (or repeatable) setup for the QA pipeline folders + config.
# USAGE:
#   python qa_controls_setup.py
#
# WHAT THIS DOES:
#   - Creates the pipeline folders:
#       qa_drop (watch) -> qa_ingest -> qa_extract -> qa_embed -> qa_index -> qa_analysis -> qa_reports
#       + qa_logs
#   - Writes qa_config.json with a revision pattern and polling interval.
#   - Writes a README in qa_drop/ explaining expected package contents.
#
# EXPECTED DROP PACKAGE STRUCTURE (example):
#   qa_drop/
#     REV00001/
#       mvr_feedback_metrics.yaml        # core checklist (required)
#       custom_checklist.md              # optional custom questions (markdown)
#       custom_prompts.md                # optional custom prompts (markdown)
#       attachments/...                  # optional evidence docs (ignored by pipeline)

import json
import os
from pathlib import Path
import textwrap
import datetime

ROOT = Path(__file__).resolve().parent

FOLDERS = [
    "qa_drop",
    "qa_ingest",
    "qa_extract",
    "qa_embed",
    "qa_index",
    "qa_analysis",
    "qa_reports",
    "qa_logs",
]

CONFIG_PATH = ROOT / "qa_config.json"

DEFAULT_CONFIG = {
    "revision_regex": r"^REV\d{5,}$",  # matches REV00001, REV12345, etc.
    "poll_seconds": 3,
    "max_batch": 5,                    # process up to N new revisions per poll
    "analysis": {
        "use_core_yaml": True,
        "core_yaml_filenames": ["mvr_feedback_metrics.yaml"],
        "custom_checklist_filenames": ["custom_checklist.md"],
        "custom_prompts_filenames": ["custom_prompts.md"],
        "report_name_template": "{rev}_report.md",
    }
}

def main():
    # Create folders
    for d in FOLDERS:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)

    # Write config if missing
    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"[setup] Wrote {CONFIG_PATH.name}")
    else:
        print(f"[setup] {CONFIG_PATH.name} already exists (leaving as-is)")

    # Drop-zone README (tour instructions)
    dz_readme = ROOT / "qa_drop" / "README.md"
    if not dz_readme.exists():
        dz_readme.write_text(textwrap.dedent(f"""
        # QA Drop Zone – Live Folder

        Place a new **revision package** here to trigger the pipeline:
        - The folder name **must match** the regex `{DEFAULT_CONFIG['revision_regex']}`.
        - Example: `REV00001`, `REV12345`.

        ## Required & Optional Files

        ```
        REV00001/
          mvr_feedback_metrics.yaml      # REQUIRED (core checklist YAML)
          custom_checklist.md            # OPTIONAL (markdown with custom questions)
          custom_prompts.md              # OPTIONAL (markdown with custom prompts)
          attachments/                   # OPTIONAL (any additional docs; ignored by pipeline)
        ```

        ## What happens next (tour)
        1. `qa_controls.py` detects the new `REVxxxxx` in `qa_drop/`.
        2. It moves/copies the package into:
           - `qa_ingest/REVxxxxx/`
           - then parses to `qa_extract/REVxxxxx/` (questions/prompts → JSON)
           - writes stubs to `qa_embed/REVxxxxx/` and `qa_index/REVxxxxx/`
           - runs analysis per question/prompt into `qa_analysis/REVxxxxx/`
           - emits a consolidated report in `qa_reports/` named like `{DEFAULT_CONFIG['analysis']['report_name_template']}`.
        3. Logs are written under `qa_logs/`.

        ## Notes
        - Multiple revisions can be dropped; the watcher processes them in batches.
        - In TEST mode, reviewers can change the markdown/yaml and re-drop a new revision (e.g., `REV00002`).
        """).strip()+"\n", encoding="utf-8")
        print(f"[setup] Wrote {dz_readme.relative_to(ROOT)}")
    else:
        print(f"[setup] {dz_readme.relative_to(ROOT)} already exists (leaving as-is)")

    # Basic log stub
    processed = ROOT / "qa_logs" / "processed_revisions.json"
    if not processed.exists():
        processed.write_text(json.dumps({"processed": [], "generated_at": datetime.datetime.utcnow().isoformat() + "Z"}, indent=2), encoding="utf-8")
        print(f"[setup] Wrote {processed.relative_to(ROOT)}")
    else:
        print(f"[setup] {processed.relative_to(ROOT)} already exists (leaving as-is)")

    print("\n[setup] Complete.\nFolders ready. Drop a new REV folder into qa_drop/ to begin.")

if __name__ == "__main__":
    main()
````

---

## `qa_controls.py`

**(watches `qa_drop/` for `REVxxxxx`, runs the pipeline, generates a report)**

```python
# PLACE IN: qa_controls.py
# PURPOSE: Watch qa_drop/ for new REVxxxxx packages and run the pipeline:
#          qa_drop -> qa_ingest -> qa_extract -> qa_embed -> qa_index -> qa_analysis -> qa_reports
#
# USAGE:
#   python qa_controls.py              # watch mode (polling loop)
#   python qa_controls.py --once       # single pass (useful for cron/CI)
#
# EXPECTED INPUT (in qa_drop/REVxxxxx/):
#   mvr_feedback_metrics.yaml          # required (core checklist)
#   custom_checklist.md                # optional (markdown)
#   custom_prompts.md                  # optional (markdown)
#
# OUTPUTS:
#   qa_ingest/REVxxxxx/*               # raw copy
#   qa_extract/REVxxxxx/questions.json # core + custom questions (array)
#   qa_extract/REVxxxxx/prompts.json   # custom prompts (array)
#   qa_embed/REVxxxxx/embeddings.json  # stub
#   qa_index/REVxxxxx/index.json       # stub
#   qa_analysis/REVxxxxx/analysis.json # per-item chat() outputs
#   qa_reports/REVxxxxx_report.md      # consolidated markdown report
#
# NOTES:
#   - This script uses only Python stdlib. The chat() function is a stub you can route to your LLM later.
#   - Simple, defensive parsing for YAML/Markdown. Non-fatal if a file is missing; it logs and continues.

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- Paths & Config ----------

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "qa_config.json"
LOGS_DIR = ROOT / "qa_logs"

DIRS = {
    "drop": ROOT / "qa_drop",
    "ingest": ROOT / "qa_ingest",
    "extract": ROOT / "qa_extract",
    "embed": ROOT / "qa_embed",
    "index": ROOT / "qa_index",
    "analysis": ROOT / "qa_analysis",
    "reports": ROOT / "qa_reports",
}

DEFAULT_CFG = {
    "revision_regex": r"^REV\d{5,}$",
    "poll_seconds": 3,
    "max_batch": 5,
    "analysis": {
        "use_core_yaml": True,
        "core_yaml_filenames": ["mvr_feedback_metrics.yaml"],
        "custom_checklist_filenames": ["custom_checklist.md"],
        "custom_prompts_filenames": ["custom_prompts.md"],
        "report_name_template": "{rev}_report.md",
    }
}

# ---------- Safe imports for YAML (optional) ----------
try:
    import yaml  # PyYAML (optional; if missing, we fallback to empty core)
except Exception:
    yaml = None

# ---------- Helpers ----------

def load_cfg() -> Dict[str, Any]:
    if CFG_PATH.exists():
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CFG

def ensure_dirs():
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[qa] {msg}")

def load_processed() -> Dict[str, Any]:
    path = LOGS_DIR / "processed_revisions.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"processed": []}

def save_processed(data: Dict[str, Any]):
    path = LOGS_DIR / "processed_revisions.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def find_new_revisions(cfg: Dict[str, Any]) -> List[Path]:
    pattern = re.compile(cfg["revision_regex"])
    found = []
    for entry in DIRS["drop"].iterdir():
        if entry.is_dir() and pattern.match(entry.name):
            found.append(entry)
    found.sort()
    return found

def copytree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

# ---------- Parsing inputs ----------

def parse_core_yaml(paths: List[Path]) -> List[Dict[str, str]]:
    """
    Returns list of {section, metric_id, metric_question}
    """
    items: List[Dict[str, str]] = []
    if not yaml:
        log("PyYAML not available; skipping core YAML parsing.")
        return items

    for yp in paths:
        if not yp.exists():
            continue
        try:
            data = yaml.safe_load(yp.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"Failed to parse YAML {yp.name}: {e}")
            continue

        # Normalize a variety of shapes into a flat list
        def add(item: Dict[str, Any], section_hint: str = None, id_hint: str = None):
            mid = str(item.get("metric_id") or item.get("id") or id_hint or f"metric_{len(items)+1:03d}")
            mq = str(item.get("metric_question") or item.get("question") or "(no question text)")
            section = str(item.get("section") or section_hint or "Ungrouped")
            items.append({"section": section, "metric_id": mid, "metric_question": mq})

        if isinstance(data, dict) and "metrics" in data:
            data = data["metrics"]

        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            for k, v in data.items():
                add(v, section_hint=v.get("section"), id_hint=k)
        elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            for sec, arr in data.items():
                for it in arr:
                    add(it, section_hint=str(sec))
        elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
            for it in data:
                add(it, section_hint=it.get("section"))
        else:
            log(f"Unrecognized YAML structure in {yp.name}; skipping.")

    return items

def parse_markdown_custom_questions(md_path: Path) -> List[Dict[str, str]]:
    """
    Very simple parser for the markdown format described earlier:
      ## section: <Name>
      - metric_id: <id>
        metric_question: <text>

    Returns list of {section, metric_id, metric_question}
    """
    if not md_path.exists():
        return []
    text = md_path.read_text(encoding="utf-8")

    items: List[Dict[str, str]] = []
    section = "Custom"

    lines = [l.rstrip() for l in text.splitlines()]
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.lower().startswith("## section:"):
            section = ln.split(":", 1)[1].strip() or "Custom"
            i += 1
            continue
        if ln.startswith("- "):
            # read the bullet + following indented lines
            buf = [ln[2:]]
            j = i + 1
            while j < len(lines) and (lines[j].startswith("  ") or lines[j].startswith("\t")):
                buf.append(lines[j].strip())
                j += 1
            blob = "\n".join(buf)
            metric_id = None
            metric_question = None
            for part in blob.splitlines():
                if part.lower().startswith("metric_id:"):
                    metric_id = part.split(":", 1)[1].strip()
                elif part.lower().startswith("metric_question:"):
                    metric_question = part.split(":", 1)[1].strip()
            if metric_id and metric_question:
                items.append({"section": section, "metric_id": metric_id, "metric_question": metric_question})
            i = j
            continue
        i += 1

    return items

def parse_markdown_prompts(md_path: Path) -> List[Dict[str, str]]:
    """
    Minimal prompt parser:
      ## section: <Name>
      - prompt_id: <id>
        prompt_text: <text>
        prompt_type: query|evidence_check|summary (optional)
    Returns list of {section, prompt_id, prompt_text, prompt_type}
    """
    if not md_path.exists():
        return []
    text = md_path.read_text(encoding="utf-8")

    items: List[Dict[str, str]] = []
    section = "Custom"

    lines = [l.rstrip() for l in text.splitlines()]
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.lower().startswith("## section:"):
            section = ln.split(":", 1)[1].strip() or "Custom"
            i += 1
            continue
        if ln.startswith("- "):
            buf = [ln[2:]]
            j = i + 1
            while j < len(lines) and (lines[j].startswith("  ") or lines[j].startswith("\t")):
                buf.append(lines[j].strip())
                j += 1
            blob = "\n".join(buf)
            prompt_id = None
            prompt_text = None
            prompt_type = "query"
            for part in blob.splitlines():
                if part.lower().startswith("prompt_id:"):
                    prompt_id = part.split(":", 1)[1].strip()
                elif part.lower().startswith("prompt_text:"):
                    prompt_text = part.split(":", 1)[1].strip()
                elif part.lower().startswith("prompt_type:"):
                    prompt_type = part.split(":", 1)[1].strip() or "query"
            if prompt_id and prompt_text:
                items.append({"section": section, "prompt_id": prompt_id, "prompt_text": prompt_text, "prompt_type": prompt_type})
            i = j
            continue
        i += 1

    return items

# ---------- Embedding / Indexing stubs ----------

def fake_embed_texts(texts: List[str]) -> List[Dict[str, Any]]:
    """Return pseudo-embeddings as tiny hashes (no external deps)."""
    out = []
    for t in texts:
        # trivial hash
        h = abs(hash(t)) % (10**8)
        out.append({"text": t, "embedding": [float((h >> i) & 255) for i in range(0, 24, 8)]})
    return out

def build_simple_index(embeds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Toy index: just stores count and checksum."""
    checksum = sum(int(sum(v["embedding"])) for v in embeds)
    return {"count": len(embeds), "checksum": int(checksum)}

# ---------- Analysis (chat stub) ----------

def chat(prompt: str) -> str:
    """
    STUB. Replace with your LLM call.
    For now, echoes a short placeholder so the pipeline is end-to-end runnable.
    """
    return f"[auto-analysis] {prompt[:300]}"

def run_analysis(rev: str, questions: List[Dict[str, str]], prompts: List[Dict[str, str]], out_dir: Path, report_path: Path):
    results = {"revision": rev, "questions": [], "prompts": []}

    # Per-question analysis
    for q in questions:
        qtext = f"{q['metric_id']}: {q['metric_question']}"
        resp = chat(qtext)
        results["questions"].append({"metric_id": q["metric_id"], "question": q["metric_question"], "response": resp})

    # Per-prompt analysis
    for p in prompts:
        ptext = f"{p['prompt_id']} ({p.get('prompt_type','query')}): {p['prompt_text']}"
        resp = chat(ptext)
        results["prompts"].append({"prompt_id": p["prompt_id"], "prompt_text": p["prompt_text"], "prompt_type": p.get("prompt_type","query"), "response": resp})

    # Save JSON
    (out_dir / "analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Write consolidated Markdown report
    lines = []
    lines.append(f"# QA Analysis Report – {rev}")
    lines.append("")
    lines.append("## Questions")
    for r in results["questions"]:
        lines.append(f"- **{r['metric_id']}** – {r['question']}")
        lines.append(f"  - Response: {r['response']}")
    lines.append("")
    lines.append("## Prompts")
    for r in results["prompts"]:
        lines.append(f"- **{r['prompt_id']}** ({r['prompt_type']}) – {r['prompt_text']}")
        lines.append(f"  - Response: {r['response']}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------- Pipeline Stages ----------

def process_revision(rev_dir: Path, cfg: Dict[str, Any]):
    rev = rev_dir.name
    log(f"Processing {rev}...")

    # 1) INGEST
    ingest_dir = DIRS["ingest"] / rev
    copytree(rev_dir, ingest_dir)
    log(f"  Ingested → {ingest_dir.relative_to(ROOT)}")

    # 2) EXTRACT (questions & prompts → JSON)
    extract_dir = DIRS["extract"] / rev
    extract_dir.mkdir(parents=True, exist_ok=True)

    core_files = [ingest_dir / n for n in cfg["analysis"]["core_yaml_filenames"]]
    questions = parse_core_yaml(core_files)

    for md_name in cfg["analysis"]["custom_checklist_filenames"]:
        questions += parse_markdown_custom_questions(ingest_dir / md_name)

    prompts: List[Dict[str, str]] = []
    for md_name in cfg["analysis"]["custom_prompts_filenames"]:
        prompts += parse_markdown_prompts(ingest_dir / md_name)

    (extract_dir / "questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")
    (extract_dir / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    log(f"  Extracted → {extract_dir.relative_to(ROOT)}")

    # 3) EMBED (stub)
    embed_dir = DIRS["embed"] / rev
    embed_dir.mkdir(parents=True, exist_ok=True)
    texts = [f"{q['metric_id']}: {q['metric_question']}" for q in questions] + [f"{p['prompt_id']}: {p['prompt_text']}" for p in prompts]
    embeddings = fake_embed_texts(texts)
    (embed_dir / "embeddings.json").write_text(json.dumps(embeddings, indent=2), encoding="utf-8")
    log(f"  Embedded → {embed_dir.relative_to(ROOT)}")

    # 4) INDEX (stub)
    index_dir = DIRS["index"] / rev
    index_dir.mkdir(parents=True, exist_ok=True)
    index_data = build_simple_index(embeddings)
    (index_dir / "index.json").write_text(json.dumps(index_data, indent=2), encoding="utf-8")
    log(f"  Indexed → {index_dir.relative_to(ROOT)}")

    # 5) ANALYSIS (per item chat + aggregate report)
    analysis_dir = DIRS["analysis"] / rev
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_name = cfg["analysis"]["report_name_template"].format(rev=rev)
    report_path = DIRS["reports"] / report_name
    run_analysis(rev, questions, prompts, analysis_dir, report_path)
    log(f"  Analyzed → {analysis_dir.relative_to(ROOT)}")
    log(f"  Report   → {report_path.relative_to(ROOT)}")

def loop_once(cfg: Dict[str, Any]) -> int:
    processed = load_processed()
    already = set(processed.get("processed", []))
    new = []

    # Find candidate revisions in drop zone
    for rev_dir in find_new_revisions(cfg):
        if rev_dir.name in already:
            continue
        new.append(rev_dir)

    if not new:
        return 0

    # Process up to max_batch
    count = 0
    for rev_dir in new[: int(cfg.get("max_batch", 5))]:
        try:
            process_revision(rev_dir, cfg)
            already.add(rev_dir.name)
            count += 1
        except Exception as e:
            log(f"ERROR while processing {rev_dir.name}: {e}")
    processed["processed"] = sorted(already)
    save_processed(processed)
    return count

def main():
    ensure_dirs()
    cfg = load_cfg()

    ap = argparse.ArgumentParser(description="QA Controls – watcher/runner")
    ap.add_argument("--once", action="store_true", help="Run a single pass then exit.")
    args = ap.parse_args()

    if args.once:
        loop_once(cfg)
        return

    log("Watcher started. Drop a new REVxxxxx folder into qa_drop/ to trigger the pipeline.")
    while True:
        handled = loop_once(cfg)
        time.sleep(int(cfg.get("poll_seconds", 3)))

if __name__ == "__main__":
    main()
```

---

## How to use (quick start)

1. **Run setup once**

```bash
python qa_controls_setup.py
```

2. **Start the watcher**

```bash
python qa_controls.py
```

3. **Drop a package** (example)

```
qa_drop/
  REV00001/
    mvr_feedback_metrics.yaml
    custom_checklist.md        # optional
    custom_prompts.md          # optional
```

4. Watch it flow:

```
qa_ingest/REV00001/
qa_extract/REV00001/{questions.json,prompts.json}
qa_embed/REV00001/embeddings.json
qa_index/REV00001/index.json
qa_analysis/REV00001/analysis.json
qa_reports/REV00001_report.md
```

---

## Notes / Next steps

* **Plug in your real LLM**: replace `chat()` with your `chat()` API call.
* **Tighten parsing** as your YAML/MD formats stabilize (right now it’s forgiving so reviewers can experiment).
* **Permissions**: if `qa_drop/` is a shared network folder, the polling loop already handles new arrivals; you can switch to `--once` in CI/cron.
* **Lockdown phase**: later, add an admin “promote” command to take a tested REV set and publish/lock a new core protocol version.
