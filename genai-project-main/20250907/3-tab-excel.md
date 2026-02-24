
---

# TL;DR (what I’d implement)

* Keep **core** immutable; experiment in **test layers**.
* Excel has **3 tabs** + a tiny bit of metadata to support versioning and overrides.
* Import Excel → **DB** with **clear precedence rules**: `core` ⟂ `test_custom` ⟂ `test_prompts`.
* Everyone can “play” in test; **nothing touches core** until promotion.
* Later, an **admin promote script** snapshots a test set into the “core” tables and bumps versions.

---

# 1) Excel structure (3 tabs + metadata)

### Tab A: `core_checklist` (read-only to reviewers)

Columns (for each metric/row):

* `protocol_version` (e.g., `1.0`)
* `section`
* `metric_id`
* `metric_question`
* `is_core` = `yes` (constant)
* `status` (optional reviewer tracking; does not impact core)
* `notes` (optional reviewer notes; does not impact core)

> Core rows are the baseline; you’ll import them into a **core table** and mark them immutable.

### Tab B: `custom_checklist` (sandbox)

* `protocol_version` (e.g., `1.0-test-r1`)
* `section`
* `metric_id` (can be new or override an existing `core` one)
* `metric_question`
* `owner` (reviewer name/alias)
* `status` (Pending/Active/Deprecated)
* `notes`

### Tab C: `custom_prompts` (sandbox)

* `protocol_version` (e.g., `1.0-test-r1`)
* `section`
* `prompt_id`
* `prompt_text`
* `prompt_type` (query / evidence\_check / summary)
* `owner`
* `status` (Pending/Active/Deprecated)
* `notes`

> Tip: add a small “About” sheet with `protocol_version`, `created_by`, `created_at`, `environment = test|locked`.

---

# 2) DB ingestion (Excel → DB)

### Minimal tables (SQLite/Postgres; same idea)

```sql
-- Core (immutable once published)
CREATE TABLE core_checklist (
  protocol_version TEXT NOT NULL,
  section          TEXT NOT NULL,
  metric_id        TEXT NOT NULL,
  metric_question  TEXT NOT NULL,
  PRIMARY KEY (protocol_version, metric_id)
);

-- Reviewer sandbox (test layer)
CREATE TABLE test_custom_checklist (
  protocol_version TEXT NOT NULL, -- e.g., 1.0-test-r1
  section          TEXT NOT NULL,
  metric_id        TEXT NOT NULL,
  metric_question  TEXT NOT NULL,
  owner            TEXT,
  status           TEXT CHECK (status IN ('Pending','Active','Deprecated')) DEFAULT 'Active',
  notes            TEXT,
  created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prompts sandbox
CREATE TABLE test_custom_prompts (
  protocol_version TEXT NOT NULL,
  section          TEXT NOT NULL,
  prompt_id        TEXT NOT NULL,
  prompt_text      TEXT NOT NULL,
  prompt_type      TEXT CHECK (prompt_type IN ('query','evidence_check','summary')) NOT NULL,
  owner            TEXT,
  status           TEXT CHECK (status IN ('Pending','Active','Deprecated')) DEFAULT 'Active',
  notes            TEXT,
  created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (protocol_version, prompt_id)
);

-- Light audit log
CREATE TABLE audit_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  actor           TEXT NOT NULL,
  action          TEXT NOT NULL,
  target          TEXT NOT NULL,      -- table/row key
  meta            TEXT,               -- JSON blob
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Import behavior

* **core\_checklist tab** → upsert into `core_checklist` only if protocol\_version is **new** (or if you intentionally republish).
* **custom\_checklist tab** → insert as **new test rows**; allow duplicates across protocol\_version so teams can branch experiments.
* **custom\_prompts tab** → same as above.

---

# 3) Precedence rules (filters / overrides)

When you **assemble the working checklist** for a test run:

1. Start with **core** for a given `core_protocol_version` (e.g., `1.0`):

```sql
SELECT section, metric_id, metric_question, 'core' AS source
FROM core_checklist
WHERE protocol_version = '1.0';
```

2. Overlay **active test custom** for the selected test version (e.g., `1.0-test-r1`):

* **If `metric_id` matches** a core `metric_id`, treat it as an **override** (use the custom row instead).
* **If `metric_id` is new**, treat it as an **addition**.

Example (Postgres-ish):

```sql
WITH core AS (
  SELECT section, metric_id, metric_question
  FROM core_checklist
  WHERE protocol_version = $1                  -- '1.0'
),
custom AS (
  SELECT section, metric_id, metric_question
  FROM test_custom_checklist
  WHERE protocol_version = $2                  -- '1.0-test-r1'
    AND status = 'Active'
),
overrides AS (
  -- rows in custom that replace core (by metric_id)
  SELECT c.section, c.metric_id, c.metric_question, 'custom_override' AS source
  FROM custom c
  JOIN core   k ON k.metric_id = c.metric_id
),
additions AS (
  -- rows in custom that are new (not in core)
  SELECT c.section, c.metric_id, c.metric_question, 'custom_addition' AS source
  FROM custom c
  LEFT JOIN core k ON k.metric_id = c.metric_id
  WHERE k.metric_id IS NULL
)
SELECT section, metric_id, metric_question, 'core' AS source
FROM core
WHERE metric_id NOT IN (SELECT metric_id FROM overrides)
UNION ALL
SELECT * FROM overrides
UNION ALL
SELECT * FROM additions
ORDER BY section, metric_id;
```

Prompts follow the same pattern: use only **Active** prompts for the chosen `protocol_version`.

---

# 4) Testing workflow (full transparency)

1. **Distribute Excel (test)** with three tabs.
2. Reviewers **edit only** `custom_checklist` + `custom_prompts`.
   (Core tab remains visible but not editable policy-wise.)
3. Import Excel → DB **test tables** (no core changes).
4. Run the app with a **selected test protocol** (e.g., `1.0 + 1.0-test-r1`):

   * UI uses the view/SQL above to resolve overrides & additions.
5. Iterate quickly:

   * Reviewers tweak questions/prompts.
   * Rerun import and refresh.
6. **Track diffs** via the DB + `audit_log` (who changed what, when).

> This gives reviewers full play in “test” without touching `core`.

---

# 5) Lockdown (“promotion”) when ready

Create a separate **admin script** that:

* Validates there are **no duplicate metric\_ids** and **no missing sections**.
* Generates the next **core protocol version** (e.g., `1.1`) from the **resolved** test set:

  * Copy **core + overrides + additions** into `core_checklist` with `protocol_version = '1.1'`.
  * Freeze it (treat as immutable).
* Archives the test set or marks it `Deprecated`.

> This is your single “gate” that turns experiments into the new standard.

---

# 6) Observability & safeguards (cheap but effective)

* Add a **header row** on each Excel tab that shows:

  * `core_protocol_version`, `test_protocol_version`, `environment`
* Add **counter metrics** after import:

  * `core_count`, `custom_override_count`, `custom_addition_count`
* Add a **lint** step before import:

  * Empty `metric_id`? Reject row.
  * Duplicate `metric_id` within the same test version? Warn/block based on strictness.
  * `status` not in enum? Normalize or block.

---

# 7) What this buys you

* **Rapid experimentation** (teams can try ideas, new sections, edits).
* **No risk to the baseline** (core never changes during test).
* **Reproducibility** (core/test versions are explicit).
* **Easy promotion** (a single, audited step to publish a new core).

---

## Want me to ship a minimal **Python importer** (Excel → SQLite) using this exact schema + precedence view?

It can:

* Read your current YAML to populate `core_checklist` (v1.0).
* Read any Excel (with 3 tabs) and upsert `test_custom_*` for a given test protocol version.
* Emit a resolved “effective checklist” CSV for verification.
