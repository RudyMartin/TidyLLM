

# 🔑 How to Handle “External Evidence” in Custom Checks

### Extend the schema with a new column:

| section\_id | custom\_question                        | evidence\_display | output\_format | external\_reference                |
| ----------- | --------------------------------------- | ----------------- | -------------- | ---------------------------------- |
| DataQuality | Does validation cover external vendors? | summary           | text           | “VendorDB: QA\_Results\_2025.xlsx” |
| Testing     | Show accuracy trend over iterations     | chart             | line           | “MLflow run #4582”                 |

---

# 📄 Report Example

**Custom Checks (Uploaded by Client)**

**Q:** *Does validation cover external vendors?*

* **Answer:** No mention found in main doc
* **External Reference (added by reviewer):** *VendorDB: QA\_Results\_2025.xlsx*
* **Commentary:** Please attach vendor QA results to complete this check

---

**Q:** *Show accuracy trend over iterations*

* **Answer (chart):** Line chart of accuracy trend
* **External Reference:** *MLflow run #4582*
* **Notes:** Pulled directly from experiment tracking

---

# 🛡 Guardrails

* **External references are *metadata only***:

  * They get logged in the report & JSON.
  * But the system doesn’t try to fetch that data automatically (keeps integration clean).
* **Audit-friendly:** Always visible in the “Needs Work / External Evidence” section so auditors see what came from outside.
* **Option:** Later, you could integrate connectors (e.g., MLflow, VendorDB) to auto-pull if client matures.

---

# 🎯 Pitch to Client

> “If you’re pulling from another system, just drop the reference into the **Custom Checks file**. It will show up in the report alongside the commentary. That way, reviewers (and auditors) can see clearly: *this check depends on VendorDB file XYZ or MLflow run #4582.*
>
> Over time, we can automate those integrations — but day one, we at least capture the external reference right in the report.”

---

👉 Do you want me to **mock up a “Custom Checks + External References” template** (Excel + JSON version) so you can hand it to them as the official format?
