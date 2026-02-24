Here’s a clean **`MRM_REQS.md`** draft that formats your DocCompliance Pro project notes so they’re consistent with the structure in `MRM_SPECS.md` and ready to compare for the 9-page limit review.

---

# 📋 **DocCompliance Pro – MRM Requirements**

## 1️⃣ **Assignments**

* **Source**: Inventory of model governance data in **MUSE** (Salesforce).
* **QA Manager Role**:

  * Determines percentage of reviews to sample.
  * Decides which reviews are in progress, which to sample, and which not to sample.
  * Monitors quarterly volumes and activity trends.
* **References**:

  * **Alex Snow** for algorithm and parameters to consider.

---

## 2️⃣ **Document Sources**

* All documents stored in **MUSE**, except:

  * Highly restrictive documents → stored on shared drives with specific permission controls.
* **MVP1**:

  * No repository creation.
  * Manual upload of files for QA review.
* **Future**:

  * Box integration planned for later in 2025.

---

## 3️⃣ **Control Checkpoints**

* All **control checkpoints** must be documented by validators.
* Validator control checkpoints must also be documented by QA.
* **MVP1 Scope**:

  * Focuses on validator control checkpoints only.

---

## 4️⃣ **QA Reviewer UX**

* Focus QA screen on:

  * **Assigned tasks** (do this), not process flows.
  * Missing items and pending QA reports.
* All **job aid** steps must be integrated:

  * Pre-filled via chat where possible.
  * Otherwise completed manually by reviewer.
* Provide **inline PDF view** of the document under review.

---

## 5️⃣ **Metadata**

* Extract from **EMAR** or **RDS** at the **time of review**.
* **Critical**: Match date to model version and review details.
* **Reference**:

  * **Hannah Shin** for sourcing guidance.

---

## 6️⃣ **Policy Documents**

* Include **Policy Standards** that are classified as *Guidelines*.
* Run through AI to:

  * Check consistency.
  * Suggest clarity improvements.
  * Recommend implementation approaches.
* **Reference**:

  * **Mark Wustefield** for standards content.

---

## 7️⃣ **Activity Tracking & Usage Metrics**

* **Usage** is a core success metric for the program.
* **User tiers**:

  1. **QA Team** – Document that all controls have been met. Share results (e.g., PDF report).
  2. **Validation Staff** – Run tool as part of peer review step.
  3. **Model Development Community** – Use as pre-flight check to improve documents over time.
* **Progression Tracking**:

  * Historical scoring to show improvement over time.


---

If you want, I can go through `MRM_SPECS.md` and mark exactly where each of these 11 requirements **maps in** (or is missing) so you have a page-by-page change plan for the 9-page limit. That will make the spec merge smooth. Would you like me to do that mapping next?
