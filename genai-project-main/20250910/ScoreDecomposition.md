

## Step 1: Define the "Overall QA Score"

Think of this like ROE: a single percentage-style score (0–100%) that tells you the quality of the report package.

---

## Step 2: Decompose into Key Drivers

Analogous to profit margin, asset turnover, and leverage, we need 3–4 multiplicative components:

1. **Compliance Adherence (C)**

   * How well the SOP and Checklist are followed.
   * Inputs: SOP docs, QA checklist completion rate.
   * Example metric: `% of checklist items satisfied`.

2. **Scope Alignment (S)**

   * Whether the Validation Scope is correctly defined and covered.
   * Inputs: Validation Scope vs. what’s in the review.
   * Example metric: `% of required scope elements addressed`.

3. **Review Rigor (R)**

   * Quality and depth of the Validation Review itself.
   * Inputs: evidence coverage, testing completeness, independence.
   * Example metric: weighted score of review completeness vs. expected tests.

4. **Peer Review Effectiveness (P)**

   * Quality of secondary checks and comments.
   * Inputs: Peer Review doc — are issues flagged, resolved, or ignored?
   * Example metric: issue resolution ratio, % of substantive peer comments addressed.

---

## Step 3: Build a Multiplicative Formula

Like DuPont (ROE = Profit Margin × Asset Turnover × Leverage), we can do:

$$
QA\_Score = C \times S \times R \times P
$$

* Each component scaled 0–1 (or 0–100%).
* Multiplication ensures that a failure in one area (say, Scope = 0.3) drags down the total score, even if others look good.
* This mirrors how one weak link (bad scope definition, poor peer review) undermines the overall QA quality.

---

## Step 4: Apply Weighting (Optional Variant)

If you want more control, use a weighted geometric mean:

$$
QA\_Score = C^{w_c} \times S^{w_s} \times R^{w_r} \times P^{w_p}
$$

where $w_c + w_s + w_r + w_p = 1$.
For example: Compliance (0.3), Scope (0.2), Review (0.3), Peer Review (0.2).

---

## Step 5: Interpretation Framework

* **80–100%**: High-quality, robust report.
* **60–79%**: Adequate but gaps exist; corrective action needed.
* **<60%**: Weak QA process; risk of governance failure.

---

✅ This makes the score:

* **Transparent** (you can trace which factor is dragging performance),
* **Comparable** (across reports, teams, or time),
* **Actionable** (if Peer Review is low, you know where to fix).

---

Would you like me to **mock up a worked example** using a fictional Credit Risk Model report (e.g., Compliance = 0.85, Scope = 0.75, Review = 0.9, Peer Review = 0.6 → QA Score ≈ 35%) so you can see how the multiplicative drag works?

