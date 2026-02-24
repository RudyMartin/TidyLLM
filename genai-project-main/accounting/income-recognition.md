

### 🧾 1. Recognition of Revenue (GAAP Principle)

Under GAAP, **mined cryptocurrency is recognized as revenue when it is earned and realizable**, meaning:

* You’ve completed the mining process (block reward confirmed to your wallet), and
* You can measure the value (typically fair market value in USD at that date).

So technically, **the income event occurs when the BTC reward is credited to your wallet**, not when it’s later converted or transferred to your checking account.
👉 However, for **small-business simplification (cash basis)**, it’s acceptable to recognize income **when the proceeds are received in cash**—if you consistently use cash accounting.

✅ **Your method:** “Market value determined as net cash received in Business Checking Account upon receipt”
→ **is acceptable under cash-basis accounting**, which most small single-member LLCs use.
🚫 Under **accrual-basis GAAP**, you’d need to mark income at the BTC/USD fair value on the date received in the wallet.

---

### ⚖️ 2. Typical Journal Entries

If you’re using **cash basis**:

```
Dr. Checking Account     $X,XXX
   Cr. Mining Revenue          $X,XXX
```

If using **accrual basis (mark-to-market at wallet receipt)**:

```
Dr. Crypto Asset (BTC)   $X,XXX  (measured at FMV at date received)
   Cr. Mining Revenue          $X,XXX
```

Then when you convert to cash:

```
Dr. Cash (USD)           $X,XXX
Dr./Cr. Gain/Loss (if BTC price changed)
   Cr. Crypto Asset (BTC)     $X,XXX
```

---

### 🪙 3. Your Documentation Should Include

| Field                    | Example Entry                                                      |
| ------------------------ | ------------------------------------------------------------------ |
| **Wallet Address**       | bc1q8qftjc50g49rmt8kdx333te76zatztg200308f                         |
| **Mining Pool**          | Ocean.xyz                                                          |
| **Reward Frequency**     | Every 7–10 days                                                    |
| **Reward Formula**       | ≈ 0.012 × BTC price                                                |
| **Recognition Basis**    | Cash basis (when proceeds received in business checking account)   |
| **Exchange Rate Source** | Coinbase USD/BTC at time of deposit                                |
| **Ledger Account**       | Mining Revenue → Income                                            |
| **Tax Category**         | Ordinary business income (Schedule C or 1120-S if S-Corp election) |

---

### 🧮 4. Example

If Ocean.xyz pays 0.012 BTC and BTC = $65,000:

**At Wallet Receipt (optional accrual):**

```
Dr. Crypto Asset  $780
   Cr. Mining Revenue $780
```

**At Cash Deposit (your simplified method):**

```
Dr. Checking Account  $775   (after exchange fees)
   Cr. Mining Revenue   $775
```

---

### ✅ Summary

| Accounting Basis                       | Your Approach                            | GAAP Conformity             |
| -------------------------------------- | ---------------------------------------- | --------------------------- |
| **Cash Basis (typical for small LLC)** | Recognize when USD hits checking account | ✅ Acceptable                |
| **Accrual Basis (strict GAAP)**        | Recognize at BTC receipt (FMV)           | ✅ Required for larger firms |

---

If you plan to elect **S-Corp** status or expand reporting later, I’d recommend:

1. Start tracking both **wallet receipt logs (timestamp + BTC amount)**, and
2. **USD equivalents** at both the wallet and bank stages—so you can reconcile if you switch to accrual accounting.

Would you like me to draft a **template ledger + policy note** you can drop into your accounting files (e.g., in Markdown or Excel format)?
