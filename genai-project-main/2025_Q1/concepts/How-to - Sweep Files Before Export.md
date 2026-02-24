# 🧹 How-to: Sweep Files Before Export

The `sweep_for_export.py` script removes temporary, dev, and junk files from your project directory to prep for public release, zip packaging, or GitHub sync.

---

## ✅ What It Does

| Action                  | Details                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| 🔍 Runs Diagnostics      | Executes `run_project_check.py` to validate structure before sweeping   |
| 🧹 Removes Dev Folders   | Deletes `logs/`, `dev/`, `untitled*/`, and folders starting with `z_`  |
| 🧻 Deletes Junk Files    | Deletes `.log`, `untitled*`, `z_*.py`, and `file_index.csv`             |
| 🛡️ Honors Preserve Tags | Keeps files/folders that match `--preserve` keywords                    |
| 🧾 Saves a Log File      | Records everything deleted into `clean_files.txt`                       |

---

## 🚀 How to Use It

```bash
python admin/sweep_for_export.py --yes
```

This will:
1. Run diagnostics
2. Clean the directory
3. Save a cleanup log to `clean_files.txt`

---

## 🧪 Examples

### 🧼 Just Preview (No Deletes)
```bash
python admin/sweep_for_export.py --dry-run
```

### ✅ Confirm Cleanup
```bash
python admin/sweep_for_export.py --yes
```

### 🛡️ Keep Special Files (e.g., `z_experiment_keep.py`)
```bash
python admin/sweep_for_export.py --yes --preserve keep snapshot
```

### 🗂 Custom Log Location
```bash
python admin/sweep_for_export.py --yes --log exports/clean_2025.txt
```

### 🚫 Keep .log Files
```bash
python admin/sweep_for_export.py --yes --keep-logs
```

---

## 🧠 Pro Tips

- Add this to your **pre-release checklist**
- Run before building your zip/streamlit package
- Pair with `run_project_check.py` for QA + cleanup
- Commit your sweep log for traceability (`git add clean_files.txt`)

---

## 📂 Script Location

```
admin/sweep_for_export.py
```

---

## 🔁 Related Tools

- `admin/run_project_check.py` → Full QA pipeline
- `admin/generate_file_index.py` → Manifest builder & header injector
- `file_index.csv` → File list + descriptions

---

Keep it lean. Keep it clean. 🚀
