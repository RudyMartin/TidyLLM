# 📦 How-to : Run Code Exports (Clean, Validate, and Zip)

This guide walks through using the `run_export_pipeline.py` script to prepare and package your project for clean export, audit, or release.

---

## ✅ What the Export Pipeline Does

The `run_export_pipeline.py` script performs:

1. **Code Clean-up**
   - Scans for incomplete files with WIP markers
   - Removes `.log` files, `__pycache__`, and `.ipynb_checkpoints` (outside dev/.git)

2. **Structural Diagnostics**
   - Verifies folder structure, flags banned files
   - Detects missing `__init__.py` or unversioned index files

3. **Manifest Generation**
   - Extracts and injects header comments
   - Outputs a versioned CSV + Markdown manifest

4. **Zipping**
   - Creates a versioned ZIP archive
   - Skips `dev/`, `.git/`, `.ipynb_checkpoints`, and `__pycache__`

---

## 🚀 How to Run

### 📦 Default (run everything from root):
```bash
python admin/run_export_pipeline.py
```

### 📝 Include a custom manifest:
```bash
python admin/run_export_pipeline.py --manifest manifest.csv
```

### 📁 Run on a specific folder:
```bash
python admin/run_export_pipeline.py --folder ./my_project --destination ./exports
```

---

## 🧼 What Gets Removed Automatically

| Type               | Action        |
|--------------------|---------------|
| `.log` files       | ✅ Removed (except in dev/) |
| `.ipynb_checkpoints/` | ✅ Fully removed |
| `__pycache__/`     | ✅ Fully removed |
| `"STILL NEEDS WORK"` | ❌ Blocks export |
| `dev/`, `.git/`    | ✅ Preserved and skipped |

---

## 🗂 Output Files

| File Type              | Where                     |
|------------------------|---------------------------|
| `file_index_*.csv`     | Root or destination folder |
| `file_index_*.md`      | Same as CSV               |
| `qaz_d_*.zip`          | Saved inside `dev/`       |

---

## 💡 Tips

- ✅ Add `# update_manifest` next to your description header to sync updates back to the manifest.
- 🧪 Use Git to track clean history between ZIP exports.
- 📁 Keep your `dev/` folder for internal artifacts and temporary logs.

---

## 📜 Example Log Output

```bash
✅ No banned filenames found.
✅ All index files are versioned correctly.
✅ Diagnostics complete.
✅ CSV file index written to file_index_20250408_9.csv
✅ Zipped into /dev/qaz_d_20250408_9.zip
```

---

## 👨‍🔬 Good to Know

- You can add `SKIP_FOLDERS` in each script to quarantine any custom folders.
- The entire system is modular and can be expanded with:
  - `--dry-run`
  - `--export-log`
  - `--target zip_only` or `--target diagnose`

---

Happy exporting! 🧼🗂️📦  
