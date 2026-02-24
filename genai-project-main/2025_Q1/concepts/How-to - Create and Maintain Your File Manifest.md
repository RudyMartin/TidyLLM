# 📦 How-to: Create and Maintain Your File Manifest

This guide walks you through generating a complete project manifest that indexes source files, adds descriptions, and optionally injects or syncs headers between files and the manifest.

---

## 🔧 What is the Manifest?

The `manifest.csv` tracks each file’s:
- ✅ Relative path (`file_name`)
- ✅ Human-readable description (`file_description`)

It serves as a **single source of truth** for documentation, automation, and internal tooling.

---

## 🚀 Step 1: Generate Manifest from Source

Use the main tool:

```bash
python generate_file_index.py --manifest manifest.csv
```

### Options:
- `--inject_headers` → Adds missing `[File:]` headers to `.py` files using the manifest
- `--output` → Specify output CSV filename (default is auto-versioned)
- `--root` → Specify folder to scan (default: current directory)

---

## ✏️ Step 2: Manually Edit Descriptions

Edit `manifest.csv` directly or update the `Description:` line at the top of the file.

Example:

```python
"""
[File: embedding_helper.py]
Description: Embeds and normalizes text using Amazon Titan.  # UPDATE_MANIFEST
"""
```

---

## 🔄 Step 3: Sync Back to Manifest

If you change a `Description:` line in a `.py` file and want it **written back into the manifest**, add:

```python
# UPDATE_MANIFEST
```

Then run the indexer again:

```bash
python generate_file_index.py --manifest manifest.csv
```

Files with this flag will overwrite their own `file_description` field in the manifest.

---

## 🚫 Excluded Files

The system automatically **ignores**:
- Files beginning with `z_`
- Files containing `admin` or `__init__`
- Anything inside `.ipynb_checkpoints/`
- Binary extensions like `.pyc`

---

## 📄 Output Files

- ✅ `file_index_YYYYMMDD.csv` — Sorted file index
- ✅ `file_index_YYYYMMDD.md` — Markdown table for documentation

---

## 🧠 Best Practices

- Use short but meaningful `Description:` lines
- Keep notes and internal comments **in a second `"""` block**
- Use the manifest to audit and track file lifecycle
- Flag `# UPDATE_MANIFEST` only for intentional updates

---

