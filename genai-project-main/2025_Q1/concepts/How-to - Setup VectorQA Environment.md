# 🛠️ How-To: Setup VectorQA Environment

This guide explains how to prepare your QA-Z environment for local development or deployment on AWS (e.g., SageMaker Studio).

---

## 📦 Setup Script Overview (`setup.py`)

This script provides CLI commands to:
- ✅ Install required packages
- 🧹 Remove conflicting packages
- 🔍 Verify core dependencies
- 📁 Create required folders

---

## ✅ Requirements

- Python 3.8+
- `pip` available in your environment
- `custom_requirements.txt` file present

---

## 🚀 Running Setup Tasks

You can run any of the following commands independently:

### 🔧 Install Required Packages
```bash
python setup.py --install
```
Installs all Python dependencies from `custom_requirements.txt`.

### 🧹 Clean Conflicting Packages
```bash
python setup.py --clean
```
Uninstalls packages known to conflict with Bedrock or vector stack.

### 🔍 Verify Critical Libraries
```bash
python setup.py --check
```
Checks that `pyarrow.Decimal32Type` is available.

### 📁 Create Log Directories
```bash
python setup.py --init-dirs
```
Creates the following directory structure:
- `logs/`
- `logs/faiss/`
- `logs/indexing/`
- `logs/repair/`

### 🧪 Run All Setup Commands Together
```bash
python setup.py --clean --install --check --init-dirs
```

---

## 💡 Tips

- Use `--install` after updating your `custom_requirements.txt`
- Run `--clean` on fresh SageMaker instances to avoid `autogluon`, `dash`, and others
- If you run into any `pyarrow` issues, check with `--check`

---

## 🔁 Notes

This script supports multiple environments (CLI + Streamlit) and assumes you're operating under a folder like `~/qaz_d`.

Use this setup once per environment or rebuild when updating your local stack.

---

Happy building 🚀
