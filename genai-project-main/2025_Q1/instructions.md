# 📘 How to Edit and Preview This Documentation

These steps help you or a collaborator preview, edit, and export this documentation as a browsable site or printable PDF.

---

## ✅ Requirements

Make sure Python is installed, then install MkDocs and the Material theme:

```bash
pip install mkdocs mkdocs-material
```

---

## 🚀 Preview the Documentation

Run the following commands from this folder (`docs/site`):

```bash
cd docs/site
mkdocs serve
```

This will start a local server at:

```
http://127.0.0.1:8000/
```

Visit that URL in your browser to view and test your documentation.

---

## 🖨️ Export to PDF

Once the site is open in your browser:

1. Go to the section you want to export
2. Use your browser menu → "Print"
3. Select "Save as PDF"

It will create a clean printable copy using the Material theme styles.

---

## ✏️ Editing Markdown Files

All source files are in:

- `../materials/` for how-to, DSPy, FAISS, and citation docs
- `../setup_guide.md`, `../development.md`, `../system_overview.md` for top-level sections

You can edit them with any text editor or Markdown editor.

To see your changes, just refresh the browser after editing!

---

## ✅ Bonus: Create New Sections

To add a new page:

1. Create a new `.md` file inside `docs/materials/` or `docs/`
2. Open `mkdocs.yml` and add a new nav entry like this:

```yaml
  - New Section: ../materials/my_new_doc.md
```

3. Save and refresh your local server.

Happy documenting!