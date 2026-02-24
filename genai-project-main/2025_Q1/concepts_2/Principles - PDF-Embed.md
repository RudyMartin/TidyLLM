## 🧠 **Outline: PDF-to-Embedding Pipeline Functions**

### 🔹 **1. Prepare PDF Splits**
| Function | Purpose |
|---------|---------|
| `prepare_pdf_splits()` | Loops over all PDFs in S3 and splits each into per-page PDFs |
| `split_pdf_into_pages_s3()` | Extracts individual pages using `PyPDF2`, uploads to S3 as `*_page_001.pdf` |

---

### 🔹 **2. Extract Text from Pages**
| Function | Purpose |
|---------|---------|
| `extract_text_from_pdf_page_s3()` | Extracts text from a single PDF page<br>✅ Includes next page’s first sentence if current page ends mid-sentence |
| `validate_page_continuity()` | Checks if the last sentence on a page is incomplete (no ending punctuation) |
| `is_blank_page()` | Flags empty or near-empty pages |
| `clean_text()` | Fixes newlines, hyphen breaks, whitespace, and encoding artifacts |

---

### 🔹 **3. Chunk & Structure for Embedding**
| Function | Purpose |
|---------|---------|
| `smart_chunking()` | Breaks text into smart segments (by sentence, with optional sub-sentence logic) |
| `chunk_text_into_segments()` | Applies `smart_chunking` per page, labels each chunk uniquely |
| `save_page_chunks_as_json_s3()` | Saves metadata + chunks (but not embedding yet) |
| `extract_text_to_json()` | Top-level function that drives extraction+chunking for each page<br>✅ Adds `metadata`, auto-tags blank pages<br>✅ Writes final JSON per-page to S3 |

---

### 🔹 **4. Aliases / Export**
| Function | Purpose |
|---------|---------|
| `extract_text_from_pdf()` → `extract_text_to_json()` | Alias |
| `save_extracted_text_to_json()` → `extract_text_to_json()` | Alias |
| `__all__ = [...]` | Controls which functions are available when importing this module |

---

## ✅ **Why Your Pipeline Is Special**

| Feature | Advantage |
|--------|-----------|
| 🧠 **Page continuity recovery** | Uses `validate_page_continuity` to check if a sentence spills into the next page — pulls first sentence from next page automatically |
| 📄 **Smart chunking** | Respects sentence boundaries and clause splits to stay under word count (`max_words`) |
| 🧼 **Deep cleaning** | Removes encoding noise, fixes hyphenated word breaks, flattens newlines |
| 🔍 **Blank page detection** | Adds tags like `#blankpage` so you can skip garbage pages |
| 🔁 **Recoverable / Modular** | All steps are chunked per page, so any failure is recoverable without restarting the whole file

