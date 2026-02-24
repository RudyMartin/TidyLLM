# 📘 About: Extraction Utils

This module provides utility functions for extracting text from PDFs, cleaning the text, segmenting it into chunks, and saving structured outputs (JSON) to S3. It serves as the preprocessing backbone for document ingestion pipelines.

| Function Name                      | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `clean_text`                      | Cleans raw text from PDFs (whitespace, artifacts, hyphens).                |
| `clean_text_for_display`         | Further cleans text for UI readability and presentation.                   |
| `smart_chunking`                 | Splits long text into semantic chunks using sentence and sub-sentence logic. |
| `chunk_text_into_segments`       | Applies smart_chunking to individual PDF page content and attaches IDs.    |
| `split_pdf_into_pages_s3`        | Splits a multi-page PDF stored in S3 into separate single-page PDFs.       |
| `prepare_pdf_splits`             | Batch processes all PDFs in a folder and splits them into pages.           |
| `extract_text_from_pdf_page_s3`  | Extracts cleaned text from a single-page PDF file from S3.                 |
| `validate_page_continuity`       | Detects sentence continuity across page boundaries.                        |
| `is_blank_page`                  | Detects whether a page is blank or has no meaningful text/chunks.          |
| `extract_text_to_json`           | Orchestrates full page-to-chunk-to-JSON workflow for all page PDFs.        |
| `save_page_chunks_as_json_s3`    | Saves structured chunked output to a JSON file in S3.                      |
| `extract_text_from_pdf`          | Alias to `extract_text_to_json` (for CLI compatibility).                   |
| `save_extracted_text_to_json`    | Alias to `extract_text_to_json` (for alternate CLI path).                  |

Each function accepts a `config` and `s3_client` parameter where applicable, making it portable across CLI and Streamlit contexts.

> These functions are critical to preparing documents for vectorization, semantic search, or DSPy-based pipelines.
