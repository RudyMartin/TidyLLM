

### ✅`show_upload_tab()` with DSPy + Modular Extraction

```python
"""
[File: tabs_03/upload_embed_tab.py]
Description: Upload a PDF, extract and embed its text, save chunks to S3, and optionally preview answers using DSPy.
"""

import streamlit as st
import uuid
import dspy
from datetime import datetime

from core.init_context import get_tab_context
from core.extraction_helper import (
    split_pdf_into_pages_s3,
    extract_text_to_json,
    chunk_text_into_segments
)
from core.s3_utils import upload_to_s3, upload_json_to_s3


# Minimal DSPy Signature for Q&A
class SimpleQA(dspy.Signature):
    context = dspy.InputField(desc="Extracted chunks from the PDF")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Relevant answer")


def show_upload_tab():
    st.header("📄 Upload PDF ➜ Embed ➜ DSPy QA")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and st.button("🚀 Process PDF"):
        try:
            clients, config, vector_manager, embed_manager = get_tab_context()
            bucket = config["bucket_name"]
            model_key = config["default_model"]

            # Upload original PDF to S3
            pdf_id = uuid.uuid4().hex
            pdf_key = f"{config['pdf_folder']}/{pdf_id}.pdf"
            upload_to_s3(clients.s3, bucket, pdf_key, uploaded_file.read())
            st.success(f"✅ Uploaded PDF to `{pdf_key}`")

            # Step 1: Split PDF into pages
            split_pdf_into_pages_s3(pdf_key, config=config, s3_client=clients.s3)

            # Step 2: Extract structured page content into a JSON file on S3
            json_key = extract_text_to_json(config=config, s3_client=clients.s3)

            # Step 3: Further chunk the JSON file
            chunks = chunk_text_into_segments(json_key, config=config, s3_client=clients.s3)
            st.info(f"📚 Extracted {len(chunks)} chunks.")

            # Step 4: Embed and upload each chunk to S3
            for chunk in chunks:
                text = chunk["text"]
                embedding = embed_manager.generate(text)
                enriched = {
                    "id": str(uuid.uuid4()),
                    "chunk": text,
                    "embedding": embedding,
                    "metadata": {
                        "source_file": uploaded_file.name,
                        "chunk_id": chunk.get("chunk_id"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                upload_json_to_s3(
                    s3_client=clients.s3,
                    bucket=bucket,
                    folder=config["json_folder"],
                    filename=f"{enriched['id']}.json",
                    json_obj=enriched
                )

            st.success("✅ All chunks embedded and uploaded to S3.")

            # Step 5: DSPy QA (optional preview)
            st.divider()
            st.subheader("💬 Ask a question about the uploaded document")

            query = st.text_input("Enter a question:")
            if query and st.button("🤖 Ask with DSPy"):
                try:
                    if "upload_qa_module" not in st.session_state:
                        dspy.settings.configure(openai="gpt-3.5-turbo")  # or Bedrock backend
                        compiled = dspy.compile(BootstrapFewShot(SimpleQA))
                        st.session_state.upload_qa_module = compiled

                    context_text = "\n\n".join([chunk["text"] for chunk in chunks[:5]])  # limit preview
                    response = st.session_state.upload_qa_module(
                        context=context_text,
                        question=query
                    )
                    st.markdown("#### 🧠 Answer")
                    st.success(response.answer)

                except Exception as e:
                    st.error(f"❌ DSPy Q&A failed: {e}")

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
```

---

### 🔍 Summary of Enhancements

| Feature | Included |
|--------|----------|
| Modular PDF splitting & extraction | ✅ |
| Uses `get_tab_context()` | ✅ |
| Uses configured `embed_manager` | ✅ |
| Saves enriched JSON chunks to S3 | ✅ |
| **DSPy Q&A preview** after upload | ✅ |
| Keeps everything session-safe | ✅ |


