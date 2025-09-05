
import streamlit as st
import json

# Initialize session state
if 'edited_data' not in st.session_state:
    st.session_state.edited_data = None

def tab_edit_examples():
    st.header("📝 Edit Validation Examples")
    file = st.file_uploader("Upload Examples to Edit", type="json")
    if file:
        data = json.load(file)
        updated = []
        for i, ex in enumerate(data):
            with st.expander(f"Example {i+1}"):
                topic = st.text_input("Topic", value=ex.get("topic", ""), key=f"topic_{i}")
                chunk = st.text_area("Report Chunk", value=ex.get("report_chunk", ""), key=f"chunk_{i}")
                ctx = st.text_area("Retrieved Context", value=ex.get("retrieved_context", ""), key=f"context_{i}")
                label = st.selectbox("Validation Result", ["Correct", "Missing Info", "Inconsistent", "Other"], index=0, key=f"label_{i}")
                updated.append({
                    "topic": topic,
                    "report_chunk": chunk,
                    "retrieved_context": ctx,
                    "validation_result": label
                })
        st.download_button("Download Edited Examples", data=json.dumps(updated, indent=2), file_name="edited_examples.json")
