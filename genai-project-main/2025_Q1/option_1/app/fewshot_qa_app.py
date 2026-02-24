
import streamlit as st
from app.tabs.fewshot_qa_app_tab1_upload_normalize import tab_upload_normalize
from app.tabs.fewshot_qa_app_tab2_split import tab_split_dataset
from app.tabs.fewshot_qa_app_tab3_edit_examples import tab_edit_examples
from app.tabs.fewshot_qa_app_tab4_dspy_prompt_config import tab_dspy_prompt_configurator
from app.tabs.fewshot_qa_app_tab5_evaluate import tab_evaluate_models
from app.tabs.fewshot_qa_app_tab6_faiss_status import tab_faiss_status
from app.tabs.fewshot_qa_app_tab7_compile_dspy import tab_compile_dspy_pipeline

st.set_page_config(page_title="VectorQA Sage", layout="wide")

st.sidebar.title("📚 VectorQA Sage")
tab = st.sidebar.radio("Select a tab", [
    "1. Upload & Normalize",
    "2. Split Dataset",
    "3. Edit Examples",
    "4. Prompt Config (DSPy)",
    "5. Evaluate Models",
    "6. FAISS & Model Status",
    "7. Compile DSPy Module"
])

if tab.startswith("1"):
    tab_upload_normalize()
elif tab.startswith("2"):
    tab_split_dataset()
elif tab.startswith("3"):
    tab_edit_examples()
elif tab.startswith("4"):
    tab_dspy_prompt_configurator()
elif tab.startswith("5"):
    tab_evaluate_models()
elif tab.startswith("6"):
    tab_faiss_status()
elif tab.startswith("7"):
    tab_compile_dspy_pipeline()
