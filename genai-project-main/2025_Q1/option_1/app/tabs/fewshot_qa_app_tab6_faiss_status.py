
import streamlit as st
import json
from core.config import CONFIG

def tab_faiss_status():
    st.header("📦 FAISS Index & Model Status")
    try:
        with open("data/faiss_model_map.json") as f:
            index_map = json.load(f)
        st.json(index_map)
    except Exception as e:
        st.warning("No FAISS index map found.")
