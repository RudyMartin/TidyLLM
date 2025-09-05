
import streamlit as st
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Initialize session state
if 'split_data' not in st.session_state:
    st.session_state.split_data = None

def tab_split_dataset():
    st.header("🔀 Split Data into Train/Test Sets")
    file = st.file_uploader("Upload Normalized Data", type="json")
    test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20)
    if file:
        data = json.load(file)
        train, test = train_test_split(data, test_size=test_size/100)
        st.write("Train Size:", len(train), "Test Size:", len(test))
        st.download_button("Download Train Set", data=json.dumps(train), file_name="train_examples.json")
        st.download_button("Download Test Set", data=json.dumps(test), file_name="test_examples.json")
