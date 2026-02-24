
import streamlit as st
import json
import dspy
from dspy.optimize import BootstrapFewShot
from core.config import CONFIG

class ValidateQA(dspy.Signature):
    topic = str
    report_chunk = str
    retrieved_context = str
    validation_result = str

class FewShotValidator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(ValidateQA)
    def forward(self, topic, report_chunk, retrieved_context):
        return self.classifier(topic=topic, report_chunk=report_chunk, retrieved_context=retrieved_context)

def tab_compile_dspy_pipeline():
    st.header("🧪 Compile DSPy Module")
    train_file = st.file_uploader("Upload Train Set", type="json")
    model_key = st.selectbox("Model Key", list(CONFIG["embedding_models"].keys()))
    if train_file and st.button("Compile"):
        data = json.load(train_file)
        trainset = [dspy.Example(**d) for d in data]
        compiled = dspy.compile(FewShotValidator(), trainset=trainset, compiler=BootstrapFewShot(metric="accuracy"))
        path = f"compiled_modules/{model_key.replace(':', '_')}_validator.dspy"
        compiled.save(path)
        st.success(f"Saved to {path}")
