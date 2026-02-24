
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def tab_evaluate_models():
    st.header("📊 Evaluate Models")
    gt = st.file_uploader("Upload Ground Truth", type="json")
    preds = st.file_uploader("Upload Predictions", type="json")
    if gt and preds:
        df_gt = pd.read_json(gt)
        df_pr = pd.read_json(preds)
        df = pd.merge(df_pr, df_gt, on=["topic", "report_chunk"], suffixes=("_pred", "_true"))
        df["correct"] = df["validation_result_pred"] == df["validation_result_true"]
        acc_by_topic = df.groupby("topic")["correct"].mean().reset_index()
        st.write("Accuracy by Topic", acc_by_topic)

        labels = sorted(set(df["validation_result_true"]) | set(df["validation_result_pred"]))
        cm = confusion_matrix(df["validation_result_true"], df["validation_result_pred"], labels=labels)
        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(cm, index=labels, columns=labels), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
