
import streamlit as st
from dspy import Example
from core.dspy_prompt_config import render_dspy_prompt

def tab_dspy_prompt_configurator():
    st.header("🔍 Prompt Strategy Configuration (DSPy Enabled)")
    run_mode = st.radio("Run Mode:", options=["LLM", "DSPy"], index=0, key="run_mode")
    strategy = st.radio("Prompting Strategy:", options=["naive", "cot", "retrieval"], index=0, key="prompt_strategy")
    st.success(f"Current Strategy: {strategy} — Mode: {run_mode}")
    example = Example(topic="sustainability", report_chunk="CO2 reduction", retrieved_context="Must meet 30% by Q2")
    prompt = render_dspy_prompt(example, strategy=strategy, retrieved_texts=example.retrieved_context)
    st.code(prompt, language="markdown")
