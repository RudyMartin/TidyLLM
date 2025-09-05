
import os
import dspy
import pickle

def save_compiled_pipeline(pipeline, model_name, save_dir="compiled_modules"):
    """Save a compiled DSPy pipeline to disk."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.dspy")
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    return save_path

def load_compiled_pipeline(model_name, save_dir="compiled_modules"):
    """Load a compiled DSPy pipeline from disk."""
    load_path = os.path.join(save_dir, f"{model_name}.dspy")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No compiled pipeline found at: {load_path}")
    with open(load_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline
