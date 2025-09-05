
import json
import os

DEFAULT_FIELDS = ["topic", "report_chunk", "retrieved_context", "validation_result"]

def load_examples(filepath):
    """Load examples from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate that data is a list
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of examples")
        
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file {filepath}: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"File encoding error in {filepath}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading file {filepath}: {e}")

def validate_example_structure(example):
    """Ensure each example has the required fields."""
    for field in DEFAULT_FIELDS:
        if field not in example:
            example[field] = ""
    return example

def normalize_examples(data):
    """Apply structure validation to all examples."""
    return [validate_example_structure(ex) for ex in data]

def add_example(data, new_example):
    """Add a new example to the dataset."""
    validated = validate_example_structure(new_example)
    data.append(validated)
    return data

def update_example(data, index, updated_example):
    """Update an existing example by index."""
    if index < 0 or index >= len(data):
        raise IndexError("Invalid index")
    validated = validate_example_structure(updated_example)
    data[index] = validated
    return data

def delete_example(data, index):
    """Delete an example by index."""
    if index < 0 or index >= len(data):
        raise IndexError("Invalid index")
    del data[index]
    return data

def save_examples(filepath, data):
    """Save examples to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

