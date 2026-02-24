import json
import pandas as pd

# Function: Export JSON data to an Excel file
def export_to_excel(json_data, excel_filename):
    """
    Converts JSON structure to an Excel file with separate columns for input and output.
    """
    df = pd.DataFrame(json_data)
    df.to_excel(excel_filename, index=False)
    print(f"Exported JSON examples to {excel_filename}")

# Function: Convert Excel back to JSON format
def excel_to_json(excel_filename, json_filename):
    """
    Reads an Excel file and converts it into JSON format for DSPy agents.
    """
    df = pd.read_excel(excel_filename)
    json_data = df.to_dict(orient="records")

    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4)
    
    print(f"Converted Excel file {excel_filename} to JSON {json_filename}")

# Function: Load example JSON for DSPy agents
def import_example_json(json_filename):
    """
    Loads JSON examples from a file for use in DSPy agents.
    """
    with open(json_filename, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    
    return json_data

# Example Usage
if __name__ == "__main__":
    # Example JSON Structure (like DSPy examples)
    example_data = [
        {"input": "What are the revenue drivers?", "output": "Retrieve financial performance data."},
        {"input": "Summarize cost reduction trends", "output": "Find reports on expense reductions."}
    ]

    # Export JSON examples to an Excel file
    export_to_excel(example_data, "example_data.xlsx")

    # Convert the Excel file back to JSON
    excel_to_json("example_data.xlsx", "example_data.json")

    # Load the JSON examples
    loaded_examples = import_example_json("example_data.json")
    print("Loaded Examples:", loaded_examples)
