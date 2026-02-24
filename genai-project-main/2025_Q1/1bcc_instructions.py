# 🚀 instructions_helper.py - Handles Loading Instructions from Local File

from configuration import *  # ✅ Universal Import

# **🔹 Local Instructions File Path**
INSTRUCTIONS_FILE = "instructions.json"

def load_instructions():
    """Loads the instructions from a local JSON file (No S3)."""
    try:
        if not os.path.exists(INSTRUCTIONS_FILE):
            logging.error(f"❌ Instructions file {INSTRUCTIONS_FILE} not found.")
            return {"message": "Error: Instructions file is missing.", "sections": []}

        with open(INSTRUCTIONS_FILE, "r", encoding="utf-8") as file:
            instructions = json.load(file)
            logging.info(f"✅ Loaded instructions from {INSTRUCTIONS_FILE}")
            return instructions

    except Exception as e:
        logging.error(f"❌ Error loading instructions: {e}")
        return {"message": "Error: Unable to load instructions.", "sections": []}
