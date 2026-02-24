import zipfile
import os

zip_path = 'Concepts.zip'
folder_to_extract = 'qaz_c/concepts'
destination_path = os.path.join('.', folder_to_extract)

# Create the target folder if it doesn't exist
os.makedirs(destination_path, exist_ok=True)

# Extract everything into the new folder
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(destination_path)

print(f"✅ All contents extracted into folder: '{os.path.abspath(destination_path)}'")
