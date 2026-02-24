import zipfile
import os
import datetime

# Import the sweep function from the sweeper module


def pack_folder(folder_to_zip, destination_path='.'):
    # First, call the sweeper to clean out unwanted files/folders
    ##sweeper.sweep_away(folder_to_zip)

    # Generate a date stamp in YYYYMMDD format
    date_stamp = datetime.datetime.now().strftime('%Y%m%d')

    # Build the base file name
    zip_base_name = f"{folder_to_zip}_{date_stamp}"

    # Check for existing file with the same name; if found, append _2, _3, etc.
    counter = 2
    zip_filename = f"{zip_base_name}.zip"
    zip_path = os.path.join(destination_path, zip_filename)

    while os.path.exists(zip_path):
        zip_filename = f"{zip_base_name}_{counter}.zip"
        zip_path = os.path.join(destination_path, zip_filename)
        counter += 1

    # Create the zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for root, dirs, files in os.walk(folder_to_zip):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, folder_to_zip)
                zip_ref.write(full_path, arcname=relative_path)

    print(f"Folder '{folder_to_zip}' packed into '{zip_path}'")

if __name__ == "__main__":
    folder_to_zip = "qaz"   # Replace with your folder name
    destination = "."      # Current directory by default
    pack_folder(folder_to_zip, destination)
