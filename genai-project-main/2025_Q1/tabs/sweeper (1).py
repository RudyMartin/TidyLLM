import os
import shutil
import sys

def sweep_away(target_dir):
    """
    Removes:
      1) Files/folders with 'untitled' in the name
      2) All logs and .log files
      3) The 'dev' folder (recursively)
      4) Any files/folders that start with 'z'
    """
    # Walk from the bottom up, so we can safely remove directories
    for root, dirs, files in os.walk(target_dir, topdown=False):
        
        # -- 1) Remove subdirectories if they match the sweep criteria --
        for d in dirs:
            dir_path = os.path.join(root, d)
            
            # If folder is named exactly "logs"
            if d == "logs":
                print(f"Removing folder: {dir_path}")
                shutil.rmtree(dir_path, ignore_errors=True)
                continue
            
            # If folder is named "dev"
            if d == "dev":
                print(f"Removing dev folder: {dir_path}")
                shutil.rmtree(dir_path, ignore_errors=True)
                continue
            
            # If folder name has "untitled" or starts with "z"
            if "untitled" in d.lower() or d.lower().startswith('z'):
                print(f"Removing folder: {dir_path}")
                shutil.rmtree(dir_path, ignore_errors=True)
        
        # -- 2) Remove files that match the sweep criteria --
        for f in files:
            file_path = os.path.join(root, f)
            lower_name = f.lower()
            
            # If file has "untitled" in its name
            if "untitled" in lower_name:
                print(f"Removing file: {file_path}")
                os.remove(file_path)
                continue

            # If it's a .log file
            if lower_name.endswith(".log"):
                print(f"Removing log file: {file_path}")
                os.remove(file_path)
                continue
            
            # If file name starts with "z"
            if lower_name.startswith('z'):
                print(f"Removing file: {file_path}")
                os.remove(file_path)

if __name__ == "__main__":
    # Check if a folder path was provided via command line
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "qaz"  # Default to 'qaz' if no argument is passed

    print(f"Sweeping away unwanted items in '{base_dir}'...")
    sweep_away(base_dir)
    print("Sweep complete!")
