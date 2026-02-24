# excel_pdf_S3
# uses an s3 file for tracking processing
# fast!!


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import boto3
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize S3 client
s3 = boto3.client("s3")

# Define S3 processing list path
PROCESSING_FILE_KEY = "development/processing/file_list.csv"

def fetch_file_list_from_s3(bucket_name, src_folder):
    """
    Fetch the list of Excel files in S3 and save it as `file_list.csv` in S3.
    """
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=src_folder)

    if "Contents" not in response:
        print("No files found in source folder.")
        return

    file_list = [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".xlsx")]

    # Create a DataFrame and set all files as unprocessed
    df = pd.DataFrame({"file_key": file_list, "processed": False})

    # Save file list to S3
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    s3.put_object(Bucket=bucket_name, Key=PROCESSING_FILE_KEY, Body=csv_buffer.getvalue(), ContentType="text/csv")
    print(f"Saved file list to s3://{bucket_name}/{PROCESSING_FILE_KEY}. Edit the file if needed.")

def process_file(bucket_name, file_key, dst_folder):
    """
    Processes a single Excel file, converts sheets to PDFs, and uploads them to S3.
    """
    try:
        print(f"Processing: {file_key}")

        # Download Excel file from S3
        excel_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        excel_bytes = BytesIO(excel_obj["Body"].read())

        # Extract base name and shorten it
        base_name = os.path.basename(file_key).replace(".xlsx", "")
        name_segments = base_name.split("_")
        short_base_name = "_".join(name_segments[:5])  # Keep first 5 segments

        # Read the Excel file
        xls = pd.ExcelFile(excel_bytes)

        for sheet_name in xls.sheet_names:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)

            # Generate in-memory PDF
            pdf_bytes = BytesIO()
            with PdfPages(pdf_bytes) as pdf:
                fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
                ax.axis("tight")
                ax.axis("off")

                # Create table with row/column labels
                table = ax.table(cellText=df_sheet.values, 
                                 colLabels=df_sheet.columns, 
                                 rowLabels=df_sheet.index, 
                                 cellLoc="center", 
                                 loc="center")

                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.auto_set_column_width([i for i in range(len(df_sheet.columns))])  # Adjust width

                # Add a title to indicate sheet name
                plt.title(f"Sheet: {sheet_name}", fontsize=10)

                # Save figure to PDF buffer
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # Save PDF to S3
            pdf_bytes.seek(0)  # Reset buffer position
            pdf_filename = f"{short_base_name}_{sheet_name}.pdf"
            pdf_s3_key = f"{dst_folder}/{pdf_filename}"

            s3.put_object(Bucket=bucket_name, Key=pdf_s3_key, Body=pdf_bytes.getvalue(), ContentType="application/pdf")

            print(f"Uploaded: s3://{bucket_name}/{pdf_s3_key}")

        return file_key  # Return the processed file key for tracking
    except Exception as e:
        print(f"Error processing {file_key}: {e}")
        return None

def process_files_from_s3_list(bucket_name, dst_folder, max_threads=4):
    """
    Reads `file_list.csv` from S3, processes unprocessed files concurrently, and updates the list back to S3.
    """
    try:
        # Fetch the existing processing file from S3
        obj = s3.get_object(Bucket=bucket_name, Key=PROCESSING_FILE_KEY)
        df = pd.read_csv(BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        print(f"Processing file not found: s3://{bucket_name}/{PROCESSING_FILE_KEY}")
        return

    # Filter unprocessed files
    unprocessed_files = df[df["processed"] == False]["file_key"].tolist()

    if not unprocessed_files:
        print("All files are already processed.")
        return

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_threads) as executor:
        future_to_file = {executor.submit(process_file, bucket_name, file, dst_folder): file for file in unprocessed_files}

        for future in as_completed(future_to_file):
            processed_file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    df.loc[df["file_key"] == result, "processed"] = True
            except Exception as e:
                print(f"Error processing file {processed_file}: {e}")

    # Save the updated processing list back to S3
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    s3.put_object(Bucket=bucket_name, Key=PROCESSING_FILE_KEY, Body=csv_buffer.getvalue(), ContentType="text/csv")

    print("Processing completed. Updated file list saved to S3.")

# Example usage:
# 1. Fetch and save file list to S3 (Run this first)
fetch_file_list_from_s3("your-bucket-name", "source_folder")

# 2. Process only unprocessed files concurrently (up to 4 files at a time)
process_files_from_s3_list("your-bucket-name", "dst_folder", max_threads=4)
