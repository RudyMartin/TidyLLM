## cconvert  local  files to  pdf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def excel_to_individual_pdfs(excel_file, output_folder):
    # Extract base name and remove extension
    base_name = os.path.basename(excel_file).replace(".xlsx", "")

    # Split by underscore and keep only the first five segments
    name_segments = base_name.split("_")
    short_base_name = "_".join(name_segments[:5])  # Keep only the first 5 parts

    # Load the Excel file
    xls = pd.ExcelFile(excel_file)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for sheet_name in xls.sheet_names:  # Loop through all sheets
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Format the new file name
        pdf_filename = f"{short_base_name}_{sheet_name}.pdf"
        pdf_file = os.path.join(output_folder, pdf_filename)

        # Create and save the sheet as a separate PDF
        with PdfPages(pdf_file) as pdf:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
            ax.axis('tight')
            ax.axis('off')

            # Create table with row/column labels
            table = ax.table(cellText=df.values, 
                             colLabels=df.columns, 
                             rowLabels=df.index, 
                             cellLoc='center', 
                             loc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.auto_set_column_width([i for i in range(len(df.columns))])  # Adjust width

            # Add a title to indicate sheet name
            plt.title(f"Sheet: {sheet_name}", fontsize=10)

            # Save the figure to a PDF file
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        print(f"Saved: {pdf_file}")  # Optional: Print progress

# Example usage
excel_to_individual_pdfs("mrc_issue_attribution_report_2025-02-08_asdasd_12312.xlsx", "output_pdfs")
