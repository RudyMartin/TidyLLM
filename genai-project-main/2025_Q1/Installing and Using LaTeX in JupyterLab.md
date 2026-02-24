# Installing and Using LaTeX in JupyterLab

## **Step 1: Install LaTeX**

### **Linux (Ubuntu/Debian)**
For AWS Sagemaker : Run the following commands  in new terminal  window:
```bash
sudo apt update
sudo apt install texlive texlive-latex-extra texlive-fonts-recommended
```

### **MacOS (Homebrew)**
```bash
brew install mactex
```
After installation, restart your terminal and run:
```bash
sudo tlmgr update --self && sudo tlmgr update --all
```

### **Windows**
1. Download and install [MikTeX](https://miktex.org/download).
2. During setup, select **"Install missing packages on the fly"**.
3. After installation, open `cmd` or PowerShell and run:
   ```powershell
   miktex-console
   ```
   and install additional packages if needed.

---

## **Step 2: Verify `pdflatex` Installation**

Again in same terminal window : Check if `pdflatex` is installed by running:
```bash
pdflatex --version
```
If it is not found, install it using:
```bash
sudo apt install texlive-latex-base
```

---

## **Step 3: Use LaTeX in Jupyter Notebook**

Now create a `.tex` file inside Jupyter and compile it to a PDF.

Open either a new notebook or python file and run this example.

### **Example: Generate a LaTeX Document in Jupyter**
```python
latex_content = r"""
\documentclass{article}
\usepackage{graphicx}
\begin{document}
\title{Hello, LaTeX in Jupyter!}
\author{Your Name}
\date{\today}
\maketitle

This is an example of LaTeX being used in Jupyter Notebook!

\end{document}
"""

tex_file = "example.tex"

with open(tex_file, "w") as f:
    f.write(latex_content)

print(f"LaTeX file created: {tex_file}")
```

---

## **Step 4: Compile LaTeX to PDF in Jupyter**

Use `pdflatex` to compile the `.tex` file into a PDF:

Paste the following code into notebook cell or python file.

```python
import subprocess

pdf_file = "example.pdf"

try:
    result = subprocess.run(["pdflatex", tex_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print(f"PDF successfully generated: {pdf_file}")
    else:
        print("Error in LaTeX compilation:", result.stderr)

except Exception as e:
    print("Exception occurred:", str(e))
```

---

## **Step 5: Display the Generated PDF in Jupyter**

To display the generated PDF inside Jupyter Notebook:

```python
from IPython.display import display, FileLink

# Display a link to the generated PDF
display(FileLink(pdf_file))
```

---

## **Final Notes**
- Make sure `pdflatex` is installed (`pdflatex --version`).
- If you prefer a modern LaTeX compiler, install `tectonic` using:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://tectonic-typesetting.github.io/install.sh | sh
  ```
- If using Windows, restart your terminal after installing MikTeX.

Once these steps are complete, you can smoothly generate PDFs from LaTeX inside **JupyterLab**! 🚀

