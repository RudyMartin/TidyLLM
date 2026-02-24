### **📌 Running Streamlit on AWS SageMaker JupyterLab**
Since you will run the Streamlit app **inside AWS SageMaker JupyterLab**, we need to ensure that:
✅ **Streamlit runs inside JupyterLab without blocking execution.**  
✅ **The app is accessible within the Jupyter environment.**  
✅ **AWS credentials are properly set up for S3 access.**  

---

## **🚀 1️⃣ Install Dependencies**
First, install Streamlit and the required libraries in the **Jupyter Notebook terminal** (not in a cell):

```sh
pip install --upgrade streamlit boto3
```

---

## **🚀 2️⃣ Start Streamlit in SageMaker**
Run this command **inside a Jupyter Notebook cell** to launch the Streamlit app:

```python
!streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
```

✅ `--server.port 8501` → Runs Streamlit on port **8501** (default).  
✅ `--server.enableCORS false` → **Disables CORS**, allowing access inside Jupyter.  
✅ `--server.enableXsrfProtection false` → **Disables CSRF protection**, avoiding conflicts inside AWS SageMaker.  

---

## **🚀 3️⃣ Accessing Streamlit in SageMaker**
- **Once you run the command,** it will show a URL like:
  ```
  You can now view your Streamlit app in your browser.

  Network URL: http://localhost:8501
  ```
- To **open it in SageMaker**, replace `localhost` with your **Jupyter server address**:
  ```
  http://<your-sagemaker-notebook-url>:8501
  ```
- Alternatively, click **"Open in a new tab"** if SageMaker provides a link.

---

## **🚀 4️⃣ AWS S3 Access for Streamlit**
Since your JSON files are on **S3**, ensure **AWS credentials** are set up. In your Jupyter Notebook terminal, run:

```sh
aws configure
```
Then enter:
- **AWS Access Key ID**
- **AWS Secret Access Key**
- **Default region name** (e.g., `us-east-1`)

📌 **SageMaker already has an IAM role.** If your instance has **S3 read/write permissions**, `aws configure` may not be needed.

---

## **🚀 5️⃣ Modifying Streamlit to Auto-Detect SageMaker**
To **automatically detect SageMaker and set the correct URL**, modify **`app.py`**:

```python
import streamlit as st
import os

# Check if running inside AWS SageMaker
if "SAGEMAKER_ROLE" in os.environ:
    st.sidebar.info("✅ Running inside AWS SageMaker")

st.title("📄 S3 Batch Embedding Processor")
st.markdown("Select an S3 folder and embedding model to process all JSON files in that folder.")
```

✅ **Detects SageMaker** and shows a sidebar message.  
✅ **Helps users confirm they're in the right environment.**  

---

## **🚀 6️⃣ Running the Full Workflow**
### **Inside Jupyter Notebook**
1️⃣ Open a new **terminal in JupyterLab**  
2️⃣ Run:
```sh
streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
```
3️⃣ Open **`http://<your-sagemaker-url>:8501`** in a browser.

---

## **✅ Summary**
✔ **Runs Streamlit inside AWS SageMaker JupyterLab.**  
✔ **Provides a working URL to access the app.**  
✔ **Handles AWS S3 credentials for embedding processing.**  
✔ **Detects if running inside SageMaker and updates UI.**  

Would you like to **add auto-reloading** so Streamlit updates instantly when you edit the code? 🚀😊
