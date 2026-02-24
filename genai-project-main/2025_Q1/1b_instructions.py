{
    "message": "Welcome to the Key Drivers Report Processing App!",
    "sections": [
        {
            "title": "Getting Started",
            "content": "Upload your PDF file in the 'Upload & Process' tab. The system will extract and analyze the document."
        },
        {
            "title": "Searching Queries",
            "content": "Enter a query in the 'Query Search' tab to retrieve relevant documents using AI embeddings. The app uses FAISS indexing and DSPy query optimization to find the most relevant documents."
        },
        {
            "title": "Model Selection & Reverting",
            "content": "You can select your preferred embedding model (Titan, Cohere, Claude) and revert to older trained versions from the sidebar."
        },
        {
            "title": "Tracking Analytics",
            "content": "The system tracks latency, error rates, and query success metrics. View real-time data in the sidebar charts."
        },
        {
            "title": "Managing FAISS Indexes",
            "content": "Each embedding model and dimension has its own FAISS index stored on S3. You can delete an outdated FAISS index using the 'Delete FAISS Index' button in the sidebar. This allows you to reset indexing for a specific model and reprocess documents cleanly."
        },
        {
            "title": "Key Enhancements",
            "content": [
                "✅ Separate FAISS Index for Each Embedding Model & Dimension – Stored and retrieved independently.",
                "✅ Automatic FAISS Reindexing – Ensures FAISS stays updated for every model.",
                "✅ DSPy Optimized Query Expansion – Enhances FAISS search accuracy.",
                "✅ FAISS Persistence to S3 – Ensures data isn't lost across sessions."
            ]
        },
        {
            "title": "Deployment Instructions",
            "content": "1. Clone the repository: `git clone https://github.com/yourrepo.git`\n"
                      "2. Install dependencies: `pip install -r requirements.txt`\n"
                      "3. Run the Streamlit app: `streamlit run st_app_keydrivers.py`\n"
                      "4. Ensure AWS credentials are set up correctly for Bedrock access."
        },
        {
            "title": "Troubleshooting S3 Access",
            "content": "Ensure your AWS credentials and permissions are correctly configured. If you face access issues, update your credentials and restart the application."
        }
    ]
}
