# 🚀 st_app_keydrivers.py - Main Streamlit App for Key Drivers Report Processing

from configuration import *  # ✅ Universal Import
from instructions_helper import load_instructions
from embedding_helper import AmazonEmbeddingVectorizer
from key_helpers import list_objects_s3, save_page_chunks_as_json_s3, smart_lexical_chunking

# **🔹 Load Instructions**
instructions_data = load_instructions()

# **🔹 Initialize FAISS Index**
dimension = MODEL_OPTIONS[CONFIG["embedding_model"]]["dimensions"]
index = faiss.IndexFlatL2(dimension)

# **🔹 User Session Tracking**
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{datetime.datetime.utcnow().timestamp()}"

# **🔹 Sidebar: Model Selection & Instructions**
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose an Embedding Model:", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[model_choice]["id"]

with st.sidebar.expander("📖 Instructions"):
    st.write(instructions_data["message"])
    for section in instructions_data.get("sections", []):
        st.subheader(section["title"])
        st.write(section["content"])

# **🔹 Initialize Vectorizer**
vectorizer = AmazonEmbeddingVectorizer(model_id=selected_model)

# **🔹 Upload & Process PDF**
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    st.success("PDF uploaded successfully! Extracting text...")

    json_container = st.empty()
    final_json = {
        "metadata": {"source": "generated_text", "timestamp": datetime.datetime.utcnow().isoformat() + "Z"},
        "content": [],
        "embeddings": {}
    }

    for i in range(1, 11):
        new_page = {"page_number": i, "text": f"This is extracted text from page {i}."}
        final_json["content"].append(new_page)
        json_container.json(final_json)
        time.sleep(0.5)

    st.download_button("Download Extracted JSON", json.dumps(final_json, indent=4), file_name="extracted_data.json")

# **🔹 Query Processing**
query_text = st.text_input("Enter a query:")
if st.button("Search"):
    query_embedding = np.array([vectorizer.get_embedding(query_text)], dtype=np.float32)
    _, indices = index.search(query_embedding, 3)
    retrieved_docs = [f"Document {i+1}" for i in indices[0]]

    st.subheader("Retrieved Documents:")
    st.write(retrieved_docs)

# **🔹 DSPy Retraining**
if st.button("Retrain DSPy Using Feedback"):
    model_version = f"{selected_model}_v{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    retrain_log = pd.DataFrame([[datetime.datetime.utcnow().isoformat(), model_version, len(retrieved_docs)]],
                               columns=["timestamp", "model_version", "trained_on_feedback_count"])
    retrain_log.to_csv("model_versions.csv", mode="a", header=False, index=False)
    st.success(f"DSPy retrained! Version: {model_version}")

# **🔹 Show Model Versions**
if st.button("Show Model Versions"):
    try:
        st.dataframe(pd.read_csv("model_versions.csv"))
    except FileNotFoundError:
        st.warning("No retraining logs found.")
