# **Testing Three Models and Three DSPy-Based Workflow Options**

## **Overview**
This document outlines the testing process for three AI models within three different workflow options using DSPy. The goal is to progressively enhance report automation by leveraging structured data processing, vector search, and multi-agent AI architectures.

## **3 Models to Test**
| **Model** | **ID** | **Purpose Notes** |
|-----------|------------------------------|-----------------------------------|
| amazon.titan-embed-text-v2:0 | Text Embedding | Ideal for document/vector search. |
| anthropic.claude-3-sonnet-20240229-v1:0 | General LLM | Best balance of speed vs capability for text tasks. |
| meta.llama3-70b-instruct-v1:0 | Large-Scale AI | Very powerful for reasoning-heavy tasks. |

The following workflow options incrementally introduce complexity, beginning with FAISS-based retrieval, followed by PGVector-powered retrieval, and finally a multi-agent PGVector + FAISS partitioned system.

---

## **🚀 Option 1: Simple Workflow (Universal FAISS Index)**
### **Goal:** Enable **universal FAISS-based semantic retrieval** of relevant **project details** to better contextualize report findings.

### **🔹 Modules**
1. **Text & Embedding Module** → Extracts text and creates **vector embeddings** using **Titan, Cohere, or Anthropic**.
2. **FAISS Vector Store Module** → Stores embeddings of **sd_reports** and **sd_details** in a **single FAISS index**.
3. **DSPy Query Engine** → Uses **BootstrapFewShot** to retrieve the most relevant source details.
4. **Report Refinement Module** → Uses **DSPy Compiler** to improve the process of report validation and correction.

### **🔹 Workflow**
1. Extract text from PDFs.
2. Convert text to **embeddings** and store them in a **universal FAISS index**.
3. Query **FAISS** using vector search for **semantic similarity**.
4. Use **BootstrapFewShot** to validate and refine retrieved information.
5. Generate improved reports with DSPy-driven refinement.

✅ **Pros:** Quick to implement, FAISS is **fast and efficient** for similarity search.  
❌ **Cons:** FAISS is in-memory (not persistent), requiring reloading after restart.

---

## **🚀 Option 2: Intermediate Workflow (Persistent PGVector + DSPy Retrieval)**
### **Goal:** Improve **scalability and persistence** by using **PGVector** as a persistent vector store.

### **🔹 Modules**
1. **Text & Embedding Module** → Extracts text and generates embeddings (Titan, Cohere, or Anthropic).
2. **PGVector Store Module** → Stores embeddings **persistently** in PostgreSQL + PGVector.
3. **DSPy Query Engine** → Uses **KNN Few-Shot** to retrieve the most relevant source details.
4. **Report Refinement Module** → Uses **MIPRO** to optimize report improvements.

### **🔹 Workflow**
1. Extract text from PDFs.
2. Convert text to **embeddings** and store them in **PGVector**.
3. Query **PGVector** for **persistent semantic retrieval**.
4. Use **KNN Few-Shot** to refine document retrieval dynamically.
5. Use **MIPRO** to improve report sections using retrieved context.
6. Generate improved reports with DSPy-driven refinements.

✅ **Pros:** Enables **scalable and persistent** semantic search.  
❌ **Cons:** Requires additional setup for **PostgreSQL + PGVector**.

---

## **🚀 Option 3: Advanced Workflow (Partitioned FAISS + PGVector with Multi-Agent DSPy)**
### **Goal:** Build a **highly scalable, multi-agent DSPy pipeline** that optimally balances FAISS and PGVector retrieval for **contextual AI-driven report improvements**.

### **🔹 Modules**
1. **Text & Embedding Module** → Extracts text and generates embeddings using **Titan, Cohere, or Anthropic**.
2. **Partitioned FAISS + PGVector Store**  
   - **FAISS Partitioned Index** → Handles **short-term embeddings** (recent reports).  
   - **PGVector Persistent Store** → Stores **long-term knowledge** for scalable retrieval.
3. **Multi-Agent DSPy Query System**  
   - **Retriever Agent**: Uses **FAISS + PGVector hybrid search** to fetch context.
   - **Evaluator Agent**: Uses **COPRO** to compare extracted insights vs. project details.
   - **Rewriter Agent**: Uses **MIPROv2** to improve reports dynamically.
4. **Final Report Generator** → Uses DSPy **BootstrapFewShot** + reinforcement-based improvements.

### **🔹 Workflow**
1. Extract text from **sd_reports** and **sd_details**.
2. Convert text to **vector embeddings** and store in **FAISS (short-term) and PGVector (long-term)**.
3. **Retriever Agent** queries **partitioned FAISS + PGVector** for the most relevant project details.
4. **Evaluator Agent** runs **contrastive analysis** on extracted insights vs. project details.
5. **Rewriter Agent** refines **sd_reports** using **MIPROv2**.
6. **Final Report Generator** produces an improved **version of reports** with multi-agent DSPy improvements.

✅ **Pros:** Highly **scalable**, **optimized retrieval**, **multi-agent intelligence**.  
❌ **Cons:** **More complex** setup, requires multi-layer **embedding management** and DSPy orchestration.

---

## **📌 Final Decision: Which One to Start With?**
| **Option** | **Best If You Want...** |
|-----------|--------------------|
| **Simple (FAISS + DSPy Few-Shot)** | A **quick baseline** using a universal FAISS index. |
| **Intermediate (PGVector + DSPy Retrieval)** | **Persistent vector storage** for long-term scalability. |
| **Advanced (Partitioned FAISS + PGVector + Multi-Agent DSPy)** | **Scalable, multi-agent AI-reporting system**. |

---

### **Next Steps**
- **Start with Option 1 (FAISS + DSPy Few-Shot)** → Validate FAISS retrieval and DSPy pipeline.
- **Then Test PGVector vs. FAISS** → Evaluate performance for long-term storage.
- **If needed, move to Multi-Agent DSPy** for fully automated AI-driven report optimization.

