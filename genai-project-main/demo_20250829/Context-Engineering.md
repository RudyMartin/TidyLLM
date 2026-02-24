

# Managing Context Collapse

| **Strategy Name**                       | **Description**                                                                          |
| --------------------------------------- | ---------------------------------------------------------------------------------------- |
| Active RAG                              | Dynamically selects retrieval strategies during generation based on context.             |
| Algorithm of Thoughts (AoT)             | Uses structured reasoning algorithms (like search/planning trees) to guide LLM thinking. |
| A-Mem                                   | Augmented memory mechanism for persistent or episodic recall.                            |
| Auto RAG                                | Automates the selection and tuning of retrieval + generation strategies.                 |
| BM25 + Dense Hybrid Search              | Combines sparse (BM25) and dense vector search for more robust retrieval.                |
| Chain-of-Thought Guided Retrieval       | Uses intermediate reasoning steps (CoT) to refine retrieval queries.                     |
| ColBERT                                 | Late-interaction dense retriever balancing efficiency and precision.                     |
| Corrective RAG (CRAG)                   | Applies corrective feedback loops to refine or fix retrieval outputs.                    |
| Episodic Memory Buffer                  | Stores contextual episodes for reuse across interactions.                                |
| Fusion-in-Decoder (FiD)                 | Fuses multiple retrieved documents directly in the decoder during generation.            |
| Graph RAG                               | Retrieval augmented with knowledge graphs to capture entity relationships.               |
| Hybrid RAG                              | Combines multiple retrieval approaches (dense, sparse, symbolic, etc.).                  |
| HyDE (Hypothetical Document Embeddings) | Generates hypothetical documents to improve retrieval relevance.                         |
| Knowledge-Graph RAG                     | Uses structured graph knowledge bases as retrieval sources.                              |
| Long RAG                                | Optimized for handling long documents and context windows.                               |
| Map-Reduce Summarization                | Splits long inputs into smaller parts, summarizes, then merges them.                     |
| Memo RAG                                | Adds memoization to store/reuse retrieval results across turns.                          |
| Multi-Hop RAG                           | Supports multi-step reasoning and retrieval across chained queries.                      |
| Query Decomposition                     | Breaks complex queries into simpler sub-queries for better retrieval.                    |
| Query Rewriting                         | Reformulates queries to improve retrieval precision and recall.                          |
| R3 Mem                                  | Recursive retrieval + reasoning + memory model for persistent context.                   |
| RAG Fusion                              | Combines results from multiple retrievers and reranks them for generation.               |
| REPLUG                                  | Retrieval-augmented plug-in for LLMs without fine-tuning.                                |
| Rolling Buffer Summaries                | Maintains rolling context by continuously summarizing past dialogue.                     |
| Self-RAG (Self-Reflective RAG)          | Model reflects on its own responses and retrieves corrections if needed.                 |
| Sliding Window Summarization            | Moves a window across long context to incrementally summarize.                           |
| Time-Weighted Vector Store Retriever    | Prioritizes recent or temporally weighted documents in retrieval.                        |
| Tree-Summarize                          | Uses tree-structured summaries (merge from leaves to root).                              |
| Tree of Thoughts Summarization          | Applies tree search to explore multiple summarization paths.                             |

