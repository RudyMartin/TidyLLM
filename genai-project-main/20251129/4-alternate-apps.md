 Bash command

   cat << 'EOF'
   # IR RANKING EXPLAINED

   ## What is Information Retrieval Ranking?

   IR Ranking is the problem: "Given a query and a collection of documents,
   rank the documents by relevance to the query."

   ### Real-World Examples
   - Google search: You type "best pizza near me"
     - Google has 1 billion web pages
     - Must rank them by relevance

   - Recommendation systems: Netflix, Amazon
     - You have user preferences
     - Rank products by relevance

   - Legal document discovery: Find relevant court cases
     - Query: "liability in car accidents"
     - Rank 1 million cases by relevance

   ## How YRSN Would Be Applied

   ### Current IR Approaches
   1. **TF-IDF**: Word frequency scores
   2. **BM25**: Statistical ranking
   3. **Learning to Rank**: Neural networks learn ranking function

   ### YRSN Approach
   Decompose ranking signals into Y=R+S+N:

   - **R (Relevant)**: Actual topicality
     - Query words match document
     - Semantic similarity (word embeddings)
     - Authority signals (citations, page rank)

   - **S (Superfluous)**: Systematic biases
     - Document length bias (long docs ranked higher)
     - Popularity bias (popular docs ranked higher)
     - Position bias in training data

   - **N (Noise)**: Harmful interference
     - Outdated information
     - Spam/misleading content
     - Query ambiguity

   ### YRSN Benefits for IR
   1. **Collapse Detection**: Detect when ranker learns bias instead of relevance
      - POISONING: Too much spam in results
      - DISTRACTION: Ranking by length instead of relevance
      - CONFUSION: Mixed signals (spam + genuine relevance)

   2. **Robust PCA**: Decompose relevance signals
      - L (Low-rank): Genuine topical relevance
      - S (Sparse): Individual document signals (outliers)
      - N (Noise): Random ranking fluctuations

   3. **Self-Calibration**: Automatically adjust ranking weights
      - Reduce weight on popularity (S)
      - Boost weight on topicality (R)
      - Filter spam (N)

   ## Benchmark: benchmark_ir_ranking_collapse_detection.py

   ### Goal
   Show that YRSN collapse detection catches when rankers fail

   ### Setup
   1. Create query sets with different characteristics:
      - Normal queries (balanced documents)
      - Poisoned queries (spam-heavy dataset)
      - Distracted queries (document-length bias)
      - Confused queries (mixed types)

   2. Train neural ranker on each query set

   3. Measure: Does YRSN detect collapse?

   ### Expected Results
   - Normal: No collapse detected ✅
   - Poisoned: High N detected, Collapse.POISONING ✅
   - Distracted: High S detected, Collapse.DISTRACTION ✅
   - Confused: High N+S, Collapse.CONFUSION ✅

   ### Why This Matters
   Traditional IR metrics (NDCG, MAP) only measure accuracy.
   YRSN shows WHY the ranker fails (poisoning vs distraction vs confusion).

   ---

   ## Benchmark: benchmark_ir_ranking_quality.py

   ### Goal
   Show that YRSN decomposition improves ranking quality

   ### Setup
   Dataset: TREC (standard IR benchmark with relevance labels)

   1. Train baseline ranker (neural net)
   2. Train YRSN-guided ranker:
      - Decompose training signals into R/S/N
      - Boost R component (genuine relevance)
      - Downweight S component (biases)
      - Filter N component (noise)

   3. Evaluate both on test set

   ### Metrics
   - Traditional: NDCG@10, MAP, MRR
   - YRSN: R/S/N decomposition, collapse type

   ### Expected Results
   - YRSN ranker: 2-5% NDCG improvement
   - Less sensitive to input noise
   - Better handling of domain shift (test differs from training)

   ---

   ## Why Add IR Ranking as Application?

   ### Current State
   - Sudoku: Symbolic reasoning (constraint satisfaction)
   - (Proposed) Time Series: Continuous prediction
   - (Proposed) IR Ranking: Text understanding (ranking)

   ### Different Data Type
   - Sudoku: Discrete board + rules
   - Time Series: 1D continuous signal
   - IR: Text (queries + documents)

   ### Different Algorithm Mix
   - Sudoku: Uses CTM, NLM, Certainty Loss, Bit-Slicing, Gradient
   - Time Series: Uses ESN, FIF, Signal Separation, Gradient
   - IR Ranking: Uses Robust PCA, Collapse Detection, Phase Transitions, Gradient

   ### Proof of YRSN Generality
   If YRSN works on:
   1. Sudoku (constraints)
   2. Time Series (signals)
   3. IR Ranking (text ranking)

   Then YRSN is truly general, not domain-specific!

   ---

   ## Implementation Effort

   ### IR Application (250 lines total)
   ```
   src/yrsn_context/applications/ir_ranking/
     encoder.py (80 lines)
       - QueryDocumentEncoder
       - Maps query + document -> YRSN R/S/N components

     ranker.py (100 lines)
       - YRSNGuidedRanker
       - Learns ranking function with YRSN guidance
       - Collapsedetection integration

     solver.py (40 lines)
       - IRRanker wrapper for consistency

   examples/
     benchmark_ir_ranking_collapse_detection.py (110 lines)
     benchmark_ir_ranking_quality.py (120 lines)
   ```

   ### Time Series vs IR Ranking

   **Time Series** (370 lines):
   - Natural fit for ESN/FIF algorithms
   - Straightforward signal decomposition
   - Standard datasets (UCR)
   - **Recommendation: DO THIS FIRST**

   **IR Ranking** (250 lines):
   - Validates decomposition algorithms (Robust PCA, Collapse Detection)
   - Shows YRSN on text data
   - Less "ready to go" (requires TREC dataset, more setup)
   - **Recommendation: DO THIS SECOND (if time allows)**

   ---

   ## Decision Matrix

   | Aspect | Time Series | IR Ranking |
   |--------|-------------|------------|
   | Effort | ~370 lines | ~250 lines |
   | Complexity | Medium | Hard (text processing) |
   | Algorithms validated | 5 (ESN, FIF, Signal Sep) | 3 (PCA, Collapse, Phase Trans) |
   | Domain fit | Natural | Good |
   | Dataset availability | Easy (UCI, standard) | Medium (TREC, requires setup) |
   | Expected impact | High (ESN validation) | High (decomposition validation) |
   | Time to implement | 1-2 weeks | 2-3 weeks |

   **Recommendation**:
   1. Time Series first (faster, validates critical algorithms)
   2. IR Ranking second (deeper validation of decomposition theory)
   3. Together: Complete validation of YRSN generality

   EOF

   Explain IR Ranking application

 Do you want to proceed?
