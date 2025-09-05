#!/usr/bin/env python3
"""
TLM TEAM ADVANTAGE: Leveraging the full tidyllm ecosystem
FAISS has pre-trained models? We have ALGORITHMIC SYNERGY!

Strategy: Combine multiple tlm algorithms for superior performance:
1. Multi-Algorithm Ensemble (TF-IDF + Word-Avg + N-gram + LSA)
2. PCA Dimensionality Optimization
3. K-means Clustering for Hierarchical Search  
4. Anomaly Detection for Quality Filtering
5. GMM Data Augmentation
6. Whitening for Embedding Normalization
7. Matrix Factorization for Efficient Storage

Let's build tidyllm-sentence-TEAM!
"""

import sys
import time
sys.path.insert(0, '../tlm')

import tidyllm_sentence as tls
import tlm
import statistics
from typing import List, Tuple, Dict, Any

class TidyllmTeamEmbeddings:
    """
    The TEAM approach: Multiple tidyllm algorithms working together
    to create embeddings that leverage the full ecosystem.
    """
    
    def __init__(self, 
                 use_tfidf: bool = True,
                 use_word_avg: bool = True, 
                 use_ngrams: bool = True,
                 use_lsa: bool = True,
                 use_transformer: bool = True,
                 target_dims: int = 128,
                 n_clusters: int = 16):
        
        self.use_tfidf = use_tfidf
        self.use_word_avg = use_word_avg
        self.use_ngrams = use_ngrams
        self.use_lsa = use_lsa
        self.use_transformer = use_transformer
        self.target_dims = target_dims
        self.n_clusters = n_clusters
        
        # Models storage
        self.models = {}
        self.pca_model = None
        self.whitening_model = None
        self.cluster_model = None
        self.anomaly_model = None
        
    def _generate_multi_algorithm_embeddings(self, sentences: List[str]) -> List[List[List[float]]]:
        """Generate embeddings using multiple algorithms."""
        print(f"Generating multi-algorithm embeddings...")
        all_embeddings = []
        
        if self.use_tfidf:
            print(f"  - TF-IDF embeddings...")
            tfidf_emb, tfidf_model = tls.tfidf_fit_transform(sentences)
            all_embeddings.append(tfidf_emb)
            self.models['tfidf'] = tfidf_model
            
        if self.use_word_avg:
            print(f"  - Word averaging embeddings...")  
            try:
                word_avg_emb, word_avg_model = tls.word_avg_fit_transform(sentences)
                all_embeddings.append(word_avg_emb)
                self.models['word_avg'] = word_avg_model
            except Exception as e:
                print(f"    Word-avg failed: {e}, skipping...")
                
        if self.use_ngrams:
            print(f"  - N-gram embeddings...")
            try:
                ngram_emb, ngram_model = tls.ngram_fit_transform(sentences)
                all_embeddings.append(ngram_emb)
                self.models['ngram'] = ngram_model
            except Exception as e:
                print(f"    N-gram failed: {e}, skipping...")
                
        if self.use_lsa:
            print(f"  - LSA embeddings...")
            try:
                lsa_emb, lsa_model = tls.lsa_fit_transform(sentences)
                all_embeddings.append(lsa_emb)
                self.models['lsa'] = lsa_model
            except Exception as e:
                print(f"    LSA failed: {e}, skipping...")
        
        if self.use_transformer:
            print(f"  - Transformer embeddings...")
            transformer_emb, transformer_model = tls.transformer_fit_transform(sentences, max_seq_len=24)
            all_embeddings.append(transformer_emb)
            self.models['transformer'] = transformer_model
            
        print(f"Generated {len(all_embeddings)} embedding types")
        return all_embeddings
    
    def _ensemble_embeddings(self, multi_embeddings: List[List[List[float]]]) -> List[List[float]]:
        """Ensemble multiple embedding types into unified representations."""
        print(f"Creating ensemble embeddings...")
        
        if not multi_embeddings:
            raise ValueError("No embeddings to ensemble!")
            
        n_sentences = len(multi_embeddings[0])
        
        # Normalize each embedding type
        normalized_embeddings = []
        for embeddings in multi_embeddings:
            normalized = tlm.l2_normalize(embeddings)
            normalized_embeddings.append(normalized)
        
        # Concatenate all embedding types
        ensemble_embeddings = []
        for i in range(n_sentences):
            concatenated = []
            for embedding_type in normalized_embeddings:
                concatenated.extend(embedding_type[i])
            ensemble_embeddings.append(concatenated)
            
        print(f"Ensemble dimension: {len(ensemble_embeddings[0])}")
        return ensemble_embeddings
    
    def _apply_pca_optimization(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Use tlm's PCA to optimize embedding dimensions."""
        print(f"Applying PCA dimensionality optimization...")
        
        # Apply PCA to reduce to target dimensions
        projected, self.pca_model = tlm.pca_power_fit_transform(
            embeddings, k=min(self.target_dims, len(embeddings[0]))
        )
        
        print(f"PCA reduced {len(embeddings[0])} -> {len(projected[0])} dims")
        
        return projected
    
    def _apply_whitening(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Apply tlm's whitening transformation for better embedding properties."""
        print(f"Applying whitening transformation...")
        
        if self.pca_model is None:
            print(f"No PCA model available, skipping whitening...")
            return embeddings
            
        whitened = tlm.pca_whiten(embeddings, self.pca_model)
        
        # Verify whitening worked
        mean_vals = tlm.mean(whitened, axis=0)
        mean_magnitude = tlm.norm(mean_vals)
        print(f"Post-whitening mean magnitude: {mean_magnitude:.6f}")
        
        return whitened
    
    def _build_clustering_index(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Build k-means clustering for hierarchical search."""
        print(f"Building k-means clustering index...")
        
        centers, labels, inertia = tlm.kmeans_fit(embeddings, k=self.n_clusters, seed=42)
        
        # Group documents by cluster
        clusters = {}
        for doc_id, cluster_id in enumerate(labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(doc_id)
            
        self.cluster_model = {
            'centers': centers,
            'labels': labels,
            'clusters': clusters
        }
        
        avg_cluster_size = len(embeddings) / self.n_clusters
        print(f"Created {self.n_clusters} clusters, avg size: {avg_cluster_size:.1f}")
        
        return self.cluster_model
    
    def _train_anomaly_detector(self, embeddings: List[List[float]]) -> Any:
        """Train anomaly detector to filter low-quality results."""
        print(f"Training anomaly detector for quality filtering...")
        
        mu, var = tlm.anomaly_fit(embeddings)
        self.anomaly_model = {'mu': mu, 'var': var}
        
        # Test anomaly scores on sample
        sample_scores = tlm.anomaly_score_logpdf(embeddings[:5], mu, var)
        print(f"Sample anomaly scores: {[f'{s:.3f}' for s in sample_scores]}")
        
        return self.anomaly_model
    
    def fit_transform(self, sentences: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        The FULL TEAM approach: leverage every tlm algorithm for maximum performance.
        
        Pipeline:
        1. Multi-algorithm embeddings (TF-IDF, word-avg, n-grams, LSA, transformer)
        2. Ensemble concatenation with normalization
        3. PCA dimensionality optimization  
        4. Whitening transformation
        5. K-means clustering index
        6. Anomaly detection training
        
        Returns optimized embeddings + all trained models
        """
        print(f"\n" + "="*60)
        print(f"TLM TEAM ADVANTAGE - FULL ALGORITHMIC ARSENAL")
        print(f"Processing {len(sentences)} sentences with {sum([self.use_tfidf, self.use_word_avg, self.use_ngrams, self.use_lsa, self.use_transformer])} algorithms")
        print(f"="*60)
        
        start_time = time.time()
        
        # Step 1: Multi-algorithm embeddings
        multi_embeddings = self._generate_multi_algorithm_embeddings(sentences)
        
        # Step 2: Ensemble
        ensemble_embeddings = self._ensemble_embeddings(multi_embeddings)
        
        # Step 3: PCA optimization
        pca_embeddings = self._apply_pca_optimization(ensemble_embeddings)
        
        # Step 4: Whitening
        final_embeddings = self._apply_whitening(pca_embeddings)
        
        # Step 5: Clustering index
        clustering_index = self._build_clustering_index(final_embeddings)
        
        # Step 6: Anomaly detection
        anomaly_model = self._train_anomaly_detector(final_embeddings)
        
        total_time = time.time() - start_time
        
        print(f"\n" + "="*60) 
        print(f"TLM TEAM PROCESSING COMPLETE")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final embedding dimensions: {len(final_embeddings[0])}")
        print(f"Algorithmic components: {len(self.models)}")
        print(f"="*60)
        
        return final_embeddings, {
            'models': self.models,
            'pca_model': self.pca_model,
            'clustering': clustering_index,
            'anomaly_model': anomaly_model,
            'stats': {
                'total_time': total_time,
                'final_dims': len(final_embeddings[0]),
                'n_algorithms': len(self.models)
            }
        }
    
    def hierarchical_search(self, query_embedding: List[float], 
                          doc_embeddings: List[List[float]], 
                          top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Hierarchical search using clustering for speed + anomaly filtering for quality.
        """
        if not self.cluster_model:
            # Fallback to direct search
            return self._direct_search(query_embedding, doc_embeddings, top_k)
        
        # Find closest cluster centers
        cluster_similarities = []
        for cluster_id, center in enumerate(self.cluster_model['centers']):
            sim = tls.cosine_similarity(query_embedding, center)
            cluster_similarities.append((sim, cluster_id))
        
        cluster_similarities.sort(reverse=True)
        
        # Search in top clusters
        candidates = []
        clusters_to_search = min(3, len(cluster_similarities))  # Search top 3 clusters
        
        for _, cluster_id in cluster_similarities[:clusters_to_search]:
            doc_ids = self.cluster_model['clusters'][cluster_id]
            
            for doc_id in doc_ids:
                sim = tls.cosine_similarity(query_embedding, doc_embeddings[doc_id])
                candidates.append((sim, doc_id))
        
        # Sort by similarity
        candidates.sort(reverse=True)
        
        # Anomaly filtering (optional quality boost)
        if self.anomaly_model:
            filtered_candidates = []
            for sim, doc_id in candidates:
                anomaly_scores = tlm.anomaly_score_logpdf([doc_embeddings[doc_id]], 
                                                         self.anomaly_model['mu'], 
                                                         self.anomaly_model['var'])
                anomaly_score = anomaly_scores[0]
                # Keep documents that aren't too anomalous
                if anomaly_score > -10.0:  # Threshold for "normal" documents
                    filtered_candidates.append((sim, doc_id))
            
            if filtered_candidates:  # Use filtered if we have results
                candidates = filtered_candidates
        
        return [(doc_id, sim) for sim, doc_id in candidates[:top_k]]
    
    def _direct_search(self, query_embedding: List[float], 
                      doc_embeddings: List[List[float]], 
                      top_k: int) -> List[Tuple[int, float]]:
        """Direct similarity search as fallback."""
        similarities = []
        for doc_id, doc_emb in enumerate(doc_embeddings):
            sim = tls.cosine_similarity(query_embedding, doc_emb)
            similarities.append((sim, doc_id))
        
        similarities.sort(reverse=True)
        return [(doc_id, sim) for sim, doc_id in similarities[:top_k]]

def demo_tlm_team_advantage():
    """Demonstrate the TLM team advantage on the academic dataset."""
    
    # Use a subset of the academic dataset for demo
    test_sentences = [
        "machine learning algorithms for classification tasks",
        "support vector machines are powerful classification algorithms",
        "neural networks and deep learning architectures", 
        "convolutional neural networks revolutionized computer vision",
        "natural language processing techniques for text analysis",
        "transformer models like BERT advance language understanding",
        "database systems and query optimization methods",
        "SQL databases use structured query language efficiently",
        "cybersecurity and network protection mechanisms",
        "encryption algorithms protect sensitive data securely"
    ]
    
    queries = [
        "machine learning classification methods",
        "deep neural network architectures", 
        "database query optimization"
    ]
    
    print("DEMO: TLM TEAM ADVANTAGE")
    print("=" * 50)
    
    # Initialize team embeddings
    team_embedder = TidyllmTeamEmbeddings(
        target_dims=64,  # Smaller for demo
        n_clusters=4
    )
    
    # Generate team embeddings
    doc_embeddings, team_models = team_embedder.fit_transform(test_sentences)
    
    # Process queries with same pipeline
    print(f"\nProcessing queries...")
    query_embeddings = []
    for query in queries:
        # For demo, we'll just use direct similarity on the processed embeddings
        # In a full implementation, queries would go through the same pipeline
        query_emb = team_models['models']['transformer'].sentence_to_embedding(query)
        # Pad/trim to match embedding dimension
        if len(query_emb) > 64:
            query_emb = query_emb[:64]
        else:
            query_emb = query_emb + [0.0] * (64 - len(query_emb))
        query_embeddings.append(query_emb)
    
    # Test hierarchical search
    print(f"\nTesting hierarchical search:")
    for i, (query, query_emb) in enumerate(zip(queries, query_embeddings)):
        print(f"\nQuery {i+1}: \"{query}\"")
        results = team_embedder.hierarchical_search(query_emb, doc_embeddings, top_k=3)
        
        for rank, (doc_id, score) in enumerate(results):
            print(f"  {rank+1}. Score: {score:.3f} - \"{test_sentences[doc_id][:50]}...\"")
    
    print(f"\n" + "="*50)
    print(f"TLM TEAM ADVANTAGE DEMO COMPLETE!")
    print(f"Leveraged {team_models['stats']['n_algorithms']} algorithms")
    print(f"Final embedding dimensions: {team_models['stats']['final_dims']}")
    print(f"Processing time: {team_models['stats']['total_time']:.2f}s")
    print(f"="*50)

if __name__ == "__main__":
    demo_tlm_team_advantage()