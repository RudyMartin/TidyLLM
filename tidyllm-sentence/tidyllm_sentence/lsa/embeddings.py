import math
from typing import List, Dict, Tuple, Optional
from ..utils.tokenize import word_tokenize

Vector = List[float]
Matrix = List[Vector]

__all__ = ['fit', 'transform', 'fit_transform', 'LSAModel']

class LSAModel:
    """LSA (Latent Semantic Analysis) sentence embedding model."""
    
    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self.vocabulary: Dict[str, int] = {}
        self.tfidf_model = None
        self.components: Matrix = []  # SVD components
        self.explained_variance: Vector = []
        self.vocab_size = 0
    
    def _build_tfidf_matrix(self, sentences: List[str]) -> Matrix:
        """Build TF-IDF document-term matrix."""
        # Build vocabulary
        vocab_set = set()
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            vocab_set.update(tokens)
        
        vocab_list = sorted(vocab_set)
        self.vocabulary = {word: i for i, word in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)
        
        # Compute IDF scores
        n_docs = len(sentences)
        doc_frequencies = [0] * self.vocab_size
        
        for sentence in sentences:
            tokens = set(word_tokenize(sentence))
            for token in tokens:
                if token in self.vocabulary:
                    doc_frequencies[self.vocabulary[token]] += 1
        
        idf_scores = []
        for df in doc_frequencies:
            if df == 0:
                idf = 0.0
            else:
                idf = math.log(n_docs / df)
            idf_scores.append(idf)
        
        # Build TF-IDF matrix
        tfidf_matrix = []
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            token_counts = {}
            
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            tfidf_vector = [0.0] * self.vocab_size
            total_tokens = len(tokens)
            
            if total_tokens > 0:
                for token, count in token_counts.items():
                    if token in self.vocabulary:
                        idx = self.vocabulary[token]
                        tf = count / total_tokens
                        idf = idf_scores[idx]
                        tfidf_vector[idx] = tf * idf
            
            tfidf_matrix.append(tfidf_vector)
        
        return tfidf_matrix
    
    def _simple_svd(self, matrix: Matrix, k: int) -> Tuple[Matrix, Vector]:
        """Simple SVD using power iteration (similar to tlm's PCA)."""
        # Transpose matrix for term-document format
        n_docs = len(matrix)
        n_terms = len(matrix[0]) if matrix else 0
        
        if n_terms == 0 or k == 0:
            return [], []
        
        # Create term-document matrix (transpose)
        td_matrix = []
        for j in range(n_terms):
            column = [matrix[i][j] for i in range(n_docs)]
            td_matrix.append(column)
        
        # Compute covariance matrix (A^T A)
        cov_matrix = []
        for i in range(n_terms):
            row = []
            for j in range(n_terms):
                dot_prod = sum(td_matrix[i][d] * td_matrix[j][d] for d in range(n_docs))
                row.append(dot_prod / n_docs)
            cov_matrix.append(row)
        
        # Power iteration for top k components
        import random
        rng = random.Random(42)
        
        components = []
        eigenvalues = []
        
        for comp_idx in range(min(k, n_terms)):
            # Random initialization
            v = [rng.gauss(0, 1) for _ in range(n_terms)]
            
            # Power iteration
            for _ in range(50):  # Max iterations
                # Matrix-vector multiplication: Cv
                new_v = [0.0] * n_terms
                for i in range(n_terms):
                    for j in range(n_terms):
                        new_v[i] += cov_matrix[i][j] * v[j]
                
                # Normalize
                norm = math.sqrt(sum(x * x for x in new_v))
                if norm > 1e-10:
                    v = [x / norm for x in new_v]
                else:
                    break
            
            # Compute eigenvalue
            eigenval = sum(cov_matrix[i][j] * v[i] * v[j] for i in range(n_terms) for j in range(n_terms))
            
            components.append(v)
            eigenvalues.append(max(0, eigenval))
            
            # Deflate covariance matrix (remove this component)
            for i in range(n_terms):
                for j in range(n_terms):
                    cov_matrix[i][j] -= eigenval * v[i] * v[j]
        
        return components, eigenvalues
    
    def _transform_with_components(self, tfidf_vector: Vector) -> Vector:
        """Transform TF-IDF vector using LSA components."""
        if not self.components:
            return [0.0] * self.n_components
        
        result = []
        for component in self.components:
            # Dot product with component
            proj = sum(tfidf_vector[i] * component[i] for i in range(len(tfidf_vector)))
            result.append(proj)
        
        return result

def fit(sentences: List[str], n_components: int = 100) -> LSAModel:
    """Fit LSA model on sentences."""
    model = LSAModel(n_components)
    
    # Build TF-IDF matrix
    tfidf_matrix = model._build_tfidf_matrix(sentences)
    
    # Compute SVD components
    components, eigenvalues = model._simple_svd(tfidf_matrix, n_components)
    model.components = components
    model.explained_variance = eigenvalues
    
    return model

def transform(sentences: List[str], model: LSAModel) -> Matrix:
    """Transform sentences to LSA embeddings."""
    embeddings = []
    
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        token_counts = {}
        
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Create TF-IDF vector for this sentence
        tfidf_vector = [0.0] * model.vocab_size
        total_tokens = len(tokens)
        
        if total_tokens > 0:
            for token, count in token_counts.items():
                if token in model.vocabulary:
                    idx = model.vocabulary[token]
                    tf = count / total_tokens
                    tfidf_vector[idx] = tf  # Simplified: no IDF for transform
        
        # Transform using LSA components
        lsa_embedding = model._transform_with_components(tfidf_vector)
        embeddings.append(lsa_embedding)
    
    return embeddings

def fit_transform(sentences: List[str], n_components: int = 100) -> Tuple[Matrix, LSAModel]:
    """Fit LSA model and transform sentences in one step."""
    model = fit(sentences, n_components)
    embeddings = transform(sentences, model)
    return embeddings, model