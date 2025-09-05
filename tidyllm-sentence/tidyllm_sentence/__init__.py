# tidyllm-sentence/__init__.py — Pure Python sentence embeddings

# TF-IDF based embeddings
from .tfidf.embeddings import fit as tfidf_fit, transform as tfidf_transform, fit_transform as tfidf_fit_transform

# Word averaging embeddings  
from .word_avg.embeddings import fit as word_avg_fit, transform as word_avg_transform, fit_transform as word_avg_fit_transform

# N-gram based embeddings
from .ngram.embeddings import fit as ngram_fit, transform as ngram_transform, fit_transform as ngram_fit_transform

# LSA/SVD based embeddings
from .lsa.embeddings import fit as lsa_fit, transform as lsa_transform, fit_transform as lsa_fit_transform

# Transformer-enhanced embeddings
from .transformer.embeddings import transformer_fit, transformer_transform, transformer_fit_transform

# Utilities
from .utils.tokenize import simple_tokenize, word_tokenize, char_ngrams
from .utils.similarity import cosine_similarity, semantic_search
from .utils.preprocessing import preprocess_for_embeddings, PreprocessingPipeline, STANDARD_PIPELINE, MINIMAL_PIPELINE, AGGRESSIVE_PIPELINE
from .utils.stemmer import simple_stem, porter_stem  
from .utils.stopwords import ENGLISH_STOP_WORDS, CORE_STOP_WORDS, remove_stopwords

__version__ = "0.1.0"

# Academic Validation & Research Documentation:
# - EXHIBIT_1_ACADEMIC_BENCHMARK.md: Peer-review ready comparison vs FAISS
# - TLM_TEAM_STRATEGY.md: Multi-algorithm ensemble strategy  
# - FINAL_RABBIT_SUMMARY.md: Complete competitive analysis
# 
# The tidyllm-verse: Where algorithmic sovereignty meets research competitiveness

__all__ = [
    # TF-IDF
    'tfidf_fit', 'tfidf_transform', 'tfidf_fit_transform',
    # Word averaging
    'word_avg_fit', 'word_avg_transform', 'word_avg_fit_transform', 
    # N-gram
    'ngram_fit', 'ngram_transform', 'ngram_fit_transform',
    # LSA
    'lsa_fit', 'lsa_transform', 'lsa_fit_transform',
    # Transformer
    'transformer_fit', 'transformer_transform', 'transformer_fit_transform',
    # Utils
    'simple_tokenize', 'word_tokenize', 'char_ngrams',
    'cosine_similarity', 'semantic_search',
    'preprocess_for_embeddings', 'PreprocessingPipeline', 
    'STANDARD_PIPELINE', 'MINIMAL_PIPELINE', 'AGGRESSIVE_PIPELINE',
    'simple_stem', 'porter_stem', 'ENGLISH_STOP_WORDS', 'remove_stopwords',
]