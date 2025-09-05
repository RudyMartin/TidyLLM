"""
Document Classification Implementation

Local document classification system using tidyllm-sentence embeddings.
Provides multi-category classification with confidence scoring.

Part of the tidyllm-verse: Educational ML with complete transparency
"""

import sys
import os
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

# Add path for tidyllm-sentence dependency
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tidyllm-sentence'))

try:
    import tidyllm_sentence as tls
    TLS_AVAILABLE = True
except ImportError:
    TLS_AVAILABLE = False
    print("Warning: tidyllm-sentence not available. Install tidyllm-sentence for classification.")

from ..extraction import TextExtractor

class DocumentClassifier:
    """
    Local document classification system using embeddings for semantic similarity.
    No external API calls - everything runs locally.
    """
    
    def __init__(self, categories: List[str], metadata_fields: Optional[List[str]] = None):
        """
        Initialize classifier with categories and optional metadata fields.
        
        Args:
            categories: List of document categories to classify
            metadata_fields: Optional list of metadata fields to extract
        """
        self.categories = categories
        self.metadata_fields = metadata_fields or []
        self.text_extractor = TextExtractor()
        
        if not TLS_AVAILABLE:
            print("Warning: Classification will be limited without tidyllm-sentence")
        
        # Initialize category embeddings (will be created on first use)
        self.category_embeddings = None
        self.category_model = None
        
        # Training data storage
        self.training_examples = {category: [] for category in categories}
        
    def classify_document(self, document_path: str, max_pages: int = 5) -> Dict[str, Any]:
        """
        Classify a document and extract metadata.
        
        Args:
            document_path: Path to document file
            max_pages: Maximum pages to process
            
        Returns:
            Dictionary with classification results and metadata
        """
        # Extract text from document
        document_text, extraction_meta = self.text_extractor.extract_text(document_path, max_pages)
        
        if not document_text:
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'error': extraction_meta.get('error', 'No text extracted'),
                'metadata': {}
            }
        
        # Classify the text
        category, confidence = self._classify_text(document_text)
        
        # Extract metadata if fields are specified
        metadata = {}
        if self.metadata_fields:
            from ..extraction import MetadataExtractor
            extractor = MetadataExtractor()
            metadata = extractor.extract_metadata(document_text, self.metadata_fields)
        
        return {
            'category': category,
            'confidence': confidence,
            'text_length': len(document_text),
            'extraction_metadata': extraction_meta,
            'metadata': metadata
        }
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text content directly.
        
        Args:
            text: Text content to classify
            
        Returns:
            Dictionary with classification results
        """
        category, confidence = self._classify_text(text)
        
        # Extract metadata if fields are specified
        metadata = {}
        if self.metadata_fields:
            from ..extraction import MetadataExtractor
            extractor = MetadataExtractor()
            metadata = extractor.extract_metadata(text, self.metadata_fields)
        
        return {
            'category': category,
            'confidence': confidence,
            'text_length': len(text),
            'metadata': metadata
        }
    
    def _classify_text(self, text: str) -> Tuple[str, float]:
        """
        Internal method to classify text using embeddings or fallback methods.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (category, confidence_score)
        """
        if not TLS_AVAILABLE:
            return self._fallback_classification(text)
        
        try:
            # Initialize or update category embeddings if needed
            if self.category_embeddings is None:
                self._initialize_category_embeddings()
            
            # Use the same model to transform the input text
            if self.category_model is None:
                return self._fallback_classification(text)
            
            # Transform the input text using the same TF-IDF model
            text_embeddings = tls.tfidf_transform([text], self.category_model)
            text_embedding = text_embeddings[0]
            
            # Calculate similarity to each category
            similarities = []
            for i, category in enumerate(self.categories):
                if i < len(self.category_embeddings):
                    similarity = tls.cosine_similarity(text_embedding, self.category_embeddings[i])
                    similarities.append((category, similarity))
            
            if similarities:
                # Sort by similarity and return best match
                similarities.sort(key=lambda x: x[1], reverse=True)
                best_category, best_similarity = similarities[0]
                
                # Convert similarity to confidence (0.0 to 1.0)
                confidence = min(max(best_similarity, 0.0), 1.0)
                
                return best_category, confidence
            else:
                return self._fallback_classification(text)
                
        except Exception as e:
            print(f"Classification error: {e}")
            return self._fallback_classification(text)
    
    def _initialize_category_embeddings(self):
        """Initialize embeddings for each category using representative text."""
        if not TLS_AVAILABLE:
            return
        
        # Create basic representative text for each category
        category_texts = []
        for category in self.categories:
            # Use training examples if available
            if self.training_examples[category]:
                # Use the training examples
                category_text = " ".join(self.training_examples[category])
            else:
                # Create basic representative text based on category name
                category_text = self._generate_category_representative_text(category)
            
            category_texts.append(category_text)
        
        # Generate embeddings for all categories
        try:
            self.category_embeddings, self.category_model = tls.tfidf_fit_transform(category_texts)
        except Exception as e:
            print(f"Error creating category embeddings: {e}")
            self.category_embeddings = None
            self.category_model = None
    
    def _generate_category_representative_text(self, category: str) -> str:
        """Generate representative text for a category based on its name."""
        # Basic keyword mapping for common document types
        category_keywords = {
            'invoice': 'invoice bill payment amount due total charge cost expense',
            'contract': 'agreement contract terms conditions party parties legal binding',
            'report': 'report analysis summary findings results data research study',
            'email': 'email message from to subject regarding communication correspondence',
            'memo': 'memorandum memo notice announcement information communication internal',
            'letter': 'letter correspondence communication formal written document',
            'proposal': 'proposal recommendation suggestion plan project offer bid',
            'manual': 'manual guide instructions procedure process documentation handbook',
            'policy': 'policy procedure rule regulation guideline standard requirement',
            'financial': 'financial money cost budget expense revenue profit loss accounting',
            'legal': 'legal law court attorney lawyer litigation case judgment',
            'technical': 'technical specification design engineering system software hardware',
        }
        
        # Return keywords for the category or use the category name itself
        return category_keywords.get(category.lower(), category.lower().replace('_', ' '))
    
    def _fallback_classification(self, text: str) -> Tuple[str, float]:
        """
        Fallback classification method using simple keyword matching.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (category, confidence_score)
        """
        text_lower = text.lower()
        category_scores = {}
        
        for category in self.categories:
            score = 0.0
            category_keywords = self._generate_category_representative_text(category).split()
            
            for keyword in category_keywords:
                if keyword.lower() in text_lower:
                    score += 1.0
            
            # Normalize by number of keywords
            if category_keywords:
                score /= len(category_keywords)
            
            category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            confidence = min(category_scores[best_category], 1.0)
            return best_category, confidence
        else:
            return self.categories[0] if self.categories else 'unknown', 0.0
    
    def train_category(self, category: str, training_texts: List[str]):
        """
        Add training examples for a specific category.
        
        Args:
            category: Category name
            training_texts: List of example texts for this category
        """
        if category not in self.categories:
            raise ValueError(f"Category '{category}' not in initialized categories: {self.categories}")
        
        self.training_examples[category].extend(training_texts)
        
        # Reset category embeddings to force recalculation
        self.category_embeddings = None
        self.category_model = None
    
    def get_training_summary(self) -> Dict[str, int]:
        """Get summary of training examples per category."""
        return {category: len(examples) for category, examples in self.training_examples.items()}
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return self.text_extractor.get_supported_formats()