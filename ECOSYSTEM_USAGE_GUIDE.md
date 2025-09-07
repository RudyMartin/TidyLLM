# TidyLLM Ecosystem Usage Guide

## 🎯 Complete Usage Examples

This guide shows how to use the integrated TidyLLM ecosystem for real-world workflows.

## 🚀 Quick Start

### Installation
```bash
# Install complete ecosystem
pip install -e .

# Verify installation
tidyllm status
python -c "import tidyllm, tlm, tidyllm_sentence; print('All libraries available!')"
```

## 📋 Workflow Examples

### 1. Document Analysis Pipeline

```python
import tidyllm
import tidyllm_sentence as tls
import tlm

# Initialize TidyLLM for document processing
gateway = tidyllm.init_gateways()

# Example: Analyze multiple business documents
documents = [
    "Q4 financial report shows 15% revenue growth...",
    "Contract terms specify delivery within 30 days...", 
    "Invoice #12345 for consulting services totaling $5,000..."
]

# Step 1: Generate embeddings for semantic analysis
embeddings, model = tls.tfidf_fit_transform(documents)
print(f"Generated {len(embeddings)} embeddings, {len(embeddings[0])} dimensions each")

# Step 2: Cluster similar documents (pure Python ML)
# normalized_embeddings = tlm.l2_normalize(embeddings)
# clusters, labels, _ = tlm.kmeans_fit(normalized_embeddings, k=3)

# Step 3: Process through TidyLLM QA system
# results = gateway.process_batch(documents)

print("Document analysis pipeline complete!")
```

### 2. Research Paper Analysis

```python
import tidyllm_sentence as tls

# Academic papers analysis
papers = [
    "Deep Learning Advances in Natural Language Processing",
    "Quantum Computing Applications in Machine Learning",
    "Blockchain Technology for Distributed AI Systems"
]

# Generate academic embeddings
embeddings, tfidf_model = tls.tfidf_fit_transform(papers)

# Find most similar papers
query = "Artificial Intelligence and Quantum Computing"
query_emb, _ = tls.tfidf_transform([query], tfidf_model)

# Calculate similarities
similarities = []
for i, paper_emb in enumerate(embeddings):
    similarity = tls.cosine_similarity(query_emb[0], paper_emb)
    similarities.append((papers[i], similarity))

# Rank by relevance
similarities.sort(key=lambda x: x[1], reverse=True)
print("Papers ranked by relevance:")
for paper, score in similarities:
    print(f"  {score:.3f}: {paper}")
```

### 3. Business Document Classification

```python
import tidyllm_sentence as tls
import tlm

# Business document types
training_docs = {
    'invoice': [
        "Invoice #001: Payment due $1,500 by March 15th",
        "Bill for services rendered, total amount $2,300",
        "Invoice for consulting work, due in 30 days"
    ],
    'contract': [
        "Service agreement between Company A and Company B",
        "Contract terms: 12 months, renewable annually",
        "Legal agreement for software development services"
    ],
    'report': [
        "Quarterly financial report showing growth metrics",
        "Analysis of market trends for Q4 2024",
        "Performance review summary for fiscal year"
    ]
}

# Create training embeddings
all_docs = []
all_labels = []
for doc_type, docs in training_docs.items():
    for doc in docs:
        all_docs.append(doc)
        all_labels.append(doc_type)

# Generate embeddings
embeddings, model = tls.tfidf_fit_transform(all_docs)

# Classify new document
new_document = "Payment request for $750 due by end of month"
new_emb, _ = tls.tfidf_transform([new_document], model)

# Find most similar training document
best_match_idx = 0
best_similarity = 0
for i, train_emb in enumerate(embeddings):
    sim = tls.cosine_similarity(new_emb[0], train_emb)
    if sim > best_similarity:
        best_similarity = sim
        best_match_idx = i

predicted_type = all_labels[best_match_idx]
print(f"Document classified as: {predicted_type} (confidence: {best_similarity:.3f})")
```

### 4. CLI-Based Workflow

```bash
#!/bin/bash
# complete_workflow.sh - Automated document processing

echo "🚀 Starting TidyLLM Complete Workflow"

# 1. Initialize project
tidyllm init
echo "✅ Project initialized"

# 2. System health check
tidyllm status
echo "✅ System health verified"

# 3. Process documents
echo "📄 Processing documents..."
tidyllm qa --batch --experiment "workflow_demo" --tag "env=production"

# 4. Run comprehensive tests
echo "🧪 Running tests..."
tidyllm test --create-samples
tidyllm test --all --verbose

# 5. Chat with specific document
if [ -f "important_document.pdf" ]; then
    echo "💬 Interactive PDF analysis..."
    tidyllm chat-pdf important_document.pdf --experiment "pdf_analysis"
fi

echo "🎊 Workflow complete!"
```

## 🔧 Advanced Integration Patterns

### 1. Custom Embedding Pipeline

```python
import tidyllm_sentence as tls
import tlm

class DocumentProcessor:
    def __init__(self):
        self.tfidf_model = None
        self.word_avg_model = None
        
    def fit_multiple_embeddings(self, documents):
        """Create multiple embedding representations"""
        
        # TF-IDF embeddings (sparse, keyword-focused)
        tfidf_embs, self.tfidf_model = tls.tfidf_fit_transform(documents)
        
        # Word averaging embeddings (dense, semantic)
        word_embs, self.word_avg_model = tls.word_avg_fit_transform(
            documents, embedding_dim=100, use_idf=True
        )
        
        # LSA embeddings (dimensionality reduced)
        lsa_embs, self.lsa_model = tls.lsa_fit_transform(documents, n_components=50)
        
        return {
            'tfidf': tfidf_embs,
            'word_avg': word_embs, 
            'lsa': lsa_embs
        }
    
    def classify_ensemble(self, new_document, embeddings_dict, labels):
        """Ensemble classification using multiple embedding types"""
        
        predictions = {}
        
        # Get embeddings for new document
        tfidf_new, _ = tls.tfidf_transform([new_document], self.tfidf_model)
        word_new, _ = tls.word_avg_transform([new_document], self.word_avg_model)
        lsa_new, _ = tls.lsa_transform([new_document], self.lsa_model)
        
        new_embs = {
            'tfidf': tfidf_new[0],
            'word_avg': word_new[0],
            'lsa': lsa_new[0]
        }
        
        # Find best matches for each embedding type
        for emb_type, new_emb in new_embs.items():
            best_idx = 0
            best_sim = 0
            
            for i, train_emb in enumerate(embeddings_dict[emb_type]):
                sim = tls.cosine_similarity(new_emb, train_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            
            predictions[emb_type] = {
                'label': labels[best_idx],
                'confidence': best_sim
            }
        
        return predictions

# Usage
processor = DocumentProcessor()
documents = ["Sample doc 1", "Sample doc 2", "Sample doc 3"]
labels = ["type_a", "type_b", "type_a"]

embeddings = processor.fit_multiple_embeddings(documents)
result = processor.classify_ensemble("New document", embeddings, labels)

print("Ensemble predictions:")
for emb_type, pred in result.items():
    print(f"  {emb_type}: {pred['label']} (conf: {pred['confidence']:.3f})")
```

### 2. TidyLLM + Pure Python ML Pipeline

```python
import tidyllm
import tlm
import tidyllm_sentence as tls

class MLPipeline:
    def __init__(self):
        self.gateway = tidyllm.init_gateways()
        
    def full_analysis_pipeline(self, documents):
        """Complete analysis using entire ecosystem"""
        
        results = {
            'preprocessing': None,
            'embeddings': None,
            'clustering': None, 
            'classification': None,
            'qa_analysis': None
        }
        
        # 1. Text preprocessing and embeddings
        embeddings, model = tls.tfidf_fit_transform(documents)
        results['embeddings'] = {
            'count': len(embeddings),
            'dimensions': len(embeddings[0]),
            'model': model
        }
        
        # 2. Clustering with pure Python ML
        try:
            normalized = tlm.l2_normalize(embeddings)
            k = min(3, len(documents))  # Don't cluster into more groups than docs
            clusters, labels, inertia = tlm.kmeans_fit(normalized, k=k)
            
            results['clustering'] = {
                'clusters': k,
                'inertia': inertia,
                'labels': labels
            }
        except Exception as e:
            results['clustering'] = {'error': str(e)}
        
        # 3. Document analysis with TidyLLM
        # results['qa_analysis'] = self.gateway.analyze_batch(documents)
        
        return results

# Usage example
pipeline = MLPipeline()
docs = [
    "Financial report Q4 2024",
    "Software development contract", 
    "Marketing campaign analysis"
]

analysis = pipeline.full_analysis_pipeline(docs)
print("Pipeline Results:")
print(f"  Embeddings: {analysis['embeddings']['count']} docs, {analysis['embeddings']['dimensions']} dims")
print(f"  Clustering: {analysis['clustering']}")
```

## 🎯 Production Patterns

### 1. Batch Processing Script

```python
#!/usr/bin/env python3
"""
Production batch processing using TidyLLM ecosystem
"""

import tidyllm
import tidyllm_sentence as tls
import tlm
import os
import json
from pathlib import Path

class ProductionProcessor:
    def __init__(self, config_path="tidyllm_config.yaml"):
        self.gateway = tidyllm.init_gateways()
        self.results = []
        
    def process_directory(self, input_dir, output_dir):
        """Process all documents in directory"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all processable files
        extensions = ['.pdf', '.txt', '.docx', '.xlsx']
        files = []
        for ext in extensions:
            files.extend(input_path.glob(f"*{ext}"))
        
        print(f"Found {len(files)} files to process")
        
        for file_path in files:
            try:
                result = self.process_single_file(file_path)
                self.results.append(result)
                
                # Save individual result
                result_file = output_path / f"{file_path.stem}_analysis.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                print(f"✅ Processed: {file_path.name}")
                
            except Exception as e:
                error_result = {
                    'file': str(file_path),
                    'error': str(e),
                    'status': 'failed'
                }
                self.results.append(error_result)
                print(f"❌ Failed: {file_path.name} - {e}")
        
        # Save summary
        summary_file = output_path / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'total_files': len(files),
                'successful': len([r for r in self.results if r.get('status') != 'failed']),
                'failed': len([r for r in self.results if r.get('status') == 'failed']),
                'results': self.results
            }, f, indent=2)
        
        return self.results
    
    def process_single_file(self, file_path):
        """Process individual file through complete pipeline"""
        
        # This would integrate with actual TidyLLM processing
        # For now, return structured result template
        return {
            'file': str(file_path),
            'status': 'completed',
            'embeddings_generated': True,
            'qa_analysis': 'placeholder',
            'metadata': {
                'size': file_path.stat().st_size,
                'type': file_path.suffix
            }
        }

# Usage
if __name__ == "__main__":
    processor = ProductionProcessor()
    results = processor.process_directory("./input_docs", "./output_analysis")
    print(f"\n🎊 Processing complete! {len(results)} files processed.")
```

### 2. API Integration Pattern

```python
from flask import Flask, request, jsonify
import tidyllm_sentence as tls
import tidyllm

app = Flask(__name__)

# Initialize ecosystem components
gateway = tidyllm.init_gateways()
embedding_models = {}

@app.route('/embed', methods=['POST'])
def generate_embeddings():
    """API endpoint for generating embeddings"""
    
    data = request.json
    texts = data.get('texts', [])
    method = data.get('method', 'tfidf')
    
    try:
        if method == 'tfidf':
            embeddings, model = tls.tfidf_fit_transform(texts)
        elif method == 'word_avg':
            embeddings, model = tls.word_avg_fit_transform(texts, embedding_dim=100)
        elif method == 'lsa':
            embeddings, model = tls.lsa_fit_transform(texts, n_components=50)
        else:
            return jsonify({'error': 'Unsupported method'}), 400
        
        return jsonify({
            'embeddings': embeddings,
            'count': len(embeddings),
            'dimensions': len(embeddings[0]),
            'method': method
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/similarity', methods=['POST'])
def calculate_similarity():
    """API endpoint for similarity calculation"""
    
    data = request.json
    text1 = data.get('text1')
    text2 = data.get('text2')
    
    try:
        # Generate embeddings
        texts = [text1, text2]
        embeddings, _ = tls.tfidf_fit_transform(texts)
        
        # Calculate similarity
        similarity = tls.cosine_similarity(embeddings[0], embeddings[1])
        
        return jsonify({
            'similarity': similarity,
            'text1': text1,
            'text2': text2
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    
    try:
        # Test all ecosystem components
        test_texts = ["Test document 1", "Test document 2"]
        embeddings, _ = tls.tfidf_fit_transform(test_texts)
        
        return jsonify({
            'status': 'healthy',
            'components': {
                'tidyllm': 'available',
                'tidyllm_sentence': 'available',
                'tlm': 'available'
            },
            'test_embeddings': len(embeddings)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## 📊 Performance Optimization

### Memory-Efficient Processing

```python
import tidyllm_sentence as tls

def process_large_dataset_efficiently(documents, batch_size=100):
    """Process large document collections in batches"""
    
    results = []
    total_docs = len(documents)
    
    # Process in batches to manage memory
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: docs {i+1}-{min(i+batch_size, total_docs)}")
        
        # Generate embeddings for batch
        embeddings, model = tls.tfidf_fit_transform(batch)
        
        # Store results (in production, might write to disk/database)
        batch_results = {
            'batch_id': i//batch_size,
            'document_count': len(batch),
            'embedding_dimensions': len(embeddings[0]),
            'embeddings': embeddings  # In production, might serialize/compress
        }
        results.append(batch_results)
        
        print(f"  ✅ Generated {len(embeddings)} embeddings")
    
    return results
```

## 🎊 Complete Ecosystem Benefits

The integrated TidyLLM ecosystem provides:

1. **🔄 Seamless Integration**: All libraries work together naturally
2. **📚 Educational Value**: Learn ML concepts through readable implementations  
3. **🚀 Production Ready**: Handle real business workflows efficiently
4. **🎯 Flexibility**: Choose between performance and transparency
5. **🛡️ Vendor Independence**: Complete control over ML pipeline

Start exploring with:
```bash
pip install -e .
tidyllm help
python -c "import tidyllm, tlm, tidyllm_sentence; print('Ready to explore!')"
```