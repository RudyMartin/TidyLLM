# TLM - Transparent Learning Machines

[![PyPI version](https://badge.fury.io/py/tlm.svg)](https://badge.fury.io/py/tlm)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)](https://pypi.org/project/tlm/)

> **Pure Python machine learning algorithms with zero dependencies**  
> Educational, transparent, and ready for production.

## 🎯 Why TLM?

- **🔬 Educational**: Read and understand every algorithm - no black boxes
- **🪶 Lightweight**: Zero dependencies - just Python standard library
- **🎓 Transparent**: Clear, documented implementations you can learn from
- **🚀 Production Ready**: Efficient algorithms suitable for real applications
- **🔧 NumPy-free**: Perfect for edge deployment and constrained environments

## ⚡ Quick Start

```bash
pip install tlm
```

```python
import tlm

# Linear algebra operations (NumPy-free!)
X = tlm.array([[1, 2], [3, 4]])
y = tlm.dot(X, [1, -1])  # [1*1 + 2*(-1), 3*1 + 4*(-1)] = [-1, -1]

# Machine learning in pure Python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Logistic regression
w, b, history = tlm.logreg_fit(X, y, lr=0.01, epochs=100)
predictions = tlm.logreg_predict(X, w, b)
accuracy = tlm.accuracy(y, predictions)
print(f"Accuracy: {accuracy:.3f}")

# K-means clustering  
centers, labels = tlm.kmeans_fit(X, k=3, max_iters=100)

# Attention mechanisms (transformers!)
Q = tlm.array([[1, 0], [0, 1]])  # Queries
K = tlm.array([[1, 1], [1, 0]])  # Keys  
V = tlm.array([[2, 1], [1, 2]])  # Values
attended, weights = tlm.scaled_dot_product_attention(Q, K, V)
```

## 🧠 What's Inside

**15+ Machine Learning Algorithms** implemented in pure Python with zero dependencies.

### Core Operations (NumPy replacement)
```python
# Array operations
x = tlm.zeros(5)           # [0, 0, 0, 0, 0]
A = tlm.eye(3)             # 3x3 identity matrix
B = tlm.transpose(A)       # Transpose
C = tlm.matmul(A, B)       # Matrix multiplication

# Math functions  
tlm.exp([1, 2, 3])         # [2.718, 7.389, 20.086]
tlm.softmax([1, 2, 3])     # [0.090, 0.244, 0.665]
```

### Machine Learning Algorithms

| Algorithm | Function | Use Case |
|-----------|----------|----------|
| **Logistic Regression** | `logreg_fit()` | Binary classification |
| **Softmax Regression** | `softmax_fit()` | Multi-class classification |
| **K-Means** | `kmeans_fit()` | Clustering |
| **PCA** | `pca_power_fit()` | Dimensionality reduction |
| **Gaussian Mixture** | `gmm_fit()` | Probabilistic clustering |
| **Naive Bayes** | `nb_fit()` | Text classification |
| **SVM** | `svm_fit()` | Classification with margins |
| **Matrix Factorization** | `mf_fit_sgd()` | Recommender systems |
| **Attention** | `scaled_dot_product_attention()` | Transformers |

### Model Selection & Evaluation
```python
# Train/test splits
X_train, X_test, y_train, y_test = tlm.train_test_split(X, y, test_size=0.2)

# Cross-validation
folds = tlm.kfold(X, n_folds=5)

# Metrics
confusion = tlm.confusion_matrix(y_true, y_pred)
acc = tlm.accuracy(y_true, y_pred)
```

## 🎓 Educational Examples

### Understanding Attention (Transformers)
```python
# See how attention works step by step
Q = tlm.array([[1, 0]])  # What are we looking for?
K = tlm.array([[1, 1], [0, 1], [1, 0]])  # What do we have?
V = tlm.array([[10, 5], [3, 8], [7, 2]])  # What are the values?

attended, attention_weights = tlm.scaled_dot_product_attention(Q, K, V)
print(f"Attention weights: {attention_weights[0]}")  # Where to focus
print(f"Attended values: {attended[0]}")  # Weighted combination
```

### PCA from Scratch
```python
# Reduce 100D data to 10D
X = tlm.array(your_high_dim_data)  # Shape: (n_samples, 100)
components, X_reduced = tlm.pca_power_fit_transform(X, n_components=10)

# components[0] is the first principal component
# X_reduced has shape (n_samples, 10)
```

## 🏗️ Architecture

```
tlm/
├── pure/           # Core operations (NumPy replacement)
│   └── ops.py     # Array operations, math functions
├── core/          # Activations, losses, metrics  
├── linear_models/ # Regression algorithms
├── cluster/       # Clustering algorithms
├── decomp/        # PCA, whitening
├── mixture/       # Gaussian Mixture Models
├── attention/     # Transformer mechanisms
├── anomaly/       # Anomaly detection
├── naive_bayes/   # Probabilistic classifiers
├── svm/           # Support Vector Machines
└── mf/            # Matrix factorization
```

## 🎯 Perfect For

- **🎓 Learning ML**: Understand algorithms by reading the source
- **🔬 Research**: Easy to modify and experiment with
- **📱 Edge Deployment**: No heavy dependencies
- **🏫 Teaching**: Students can see exactly how it works
- **🚀 Prototyping**: Quick experimentation without setup

## 🤝 Contributing

TLM is part of the [TidyLLM ecosystem](https://github.com/RudyMartin/TidyLLM). Contributions welcome!

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Made with ❤️ by the TidyLLM Team**  
*Transparent AI for everyone*