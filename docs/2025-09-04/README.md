# tlm (NumPy-free)

A tiny, teachable library with a NumPy-like feel implemented **entirely in pure Python**  
Creates **Python-native arrays** (using lists) instead of numpy arrays for 100% dependency-free operation.
Part of the tidyllm verse.

## Install (editable)
```bash
pip install -e .
```

## Quick Start
```python
import tlm

# Data preparation
X = [[-1.0, 0.5], [0.7, -0.2], [1.3, 0.1], [0.9, 0.4]]
y = [0, 1, 1, 1]
```

## Linear Models
```python
# Softmax regression (multiclass)
W, b, hist = tlm.softmax_fit(X, y, lr=0.1, epochs=50)
pred = tlm.softmax_predict(X, W, b)
proba = tlm.softmax_predict_proba(X, W, b)
print("accuracy:", tlm.accuracy(y, pred))

# Logistic regression (binary)
y_bin = [0, 1, 1, 1]  
w, b, hist = tlm.logreg_fit(X, y_bin, lr=0.1, epochs=50)
pred = tlm.logreg_predict(X, w, b)
```

## Clustering & Dimensionality Reduction
```python
# K-means clustering
C, labels, inertia = tlm.kmeans_fit(X, k=2, seed=0)

# PCA with power iteration
W, mu = tlm.pca_power_fit(X, k=2, max_iter=100)
X_transformed = tlm.pca_power_transform(X, W, mu)

# Gaussian Mixture Model
weights, mu, sigma = tlm.gmm_fit(X, k=2, seed=0)
cluster_proba = tlm.gmm_predict_proba(X, weights, mu, sigma)
```

## Support Vector Machines
```python
# Linear SVM (labels in {-1,+1})
y_svm = [-1, +1, +1, +1]
w, b, hist = tlm.svm_fit(X, y_svm, lr=0.1, epochs=50, C=1.0, seed=0)
pred = tlm.svm_predict(X, w, b)
loss = tlm.svm_hinge_loss(X, y_svm, w, b, C=1.0)
```

## Other Algorithms
```python
# Naive Bayes (multinomial)
model = tlm.nb_fit(X, y)
pred = tlm.nb_predict(X, model)

# Anomaly detection (Gaussian)
mu, sigma = tlm.anomaly_fit(X)
scores = tlm.anomaly_score_logpdf(X, mu, sigma)
anomalies = tlm.anomaly_predict(X, mu, sigma, threshold=-2.0)

# Matrix factorization
U, V = tlm.mf_fit_sgd(ratings_matrix, k=10, lr=0.01, epochs=100)
predicted_ratings = tlm.mf_predict_full(U, V)
```

This project is intentionally simple so students can read every loop and check **math + complexity**.

## Key Design Philosophy

**TLM creates Python-native arrays**: All operations return standard Python lists `[1.0, 2.0, 3.0]` instead of numpy arrays `array([1.0, 2.0, 3.0])`. This enables:
- ✅ **100% numpy-free operation** - no external dependencies
- ✅ **Direct numpy substitution** - use `import tlm as np` 
- ✅ **Pure Python arrays** - lists with array semantics
- ✅ **Educational transparency** - every data structure is visible

---

**Built with ❤️ using [CodeBreakers](https://github.com/rudymartin/codebreakers_manifesto) / tidyllm verse**
