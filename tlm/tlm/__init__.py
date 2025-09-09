# tlm/__init__.py — NumPy-free API (pure Python, stdlib only)

__version__ = "1.1.0"

# Core list-based ops
from .pure.ops import (
    array, shape, zeros, ones, eye,
    transpose, dot, matmul,
    add, sub, mul, div,
    sum as asum, mean, var, std, max as amax, min as amin, argmax,
    exp, log, sqrt, clip, norm, l2_normalize,
    flatten, reshape, concatenate,
    seed_rng, random_uniform, random_normal, random_choice,
)

# Activations / losses / metrics
from .core.activations import sigmoid, relu, leaky_relu, softmax
from .core.losses import mse, mae, binary_cross_entropy, cross_entropy
from .core.metrics import confusion_matrix, accuracy

# Linear models
from .linear_models.logistic_regression import (
    fit as logreg_fit,
    predict as logreg_predict,
    predict_proba as logreg_predict_proba,
)
from .linear_models.softmax_regression import (
    fit as softmax_fit,
    predict as softmax_predict,
    predict_proba as softmax_predict_proba,
)

# Clustering
from .cluster.kmeans import fit as kmeans_fit

# Model selection
from .model_selection.split import train_test_split, kfold, stratified_kfold

# SVM (linear)
from .svm.linear import hinge_loss as svm_hinge_loss, fit as svm_fit, predict as svm_predict

# PCA (power iteration) + whitening
from .decomp.pca_power import fit as pca_power_fit, transform as pca_power_transform, fit_transform as pca_power_fit_transform
from .decomp.whiten import whiten as pca_whiten

# GMM (EM, pure)
from .mixture.gmm_pure import fit as gmm_fit, predict_proba as gmm_predict_proba, predict as gmm_predict

# Anomaly detection
from .anomaly.gaussian import fit as anomaly_fit, score_logpdf as anomaly_score_logpdf, predict as anomaly_predict

# Naive Bayes
from .naive_bayes.multinomial import fit as nb_fit, predict_log_proba as nb_predict_log_proba, predict as nb_predict

# Matrix factorization
from .mf.matrix_factorization import fit_sgd as mf_fit_sgd, predict_full as mf_predict_full, mse as mf_mse

# Attention mechanisms
from .attention.scaled_dot_product import scaled_dot_product_attention, multi_head_attention, positional_encoding

__all__ = [
    # ops
    'array','shape','zeros','ones','eye','transpose','dot','matmul',
    'add','sub','mul','div','asum','mean','var','std','amax','amin','argmax',
    'exp','log','sqrt','clip','norm','l2_normalize',
    'flatten','reshape','concatenate',
    # random (classic historical algorithms)
    'seed_rng','random_uniform','random_normal','random_choice',
    # core
    'sigmoid','relu','leaky_relu','softmax','mse','mae','binary_cross_entropy','cross_entropy','confusion_matrix','accuracy',
    # linear models & clustering & selection
    'logreg_fit','logreg_predict','logreg_predict_proba',
    'softmax_fit','softmax_predict','softmax_predict_proba',
    'kmeans_fit','train_test_split','kfold','stratified_kfold',
    # svm
    'svm_hinge_loss','svm_fit','svm_predict',
    # pca
    'pca_power_fit','pca_power_transform','pca_power_fit_transform','pca_whiten',
    # gmm
    'gmm_fit','gmm_predict_proba','gmm_predict',
    # anomaly
    'anomaly_fit','anomaly_score_logpdf','anomaly_predict',
    # naive bayes
    'nb_fit','nb_predict_log_proba','nb_predict',
    # matrix factorization
    'mf_fit_sgd','mf_predict_full','mf_mse',
    # attention
    'scaled_dot_product_attention','multi_head_attention','positional_encoding',
]
