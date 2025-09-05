import random

__all__ = ['train_test_split','kfold','stratified_kfold']

def train_test_split(X, y=None, test_size=0.2, shuffle=True, seed=None):
    """Simple train/test split for lists or list-of-lists."""
    n = len(X)
    test_n = int(n * test_size) if test_size < 1 else int(test_size)
    
    idx = list(range(n))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)
    
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    
    if y is None:
        return X_train, X_test
    
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test

def kfold(n, k, shuffle=True, seed=None):
    """Generate k-fold cross validation splits."""
    idx = list(range(n))
    if shuffle:
        rng = random.Random(seed); rng.shuffle(idx)
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    folds, start = [], 0
    for s in fold_sizes:
        folds.append(idx[start:start+s]); start += s
    for i in range(k):
        test_idx = folds[i]
        train_idx = [j for t, f in enumerate(folds) if t != i for j in f]
        yield train_idx, test_idx

def stratified_kfold(y, k, shuffle=True, seed=None):
    """Generate stratified k-fold cross validation splits."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, c in enumerate(y):
        buckets[c].append(i)
    rng = random.Random(seed)
    if shuffle:
        for b in buckets.values(): rng.shuffle(b)
    bucket_splits = {c: [buckets[c][i::k] for i in range(k)] for c in buckets}
    for i in range(k):
        test_idx = [idx for c in bucket_splits for idx in bucket_splits[c][i]]
        train_set = set(range(len(y))) - set(test_idx)
        yield list(train_set), test_idx
