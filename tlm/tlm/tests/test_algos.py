import tlm, math, random

def test_softmax_row_sums():
    P = tlm.softmax([[1,2,3],[3,2,1]], axis=1)
    for row in P:
        assert abs(sum(row) - 1.0) < 1e-7

def test_kmeans_shapes():
    X = [[0.0,0.0],[1.0,1.0],[9.0,9.0],[10.0,10.0]]
    C, labels, inertia = tlm.kmeans_fit(X, 2, seed=0)
    assert len(C) == 2 and len(C[0]) == 2
    assert len(labels) == 4

def test_svm_linearly_separable():
    X = [[-2.0,-1.0],[-1.5,-1.2],[-1.0,-0.5],[2.0,1.0],[1.5,1.2],[1.0,0.5]]
    y = [-1,-1,-1, +1,+1,+1]
    w, b, hist = tlm.svm_fit(X, y, lr=0.1, epochs=40, C=1.0, seed=0)
    yp = tlm.svm_predict(X, w, b)
    assert sum(1 for a,b2 in zip(y, yp) if a==b2) / len(y) >= 0.95
