import tlm, random

def test_nb_simple():
    X = [[3,1],[4,0],[0,5],[1,6]]
    y = [0,0,1,1]
    model = tlm.nb_fit(X, y, alpha=1.0)
    preds = tlm.nb_predict(X, model)
    assert sum(int(a==b) for a,b in zip(preds,y)) >= 3

def test_anomaly_gaussian():
    rng = random.Random(0)
    X = [[rng.gauss(0,1), rng.gauss(0,1)] for _ in range(200)]
    X += [[7.0,7.0],[8.0,-8.0],[10.0,0.0]]
    mu, var = tlm.anomaly_fit(X)
    flags = tlm.anomaly_predict(X, mu, var, percentile=2.0)
    assert sum(flags) >= 3

def test_mf_sgd_decreases_mse():
    R = [
        [5.0, 3.0, None, 1.0],
        [4.0, None, None, 1.0],
        [1.0, 1.0, None, 5.0],
        [1.0, None, None, 4.0],
        [None, 1.0, 5.0, 4.0],
    ]
    P, Q, hist = tlm.mf_fit_sgd(R, k=2, epochs=20, lr=0.05, reg=0.02, seed=0)
    assert hist[-1] <= hist[0]
