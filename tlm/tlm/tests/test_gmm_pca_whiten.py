import tlm, random, math

def test_pca_power_direction():
    X = [[i/10.0 + 0.01*(i%2), 0.1*((-1)**i)] for i in range(100)]
    Z, p = tlm.pca_power_fit_transform(X, k=1, iters=300, n_init=2)
    v = p['components'][0]
    cos = abs(v[0] / max(math.sqrt(v[0]*v[0] + v[1]*v[1]), 1e-12))
    assert cos > 0.9

def test_gmm_two_blobs_diag():
    rng = random.Random(0)
    X0 = [[-2.0 + rng.gauss(0,0.3), -2.0 + rng.gauss(0,0.3)] for _ in range(60)]
    X1 = [[+2.0 + rng.gauss(0,0.3), +2.0 + rng.gauss(0,0.3)] for _ in range(60)]
    X = X0 + X1
    y = [0]*60 + [1]*60
    params = tlm.gmm_fit(X, k=2, max_iter=50, cov='diag', seed=1)
    y_hat = tlm.gmm_predict(X, params)
    acc = sum(1 for i in range(len(y)) if y[i]==y_hat[i]) / len(y)
    acc_swapped = sum(1 for i in range(len(y)) if y[i]==(1-y_hat[i])) / len(y)
    assert max(acc, acc_swapped) > 0.8

def test_whiten_scales_unit_var():
    X = [[i/10.0, 0.1*((-1)**i)] for i in range(100)]
    Z, p = tlm.pca_power_fit_transform(X, k=2, iters=200, n_init=2)
    ZW = tlm.pca_whiten(Z, p)
    def var_col(Z, j):
        mu = sum(row[j] for row in Z)/len(Z)
        return sum((row[j]-mu)**2 for row in Z)/max(len(Z)-1,1)
    v0, v1 = var_col(ZW,0), var_col(ZW,1)
    assert 0.3 < v0 < 3.0 and 0.3 < v1 < 3.0
