import numpy as np

HORIZONS = (1, 4, 16, 32)
WARMUP = 600  # >= largest p AND the causal filter's ~3 s settling time

def origins(T, warmup=WARMUP, hmax=max(HORIZONS)):
    """Forecast origins t; target is t+h. Same set for every method and horizon."""
    return np.arange(warmup, T - hmax)

def r2(Y, Yhat):
    """R^2 pooled over all channels (per-channel mean of the evaluated run)."""
    return 1 - ((Y - Yhat) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()

def persistence(Z, t0, h):
    """Predict Z[t0+h] by Z[t0]."""
    return Z[t0]

def evaluate(predict, Z, h):
    t0 = origins(len(Z))
    return r2(Z[t0 + h], predict(Z, t0, h))

def lags(Z, t0, p):
    """(n, 64, p+1): Z[t], Z[t-1], ..., Z[t-p+1] for each channel, plus a column of 1s (intercept)."""
    L = np.stack([Z[t0 - k] for k in range(p)], axis=-1)
    return np.concatenate([L, np.ones(L.shape[:2] + (1,))], axis=-1)

def fit_ar(Z_train, h, p):
    """Per-channel direct h-step AR(p) by least squares, one channel at a time (memory-safe for large p)."""
    t0 = origins(len(Z_train))
    coef = np.empty((Z_train.shape[1], p + 1))
    for c in range(Z_train.shape[1]):
        X = np.column_stack([Z_train[t0 - k, c] for k in range(p)] + [np.ones(len(t0))])
        coef[c] = np.linalg.lstsq(X, Z_train[t0 + h, c], rcond=None)[0]
    return coef

def fit_ar_all(Z_train, p, horizons=HORIZONS):
    """Same as fit_ar, but one lstsq per channel solves every horizon at once. Returns {h: (64, p+1) coef}."""
    t0 = origins(len(Z_train))
    coef = np.empty((len(horizons), Z_train.shape[1], p + 1))
    for c in range(Z_train.shape[1]):
        X = np.column_stack([Z_train[t0 - k, c] for k in range(p)] + [np.ones(len(t0))])
        Y = np.column_stack([Z_train[t0 + h, c] for h in horizons])   # (n, 4): one column per horizon
        coef[:, c, :] = np.linalg.lstsq(X, Y, rcond=None)[0].T
    return dict(zip(horizons, coef))


def fit_ar_grid(Z_train, p_grid, horizons=HORIZONS):
    """Every (h, p) from one Gram matrix per channel. Returns {(h, p): (64, p+1) coef}, intercept last."""
    t0 = origins(len(Z_train)); P = max(p_grid); C = Z_train.shape[1]
    out = {(h, p): np.empty((C, p + 1)) for h in horizons for p in p_grid}
    for c in range(C):
        X = np.column_stack([Z_train[t0 - k, c] for k in range(P)] + [np.ones(len(t0))])
        Y = np.column_stack([Z_train[t0 + h, c] for h in horizons])
        G, b = X.T @ X, X.T @ Y
        for p in p_grid:
            idx = list(range(p)) + [P]                     # first p lags + intercept column
            sol = np.linalg.solve(G[np.ix_(idx, idx)], b[idx])
            for j, h in enumerate(horizons):
                out[h, p][c] = sol[:, j]
    return out

def ar_predictor(coef):
    """sum_k coef[k] * Z[t-k] + intercept, computed by convolution (no lag matrix)."""
    p = coef.shape[1] - 1
    def predict(Z, t0, h):
        return np.column_stack([np.convolve(Z[:, c], coef[c, :p])[t0]
                                for c in range(Z.shape[1])]) + coef[:, p]
    return predict


