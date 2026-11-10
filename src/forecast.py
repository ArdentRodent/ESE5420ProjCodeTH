import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve

HORIZONS = (1, 4, 16, 32)
WARMUP = 600
SEED = 0

def boot_idx(n_subjects, B=10_000, seed=SEED):
    """(B, n_subjects) resampled subject positions, the same for every method."""
    return np.random.default_rng(seed).integers(0, n_subjects, (B, n_subjects))


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
    """R^2 of predict(Z, t0, h) against Z[t0+h] over the standard origins."""
    t0 = origins(len(Z))
    return r2(Z[t0 + h], predict(Z, t0, h))

def fit_ar(Z_train, h, p):
    """Slow, readable reference for fit_ar_grid: per-channel direct h-step AR(p) by least squares."""
    t0 = origins(len(Z_train))
    coef = np.empty((Z_train.shape[1], p + 1))
    for c in range(Z_train.shape[1]):
        X = np.column_stack([Z_train[t0 - k, c] for k in range(p)] + [np.ones(len(t0))])
        coef[c] = np.linalg.lstsq(X, Z_train[t0 + h, c], rcond=None)[0]
    return coef

def fit_ar_grid(Z_train, p_grid, horizons=HORIZONS):
    """Every (h, p) from one Gram matrix per channel. Returns {(h, p): (64, p+1) coef}, intercept last."""
    t0 = origins(len(Z_train)); P = max(p_grid); C = Z_train.shape[1]
    out = {(h, p): np.empty((C, p + 1)) for h in horizons for p in p_grid}
    X = np.ones((len(t0), P + 1))                  # allocated once; last column stays 1 (intercept)
    for c in range(C):
        z = Z_train[:, c]
        # Row i of sliding_window_view(z, P) is z[i : i+P]. Row t-P+1, reversed, is z[t], z[t-1], ..., z[t-P+1].
        # One vectorized copy instead of P column copies (~2x faster fit).
        X[:, :P] = sliding_window_view(z, P)[t0 - P + 1, ::-1]
        Y = z[t0[:, None] + np.array(horizons)]    # (n, n_horizons) targets
        G, b = X.T @ X, X.T @ Y
        for p in p_grid:
            idx = np.r_[:p, P]                     # first p lags + intercept column
            sol = np.linalg.solve(G[np.ix_(idx, idx)], b[idx])
            for j, h in enumerate(horizons):
                out[h, p][c] = sol[:, j]
    return out

def ar_predictor(coef):
    """sum_k coef[k] * Z[t-k] + intercept, for all channels in one FFT convolution along time."""
    p = coef.shape[1] - 1
    return lambda Z, t0, h: fftconvolve(Z, coef[:, :p].T, axes=0)[t0] + coef[:, p]
