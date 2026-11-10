import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import irfft, next_fast_len, rfft
from scipy.linalg import solve
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


# ---------- PCA + VAR ----------

def pca_basis(Z_train):
    """Principal axes of the run-4 channels, (64, 64) with columns sorted by variance, and each one's variance fraction.
    No centering: z-scoring on run 4 already makes each channel's run-4 mean 0."""
    evals, V = np.linalg.eigh(Z_train.T @ Z_train / len(Z_train))
    order = evals.argsort()[::-1]
    return V[:, order], evals[order] / evals.sum()

def var_columns(r, p, R, P):
    """Columns of lag_matrix(S[:, :R], t0, P) for the first r components at lags 0..p-1, plus the intercept (last)."""
    return np.r_[(np.arange(r)[:, None] * P + np.arange(p)).ravel(), R * P]

def lag_matrix(S, t0, P):
    """(n, R*P + 1): for each component j, columns j*P .. j*P+P-1 hold S[t, j], S[t-1, j], ..., S[t-P+1, j]; last column is 1."""
    X = np.ones((len(t0), S.shape[1] * P + 1))
    for j in range(S.shape[1]):
        X[:, j * P:(j + 1) * P] = sliding_window_view(S[:, j], P)[t0 - P + 1, ::-1]   # same trick as fit_ar_grid
    return X

def fit_var_grid(Z_train, V, r_grid, p_grid, horizons=HORIZONS):
    """Direct h-step VAR(p) on the first r principal components, by least squares, for every (h, r, p).
    Returns {(h, r, p): (r*p + 1, r) coef}; rows follow lag_matrix's column order, intercept last.
    Components and lags are nested, so one Gram matrix at (max r, max p) contains every smaller model as a sub-block."""
    t0 = origins(len(Z_train)); R, P = max(r_grid), max(p_grid)
    S = Z_train @ V[:, :R]
    X = lag_matrix(S, t0, P)
    Y = np.hstack([S[t0 + h] for h in horizons])           # (n, R * n_horizons) targets
    G, b = X.T @ X, X.T @ Y
    out = {}
    for r in r_grid:
        for p in p_grid:
            idx = var_columns(r, p, R, P)
            sol = solve(G[np.ix_(idx, idx)], b[idx], assume_a="pos")     # Cholesky: G is symmetric positive definite
            for i, h in enumerate(horizons):
                out[h, r, p] = sol[:, i * R:i * R + r]
    return out

def score_var_grid(Z_val, V, coefs, horizons=HORIZONS):
    """Validation R^2 for every (h, r, p) in coefs; equals evaluate(var_predictor(coef, V), Z_val, h) to ~1e-8, ~10x faster.
    Uses SSE = y'y - 2 B'X'y + B'X'X B on a single validation Gram matrix instead of forecasting each model.
    Because V is orthonormal, channel SSE = SSE of the r forecast components + sum of squares of the dropped components."""
    R = max(k[1] for k in coefs); P = max(k[2] for k in coefs)
    t0 = origins(len(Z_val))
    S = Z_val @ V                                           # all 64 components
    X = lag_matrix(S[:, :R], t0, P)
    G = X.T @ X
    out = {}
    for h in horizons:
        Y = S[t0 + h]
        b = X.T @ Y[:, :R]
        sq = (Y ** 2).sum(0)                                # per-component sum of squares
        Zh = Z_val[t0 + h]
        sst = ((Zh - Zh.mean(0)) ** 2).sum()
        for (hh, r, p), B in coefs.items():
            if hh != h:
                continue
            idx = var_columns(r, p, R, P)
            sse = (sq[:r].sum() - 2 * (B * b[idx, :r]).sum()
                   + (B * (G[np.ix_(idx, idx)] @ B)).sum() + sq[r:].sum())
            out[h, r, p] = 1 - sse / sst
    return out

def var_predictor(coef, V):
    """Forecast channels from a fit_var_grid coef: VAR on the first r components, then map back with V.
    Same predict(Z, t0, h) interface as the other methods."""
    r = coef.shape[1]; p = (coef.shape[0] - 1) // r
    A = coef[:-1].reshape(r, p, r)                          # A[j, k, i]: weight of component j at lag k on component i
    Vr = V[:, :r]
    def predict(Z, t0, h):
        S = Z @ Vr
        nf = next_fast_len(len(S) + p - 1)
        # Shat_i(t) = sum_j sum_k A[j, k, i] S_j(t - k): a sum of r*r convolutions, done as products in the frequency domain.
        Sf = rfft(S, nf, axis=0)
        acc = np.zeros((Sf.shape[0], r), complex)
        for j in range(r):
            acc += Sf[:, j:j + 1] * rfft(A[j], nf, axis=0)
        return (irfft(acc, nf, axis=0)[t0] + coef[-1]) @ Vr.T
    return predict
