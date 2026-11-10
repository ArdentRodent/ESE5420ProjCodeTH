"""Stochastic subspace identification (CVA) with a steady-state Kalman-filter forecast.

Model (innovation form):  x[t+1] = A x[t] + K e[t],   y[t] = C x[t] + e[t].
Fit: canonical correlation analysis between the past and the future of y gives states, then least squares gives C, A, K.
Forecast from data up to t:  y_hat[t+h] = C A^(h-1) x[t+1|t], with x[t+1|t] from the Kalman predictor.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import cholesky, solve_discrete_lyapunov, solve_triangular, svd

from .forecast import HORIZONS, WARMUP, origins, r2


def past_future(Y, t, s):
    """Block-Hankel rows at times t: past = y[t-s .. t-1] and future = y[t .. t+s-1], each (len(t), m*s)."""
    W = sliding_window_view(Y, s, axis=0)                    # W[i] = Y[i:i+s].T, a view (no copy)
    ms = Y.shape[1] * s
    return W[t - s].reshape(len(t), ms), W[t].reshape(len(t), ms)


def fit_ssid(Y, s, orders, warmup=WARMUP):
    """CVA subspace identification for every order n in orders, from one SVD.
    Returns {n: (A, C, K)} and the canonical correlations between past and future (their drop-off indicates the order)."""
    t = np.arange(max(s, warmup), len(Y) - s + 1)
    past, fut = past_future(Y, t, s)
    N = len(t)
    Cpp, Cff, Cfp = past.T @ past / N, fut.T @ fut / N, fut.T @ past / N
    ridge = lambda C: C + 1e-10 * np.trace(C) / len(C) * np.eye(len(C))   # guards Cholesky; band-limited EEG is near-singular
    Lp, Lf = cholesky(ridge(Cpp), lower=True), cholesky(ridge(Cff), lower=True)
    # Canonical correlations = singular values of Lf^-1 Cfp Lp^-T (whitened past-future covariance).
    M = solve_triangular(Lf, solve_triangular(Lp, Cfp.T, lower=True).T, lower=True)
    _, cc, Vt = svd(M, full_matrices=False)
    # States x_t = Vt[:n] Lp^-1 past_t. Orders are nested: the order-n states are the first n columns.
    X = past @ solve_triangular(Lp, Vt[:max(orders)].T, lower=True, trans="T")
    y = Y[t]
    models = {}
    for n in orders:
        x = X[:, :n]
        C = np.linalg.lstsq(x, y, rcond=None)[0].T                          # y_t = C x_t + e_t
        e = y - x @ C.T
        AK = np.linalg.lstsq(np.hstack([x[:-1], e[:-1]]), x[1:], rcond=None)[0].T   # x_{t+1} = A x_t + K e_t
        models[n] = (AK[:, :n], C, AK[:, n:])
    return models, cc


def clip_unstable(A, rmax=0.999):
    """Shrink eigenvalues with |lambda| > rmax onto radius rmax (Becker et al. 2018). Returns (A, number clipped)."""
    lam, W = np.linalg.eig(A)
    bad = np.abs(lam) > rmax
    if not bad.any():
        return A, 0
    lam[bad] *= rmax / np.abs(lam[bad])
    return (W @ np.diag(lam) @ np.linalg.inv(W)).real, int(bad.sum())


def kalman_states(Y, A, C, K):
    """One-step predicted states x[t|t-1] for t = 0..T (row t), starting from x = 0.
    Steady-state predictor: x[t+1|t] = (A - K C) x[t|t-1] + K y[t]."""
    F = A - K @ C
    KY = Y @ K.T                                             # input term for every t at once
    Xp = np.zeros((len(Y) + 1, len(A)))
    for t in range(len(Y)):                                  # sequential by nature; ~20k small mat-vecs, ~40 ms
        Xp[t + 1] = F @ Xp[t] + KY[t]
    return Xp


def score_ssm(Y, model, horizons=HORIZONS):
    """R^2 at each horizon of the forecast y_hat[t+h] = C A^(h-1) x[t+1|t], over the standard origins. Returns {h: R^2}."""
    A, C, K = model
    Xp = kalman_states(Y, A, C, K)
    t0 = origins(len(Y))
    return {h: r2(Y[t0 + h], Xp[t0 + 1] @ (C @ np.linalg.matrix_power(A, h - 1)).T) for h in horizons}


def simulate(n, m, T, rng, radius=(0.8, 0.98), snr=1.0, burn=1000):
    """A random stable system of known order n (even) with m outputs. Returns run() -> (T, m) output, a fresh realization per call.
    x[t+1] = A x[t] + w[t],  y[t] = C x[t] + v[t];  A has n/2 damped oscillations with |lambda| in radius;
    the noise variance is set so the average per-channel signal-to-noise ratio is snr."""
    A = np.zeros((n, n))
    for i in range(n // 2):
        rad, th = rng.uniform(*radius), rng.uniform(0.05, np.pi - 0.05)
        A[2 * i:2 * i + 2, 2 * i:2 * i + 2] = rad * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ A @ Q.T                                          # same eigenvalues, mixed coordinates
    C = rng.standard_normal((m, n))
    signal_var = np.diag(C @ solve_discrete_lyapunov(A, np.eye(n)) @ C.T).mean()
    noise_sd = np.sqrt(signal_var / snr)

    def run():
        W = rng.standard_normal((T + burn, n))
        x = np.zeros((T + burn, n))
        for t in range(T + burn - 1):
            x[t + 1] = A @ x[t] + W[t]
        return x[burn:] @ C.T + noise_sd * rng.standard_normal((T, m))
    return run
