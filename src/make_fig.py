"""Paper version of the key figure, drawn at its printed size (5.5 in wide) so the fonts stay readable.
Same data and choices as results/val_key.png in the notebook; reads only the results CSVs. Run from the project root:
    python src/make_fig.py
Writes docs/fig_key.pdf and docs/fig_key.png (docs/ is not tracked in git).
"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import forecast as fc

BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#2e9e5b", "#52514e"
R_GRID = [2, 4, 8, 16, 32, 48]
X = range(len(R_GRID))            # evenly spaced positions: 32 and 48 collide on a log axis at this size
res, var, ssm = (pd.read_csv(f"results/val_{k}.csv") for k in ("ar", "pcavar", "ssm"))
ar, persist = res[res.method == "AR"], res[res.method == "persist"]
SUBJ = sorted(res.subject.unique())
idx = fc.boot_idx(len(SUBJ))
band = lambda v: np.percentile(v[idx].mean(1), [2.5, 97.5])

def subject_r2(df, h, **setting):
    q = df[df.h == h]
    for col, val in setting.items():
        q = q[q[col] == val]
    return q.set_index("subject").r2.loc[SUBJ].values

plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.95), constrained_layout=True)
for ax, h in zip(axes, fc.HORIZONS):
    for df, cols, color, label in [(var, ["p"], BLUE, "PCA+VAR"), (ssm, ["s", "lam"], GREEN, "state space")]:
        best = df[df.h == h].groupby(["r", *cols]).r2.mean().groupby("r").idxmax()
        curves = np.array([subject_r2(df, h, **dict(zip(["r", *cols], k))) for k in best])
        lo, hi = np.array([band(c) for c in curves]).T
        ax.fill_between(X, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(X, curves.mean(1), color=color, lw=1.3, marker="o", ms=2.8, label=label)
    p_best = ar[ar.h == h].groupby("p").r2.mean().idxmax()
    a = subject_r2(ar, h, p=p_best)
    ax.axhspan(*band(a), color=ORANGE, alpha=0.18, linewidth=0)
    ax.axhline(a.mean(), color=ORANGE, lw=1.3, label="per-channel AR")
    q = subject_r2(persist, h).mean()
    if q >= min(ax.get_ylim()[0], band(a)[0]) - 0.05:
        ax.axhline(q, color=GRAY, lw=1, ls="--", label="persistence")
    ax.set_xticks(X, [str(r) for r in R_GRID]); ax.set_xlim(-0.35, 5.35)
    ax.tick_params(axis="x", labelsize=6.5); ax.tick_params(axis="y", labelsize=6.5, pad=1.5)
    ax.set_xlabel("components or states r", fontsize=6.5, labelpad=1.5)
    ax.set_title(f"h = {h} ({1000 * h / 160:.0f} ms)", fontsize=7.5, pad=3)
    ax.grid(alpha=0.25, lw=0.4); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("validation R²", labelpad=2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=7, handlelength=1.8, columnspacing=1.5)
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/fig_key.pdf")
fig.savefig("docs/fig_key.png", dpi=200, facecolor="white")
