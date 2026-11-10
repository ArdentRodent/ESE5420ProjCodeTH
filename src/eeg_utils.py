import os
import subprocess

BASE = "https://physionet.org/files/eegmmidb/1.0.0"
RUNS = (4, 8, 12)
MIN_BYTES = 1_000_000  # a real run is ~2.5 MB; anything smaller is an error page or partial file


def download(data_dir="data_eeg", subjects=range(1, 110), runs=RUNS):
    """Download EDF files, skipping any already present and complete."""
    n_new = n_skip = 0
    for s in subjects:
        sid = f"S{s:03d}"
        os.makedirs(f"{data_dir}/{sid}", exist_ok=True)
        for r in runs:
            fn = f"{sid}R{r:02d}.edf"
            path = f"{data_dir}/{sid}/{fn}"
            if os.path.exists(path) and os.path.getsize(path) > MIN_BYTES:
                n_skip += 1
                continue
            tmp = path + ".part"
            # timeouts + retries so one stalled connection can't hang the loop
            subprocess.run(
                ["curl", "-sfL", "--connect-timeout", "20", "--max-time", "120",
                 "--retry", "5", "--retry-all-errors", "-o", tmp, f"{BASE}/{sid}/{fn}"],
                check=True,
            )
            os.replace(tmp, path)  # only a finished download gets the real name
            n_new += 1
    print(f"downloaded {n_new}, skipped {n_skip} already present")


import mne
import pandas as pd
mne.set_log_level("ERROR")

def qc(data_dir="data_eeg", subjects=range(1, 110), runs=RUNS,
       sfreq=160.0, n_ch=64, n_times=19680):
    """One row per file; a subject is kept only if all its runs pass."""
    rows = []
    for s in subjects:
        for r in runs:
            path = f"{data_dir}/S{s:03d}/S{s:03d}R{r:02d}.edf"
            raw = mne.io.read_raw_edf(path, preload=False)
            reasons = []
            if raw.info["sfreq"] != sfreq:      reasons.append(f"sfreq={raw.info['sfreq']}")
            if len(raw.ch_names) != n_ch:       reasons.append(f"n_ch={len(raw.ch_names)}")
            if raw.n_times < n_times:           reasons.append(f"n_times={raw.n_times}")
            rows.append(dict(subject=s, run=r, sfreq=raw.info["sfreq"],
                             n_ch=len(raw.ch_names), n_times=raw.n_times,
                             reason="; ".join(reasons)))
    df = pd.DataFrame(rows)
    bad = df.groupby("subject")["reason"].apply(lambda x: (x != "").any())
    return df, sorted(bad[~bad].index), df[df.reason != ""]

import numpy as np

def load_run(s, r, data_dir="data_eeg", n_times=19680):
    """(T, 64) array: causal 1-40 Hz band-pass, trimmed to n_times samples."""
    raw = mne.io.read_raw_edf(f"{data_dir}/S{s:03d}/S{s:03d}R{r:02d}.edf", preload=True)
    raw.filter(1, 40, phase="minimum")   # causal: filtered value at t uses only samples <= t
    return raw.get_data()[:, :n_times].T

def load_subject(s, runs=(4, 8), data_dir="data_eeg"):
    """z-score every run with run 4's mean/std. Run 12 excluded by default."""
    X = {r: load_run(s, r, data_dir) for r in runs}
    mu, sd = X[4].mean(0), X[4].std(0)
    return {r: (X[r] - mu) / sd for r in runs}