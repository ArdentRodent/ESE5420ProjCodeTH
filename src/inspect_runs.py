import mne, numpy as np, collections, sys
mne.set_log_level('ERROR')
root = sys.argv[1]
for s in ['001','002','003']:
    for r in [1,2,4,8,12]:
        raw = mne.io.read_raw_edf(f'{root}/S{s}/S{s}R{r:02d}.edf', preload=False)
        a = raw.annotations
        c = collections.Counter(a.description)
        durs = {d: np.round(np.mean(a.duration[a.description==d]),2) for d in c}
        print(s, r, raw.info['sfreq'], len(raw.ch_names), raw.n_times, round(raw.n_times/raw.info['sfreq'],1),'s', dict(c), durs)
