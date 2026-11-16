"""Where does the bench stop looking like BAR-ATH at restarts? Same procedure on both sides, aggregates only.

Window = the first 2.5 s after a set-piece tag start: real Smart Tagging on BAR-ATH, simulated tags (same timing
statistics) on the bench's held-out d3 copies. On the bench those frames are ~90% dead (printed), and BAR-ATH
should be similar. Compares tracking features in those windows and how often the model calls them dead.

    MTAH_DEG=3 MTAH_DROP=kick python restart_cmp.py submissions/v9_d3_set6tta_lgbm8.csv cache/oof_8nk_d3 ...
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

from mtah import dataset, decode, settags

FEATS = ["ball_present", "ball_since_seen", "ball_speed", "ball_out_dist", "all_speed", "h_dball", "a_dball",
         "n_tracked", "ball_present_m75", "all_speed_m75"]
W = int(2.5 * 25)
OFF = int(float(os.environ.get("RC_OFF", "0")) * 25)  # window start relative to the tag start, seconds


def windows(starts, n, half):
    idx = np.concatenate([np.arange(s + OFF, min(s + OFF + W, n)) for s in starts if 0 <= s + OFF < n])
    return idx[half[idx] > 0]


sub, dirs = sys.argv[1], sys.argv[2:]
Xc, half_c, names = dataset.challenge()
col = {n: i for i, n in enumerate(names)}
s, g = settags.real()
lab_c = pd.read_csv(sub)["label"].to_numpy()
rows = {}
for grp in range(len(settags.GROUPS)):
    idx = windows(s[g == grp], len(Xc), half_c)
    rows[("BAR-ATH", settags.GROUPS[grp])] = (Xc[idx], np.mean(lab_c[idx] == "D"), np.nan, len(idx) // W)
idx = windows(s, len(Xc), half_c)
rows[("BAR-ATH", "all")] = (Xc[idx], np.mean(lab_c[idx] == "D"), np.nan, len(idx) // W)

per = [{os.path.basename(f): np.load(f) for f in glob.glob(os.path.join(d, "*.npz"))} for d in dirs]
bench = {k: [] for k in settags.GROUPS + ["all"]}
for m in sorted(os.path.basename(f)[:5 + 1] for f in glob.glob(os.path.join(dataset.ext_cache(dataset.CACHE, 99), "*_s99_or_*"))):
    m = m.rstrip("_")
    if not all(m + ".npz" in p for p in per):
        continue
    X, y, half, lab, _ = dataset.external(m, dataset.EVAL_SEED)
    proba = np.mean([p[m + ".npz"]["proba"][:, :3] for p in per], 0)
    pred = decode.decode(proba, half, 3, 20)
    st, sg = settags.simulate(m, lab, dataset.EVAL_SEED)
    for grp in range(len(settags.GROUPS)):
        idx = windows(st[sg == grp], len(X), half)
        bench[settags.GROUPS[grp]].append((X[idx], pred[idx] == "D", lab[idx] == "D"))
    idx = windows(st, len(X), half)
    bench["all"].append((X[idx], pred[idx] == "D", lab[idx] == "D"))
for k, v in bench.items():
    if v:
        X = np.concatenate([a for a, _, _ in v])
        rows[("bench", k)] = (X, np.concatenate([b for _, b, _ in v]).mean(), np.concatenate([c for _, _, c in v]).mean(),
                              len(X) // W)

print("%-8s %-10s %4s %6s %6s  " % ("side", "type", "n", "predD", "trueD") + " ".join("%10s" % f[:10] for f in FEATS))
for (side, k), (X, pd_, td, n) in sorted(rows.items(), key=lambda r: (r[0][1], r[0][0])):
    med = [np.nanmean(X[:, col[f]]) if f.startswith("ball_present") else np.nanmedian(X[:, col[f]]) for f in FEATS]
    print("%-8s %-10s %4d %6.2f %6.2f  " % (side, k, n, pd_, td) + " ".join("%10.2f" % v for v in med))
