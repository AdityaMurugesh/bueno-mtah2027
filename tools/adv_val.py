"""Adversarial validation: how well can a classifier tell bench frames (degraded DFL) from BAR-ATH frames?

Label-free, aggregates only. Both sides in both team orientations, every STEP-th in-play-half frame. Grouped CV
(bench: by match, BAR-ATH: by 10-minute block) so neighbouring frames never straddle folds. Prints the AUC and the
most discriminative features with their medians on each side, split by the model's own H/A vs D call
(bench: v-model OOF decode, BAR-ATH: the given submission), so live-play and dead-ball shift show separately.

    MTAH_FEAT=2 MTAH_DEG=4 MTAH_DROP=kick python adv_val.py submissions/v10_d4_set6tta_lgbm8.csv cache/oof_8nk_d4
"""

import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mtah import dataset, decode

STEP = 10
sub, oof = sys.argv[1], sys.argv[2]
lab_c = pd.read_csv(sub)["label"].to_numpy()

Xs, side, grp, dead = [], [], [], []
for sw in (False, True):
    Xc, half, names = dataset.challenge(swapped=sw)
    idx = np.where(half > 0)[0][::STEP]
    Xs.append(Xc[idx]); side.append(np.ones(len(idx))); grp.append(100 + idx // (25 * 600))
    dead.append(lab_c[idx] == "D")
ms = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(oof, "*.npz")))
for k, m in enumerate(ms):
    z = np.load(os.path.join(oof, m + ".npz"))
    pred = decode.decode(z["proba"][:, :3], z["half"], 3, 20)
    for sw in (False, True):
        X, _, half, _, _ = dataset.external(m, dataset.EVAL_SEED, swapped=sw)
        idx = np.where(half > 0)[0][::STEP]
        Xs.append(X[idx]); side.append(np.zeros(len(idx))); grp.append(np.full(len(idx), k)); dead.append(pred[idx] == "D")
X, s, g, dd = np.concatenate(Xs), np.concatenate(side), np.concatenate(grp), np.concatenate(dead)
print("frames bench %d  BAR-ATH %d  features %d" % ((s == 0).sum(), (s == 1).sum(), X.shape[1]))

ug = np.unique(g)
rng = np.random.default_rng(0)
fold_of = dict(zip(ug, rng.permutation(len(ug)) % 5))
f = np.array([fold_of[v] for v in g])
p = np.zeros(len(s)); gain = np.zeros(X.shape[1])
for k in range(5):
    tr, te = f != k, f == k
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1).fit(X[tr], s[tr])
    p[te] = clf.predict_proba(X[te])[:, 1]
    gain += clf.booster_.feature_importance("gain")
print("AUC all %.3f | live %.3f | dead %.3f" % (roc_auc_score(s, p), roc_auc_score(s[~dd], p[~dd]),
                                                roc_auc_score(s[dd], p[dd])))
order = np.argsort(-gain)[:25]
print("%-22s %6s  %9s %9s  %9s %9s" % ("feature", "gain%", "bench_L", "BA_L", "bench_D", "BA_D"))
for i in order:
    def med(mask):
        v = X[mask, i]
        return np.nanmedian(v) if np.isfinite(v).any() else np.nan
    print("%-22s %6.1f  %9.2f %9.2f  %9.2f %9.2f" % (names[i][:22], 100 * gain[i] / gain.sum(),
          med((s == 0) & ~dd), med((s == 1) & ~dd), med((s == 0) & dd), med((s == 1) & dd)))
