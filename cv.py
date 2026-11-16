"""Leave-one-match-out CV on the degraded IDSSE matches. Writes out-of-fold probabilities.

    python cv.py                 # train, save OOF probabilities to cache/oof
    python cv.py --baseline      # also score the organisers' baseline on the same bench
"""

import argparse
import os
import time

import lightgbm as lgb
import numpy as np

from mtah import dataset, decode, idsse

OOF = os.path.join("cache", "oof")
TRAIN_SEEDS = tuple(int(v) for v in os.environ.get("MTAH_LGBM_SEEDS", "0,1").split(","))
EVAL_SEED = 99
STRIDE = 3
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, num_threads=int(os.environ.get("MTAH_THREADS", "8")))
ROUNDS = 400


STRENGTH = 1.0


def training_set(matches):
    Xs, ys = [], []
    for m in matches:
        for s in TRAIN_SEEDS:
            for sw in (False, True):
                X, y, half, lab, names = dataset.external(m, s, sw, strength=STRENGTH)
                keep = np.flatnonzero(y >= 0)[::STRIDE]
                Xs.append(X[keep]); ys.append(y[keep])
    return np.concatenate(Xs), np.concatenate(ys), names


def baseline_labels(X, names, half, window=50):
    """Organisers' baseline on our features: nearest team when the ball is seen, D otherwise."""
    present = X[:, names.index("ball_present")] > 0
    diff = X[:, names.index("dball_diff")]
    code = np.full(len(X), -1)
    code[present & (diff < 0)] = 0
    code[present & (diff >= 0)] = 1
    hc = np.concatenate(([0], np.cumsum(code == 0))); ac = np.concatenate(([0], np.cumsum(code == 1)))
    i = np.arange(len(X)); lo = np.maximum(0, i - window); hi = np.minimum(len(X), i + window + 1)
    h = hc[hi] - hc[lo]; a = ac[hi] - ac[lo]
    lab = np.where(h + a == 0, "D", np.where(h >= a, "H", "A"))
    lab[half == 0] = "X"
    return lab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--matches", nargs="*", default=idsse.MATCHES)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--oof", default=OOF)
    args = ap.parse_args()
    STRENGTH = args.strength
    OOF = args.oof
    os.makedirs(OOF, exist_ok=True)
    for held in args.matches:
        t0 = time.time()
        Xtr, ytr, names = training_set([m for m in idsse.MATCHES if m != held])
        model = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr), ROUNDS)
        X, y, half, lab, _ = dataset.external(held, EVAL_SEED, strength=STRENGTH)
        proba = model.predict(X)
        np.savez(os.path.join(OOF, held + ".npz"), proba=proba, half=half, labels=lab)
        line = "{} trained on {} rows in {:.0f}s".format(held, len(ytr), time.time() - t0)
        pred = decode.decode(proba, half, 3, 80)
        line += "  model+viterbi(3,80) {}".format({k: round(v, 3) for k, v in decode.score(lab, pred).items()})
        if args.baseline:
            line += "\n    baseline50 {}".format(
                {k: round(v, 3) for k, v in decode.score(lab, baseline_labels(X, names, half)).items()})
        print(line, flush=True)
        if held == args.matches[0]:
            imp = model.feature_importance("gain")
            top = np.argsort(imp)[::-1][:25]
            print("  top features:", [names[i] for i in top])
