"""Label-free calibration of player tracking loss vs tempo (aggregates only).

Prints, per collective-speed bin (6 s mean of tracked players' speed, the all_speed_m75 feature), quantiles of
the number of missing players (22 - n_tracked), plus run lengths of heavy-loss stretches. Run on BAR-ATH
(the ATD) and on a bench copy to compare.

    MTAH_FEAT=2 MTAH_DEG=5 python calib_players.py [match ...]
"""

import sys

import numpy as np

from mtah import dataset

E = [0, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 99]
Q = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]


def runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return (e[1::2] - e[::2]) / 25.0


def report(name, X, half, c):
    g = half > 0
    s, n = X[g, c["all_speed_m75"]], X[g, c["n_tracked"]]
    miss = 22 - n
    b = np.digitize(s, E) - 1
    print(name, " q:", " ".join("%4g" % q for q in Q))
    for k in range(len(E) - 1):
        m = miss[b == k]
        print("  %4.2f-%-4g %.3f" % (E[k], E[k + 1], (b == k).mean()), " ".join("%4d" % v for v in np.quantile(m, Q)),
              " mean %.2f" % m.mean())
    for thr in (4, 8):
        r = runs(miss >= thr)
        print("  runs missing>=%d: n %d  median %.1f s  p90 %.1f s  share %.3f" % (thr, len(r), np.median(r),
                                                                                np.percentile(r, 90), (miss >= thr).mean()))


if __name__ == "__main__":
    X, half, names = dataset.challenge()
    c = {n: i for i, n in enumerate(names)}
    report("BAR-ATH", X, half, c)
    for m in sys.argv[1:]:
        X, _, half, _, _ = dataset.external(m, dataset.EVAL_SEED)
        report("bench " + m, X, half, c)
