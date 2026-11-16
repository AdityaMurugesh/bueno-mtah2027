"""Compare label-free statistics of the challenge ATD with degraded external copies (aggregates only).

    python calib.py --deg 2 --matches J03WMX J03WN1
"""

import argparse

import numpy as np

from mtah import atd, dataset, degrade, features, idsse

BINS = [0, 0.6, 0.9, 1.2, 1.5, 2.0, 2.5, 3.5, 99]


def runs(mask):
    """Lengths (s) of True runs."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) / 25.0


def report(name, X, names, P, B, g):
    names = list(names)
    pres = X[g, names.index("ball_present")] > 0.5
    sp = X[g, names.index("all_speed_m75")]
    idx = np.digitize(sp, BINS) - 1
    curve = " ".join("%.2f/%2d%%" % (pres[idx == b].mean() if (idx == b).any() else np.nan, 100 * (idx == b).mean())
                     for b in range(len(BINS) - 1))
    mr, pr = runs(~pres), runs(pres)
    Pg, Bg = P[g][..., :2], B[g][:, :2]
    d2p = Pg[2:] - 2 * Pg[1:-1] + Pg[:-2]
    d2b = Bg[2:] - 2 * Bg[1:-1] + Bg[:-2]
    jp = np.nanpercentile(np.hypot(d2p[..., 0], d2p[..., 1]), [50, 90])
    jb = np.nanpercentile(np.hypot(d2b[:, 0], d2b[:, 1]), [50, 90, 99])
    kick = (X[g, names.index("h_until_kick")] < features.CAP).mean()
    print("%-14s present %.2f | miss run med %4.1f p90 %4.1f | pres run med %4.1f p90 %4.1f | kick-near %.2f" % (
        name, pres.mean(), np.median(mr), np.percentile(mr, 90), np.median(pr), np.percentile(pr, 90), kick))
    print("   P(present)/%%frames by speed %s" % curve)
    print("   jerk players %.4f %.4f  ball %.4f %.4f %.3f" % (*jp, *jb))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, default=2)
    ap.add_argument("--matches", nargs="*", default=idsse.MATCHES[:2])
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    d = atd.load()
    Xc, half, names = dataset.challenge()
    report("ATD", Xc, names, d["players"], d["ball"], half > 0)
    fn = {1: degrade.degrade, 2: degrade.degrade2, 3: degrade.degrade3, 4: degrade.degrade4, 5: degrade.degrade5,
          6: degrade.degrade6, 7: degrade.degrade7, 8: degrade.degrade8}[args.deg]
    for m in args.matches:
        e = fn(idsse.load(m), seed=args.seed)
        X, names = features.frame_features(e["players"], e["ball"], e["team"], e["half"])
        report(m + " d%d" % args.deg, X, names, e["players"], e["ball"], e["half"] > 0)
