"""Ball presence vs tempo, split by state (aggregates only, label-free on BAR-ATH).

BAR-ATH live proxy: frames inside the organisers' in-possession phase tags (independent of our model); also our
submission's H/A vs D split. Bench: true live vs dead on the eval copies. Same tempo feature (all_speed_m75).

    MTAH_FEAT=2 MTAH_DEG=6 MTAH_DROP=kickclock python calib_ball2.py submissions/v12_d6c_set6tta_lgbm8.csv
"""

import sys

import numpy as np
import pandas as pd

from mtah import dataset, idsse, tagging

E = [0, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 99]
ATTACK = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES", "LONG BALL"}


def row(name, X, mask, c):
    s, p = X[mask, c["all_speed_m75"]], X[mask, c["ball_present"]]
    b = np.digitize(s, E) - 1
    cells = ["%5.2f" % p[b == k].mean() if (b == k).sum() > 300 else "   - " for k in range(len(E) - 1)]
    print("%-22s %6d  %s  | all %.2f" % (name, mask.sum(), " ".join(cells), p.mean()))



def curve(X, c, mask=None):
    s, p = X[:, c["all_speed_m75"]], X[:, c["ball_present"]]
    if mask is not None:
        s, p = s[mask], p[mask]
    b = np.digitize(s, E) - 1
    return np.array([p[b == k].mean() if (b == k).sum() > 300 else np.nan for k in range(len(E) - 1)])


def fit(iters=3, matches=("J03WMX", "J03WOH", "J03WQQ")):
    """Fit degrade.ATD_PRESENT7 so that the d7 bench presence-vs-tempo curve matches BAR-ATH's (all frames)."""
    from mtah import degrade, features
    Xc, hc, names = dataset.challenge()
    c = {n: i for i, n in enumerate(names)}
    target = curve(Xc[hc > 0], c)
    centres = np.array([0.25, 0.62, 0.87, 1.12, 1.37, 1.75, 2.25, 2.75, 3.5, 5.0])
    raw = {m: idsse.load(m) for m in matches}
    for it in range(iters + 1):
        cs = []
        for m in matches:
            d = degrade.degrade7(raw[m], seed=dataset.EVAL_SEED * 1000 + idsse.MATCHES.index(m))
            Xb, nb = features.frame_features(d["players"], d["ball"], d["team"], d["half"])
            cb = {n: i for i, n in enumerate(nb)}
            cs.append(curve(Xb[d["half"] > 0], cb))
        bench = np.nanmean(cs, 0)
        print("iter %d table %s" % (it, np.round(degrade.ATD_PRESENT7, 3).tolist()))
        print("   target %s\n   bench  %s" % (np.round(target, 2).tolist(), np.round(bench, 2).tolist()), flush=True)
        if it == iters:
            break
        ok = np.isfinite(target) & np.isfinite(bench)
        ratio = np.interp(degrade.ATD_SPEED, centres[ok], (target[ok] / bench[ok]))
        degrade.ATD_PRESENT7 = np.clip(degrade.ATD_PRESENT7 * ratio, 0.01, 0.98)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fit"]:
        fit()
        sys.exit()
    X, half, names = dataset.challenge()
    c = {n: i for i, n in enumerate(names)}
    lab = pd.read_csv(sys.argv[1])["label"].to_numpy()
    att = np.zeros(len(X), bool)
    for e in tagging.load():
        if e["code"] in ATTACK:
            att[e["start"]:e["end"] + 1] = True
    g = half > 0
    print("%-22s %6s  %s" % ("set", "n", " ".join("%5g" % v for v in E[1:])))
    row("BAR-ATH phase-tag live", X, g & att, c)
    row("BAR-ATH pred live", X, g & np.isin(lab, ["H", "A"]), c)
    row("BAR-ATH pred dead", X, g & (lab == "D"), c)
    row("BAR-ATH all", X, g, c)
    Xs, Ls = [], []
    for m in idsse.MATCHES:
        Xb, _, hb, lb, _ = dataset.external(m, dataset.EVAL_SEED)
        k = hb > 0
        Xs.append(Xb[k]); Ls.append(lb[k])
    Xb, lb = np.concatenate(Xs), np.concatenate(Ls)
    row("bench true live", Xb, np.isin(lb, ["H", "A"]), c)
    row("bench true dead", Xb, lb == "D", c)
    row("bench all", Xb, np.ones(len(Xb), bool), c)
