"""Grid-search the decoder on out-of-fold IDSSE probabilities (never on BAR-ATH).

    python tune_decode.py --dir cache/oof_seq_v2 --team 3,5,10 --dead 20,80 --dbias 0,0.5
    python tune_decode.py --dirs cache/oof_seq_v1 cache/oof_v2 ...   # averages the probabilities
"""

import argparse
import glob
import itertools
import os

import numpy as np

from mtah import decode

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="*", default=[os.path.join("cache", "oof")])
ap.add_argument("--team", default="3,5,10,15")
ap.add_argument("--dead", default="20,80")
ap.add_argument("--dbias", default="0", help="added to log p(D) before decoding")
ap.add_argument("--beta", default="0", help="weight of the boundary head's log-odds on the switch cost")
args = ap.parse_args()

per_dir = [{os.path.basename(f): np.load(f) for f in glob.glob(os.path.join(d, "*.npz"))} for d in args.dirs]
common = sorted(set.intersection(*[set(p) for p in per_dir]))
oof = []
for name in common:
    z = per_dir[0][name]
    pr = [p[name]["proba"] for p in per_dir]
    proba = np.mean([q[:, :3] for q in pr], 0)
    bnd = [q[:, 3:4] for q in pr if q.shape[1] > 3]
    if bnd:
        proba = np.concatenate([proba, np.mean(bnd, 0)], 1)
    oof.append((proba, z["half"], z["labels"]))
print(len(oof), "matches:", " ".join(n[:-4] for n in common))
grid = lambda s: [float(v) for v in s.split(",")]
rows = []
for ct, cd, db, be in itertools.product(grid(args.team), grid(args.dead), grid(args.dbias), grid(args.beta)):
    s = []
    for proba, half, lab in oof:
        p = proba.copy()
        p[:, 2] *= np.exp(db)
        s.append(decode.score(lab, decode.decode(p, half, ct, cd, beta=be)))
    r = {k: float(np.mean([x[k] for x in s])) for k in s[0]}
    rows.append((ct, cd, db, be, r))
    print("team %5.1f dead %5.1f dbias %+.2f beta %.2f  spellF1 %.4f  macroF1 %.4f  pred/gt %.0f/%.0f  team_acc %.3f" % (
        ct, cd, db, be, r["spellF1"], r["macroF1"], r["n_pred"], r["n_gt"], r["team_acc"]), flush=True)
best = max(rows, key=lambda x: x[4]["spellF1"])
print("best", best[:4], round(best[4]["spellF1"], 4))
