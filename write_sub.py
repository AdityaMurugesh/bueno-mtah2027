"""Decode saved BAR-ATH probabilities into a submission CSV. Several files are averaged.

    python write_sub.py cache/bar_ath_proba_seq.npy 10 80 submissions/v2_gru.csv
    python write_sub.py cache/a.npy,cache/b.npy 3 80 submissions/v3_ens.csv
    python write_sub.py cache/a.npy 10 80 submissions/x.csv 1.0   # 5th arg: boundary weight beta
"""
import sys

import numpy as np

from mtah import dataset, decode

pr = [np.load(p) for p in sys.argv[1].split(",")]
proba = np.mean([q[:, :3] for q in pr], 0)
bnd = [q[:, 3:4] for q in pr if q.shape[1] > 3]
if bnd:
    proba = np.concatenate([proba, np.mean(bnd, 0)], 1)
beta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
_, half, _ = dataset.challenge()
labels = decode.decode(proba, half, float(sys.argv[2]), float(sys.argv[3]), beta=beta)
with open(sys.argv[4], "w", newline="\n") as f:
    f.write("frame,label\n")
    f.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
g = half > 0
print("wrote", sys.argv[4], "graded share H/A/D", [round(float(np.mean(labels[g] == c)), 3) for c in "HAD"])
