"""Paired per-match spellF1 difference between OOF dirs (same eval copies), Viterbi costs 3/20.

    python paired.py cache/oof_A cache/oof_B [cache/oof_C cache/oof_D]   # A-B (and C-D)
"""

import glob
import os
import sys

import numpy as np

from mtah import decode


def per(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        out[os.path.basename(f)] = decode.score(z["labels"], decode.decode(z["proba"][:, :3], z["half"], 3, 20))["spellF1"]
    return out


args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    A, B = per(a), per(b)
    ks = sorted(set(A) & set(B))
    d = [A[k] - B[k] for k in ks]
    print("%s %.4f - %s %.4f (%d matches): %s  mean %+.4f" % (os.path.basename(a), np.mean([A[k] for k in ks]),
          os.path.basename(b), np.mean([B[k] for k in ks]), len(ks), " ".join("%+.3f" % v for v in d), np.mean(d)))
