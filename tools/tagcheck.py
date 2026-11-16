"""Full-match proxy check of a submission against the organisers' Smart Tagging (an input, not ground truth).

The public 5-minute window holds only 4 dead stretches; Smart Tagging covers all 90 minutes:
  - SET PIECES (59 restarts): share of frames labelled D at offsets around each tagged start
  - team phases: in-possession tags (build up, progression, ...) name the team with the ball, defending tags
    the team without it; we report how often the alive labels agree with that team
Prints aggregates only.

    python tagcheck.py submissions/v2_gru.csv submissions/v5_gru_nk.csv
"""

import sys

import numpy as np
import pandas as pd

from mtah import tagging

HOME = "Barcelona"
ATTACK = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES", "LONG BALL"}
DEFEND = {"DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN ATTACKING THIRD",
          "DEFENSIVE TRANSITION"}
OFFS = [-10, -6, -3, -1, 0, 1, 3, 6, 10, 20]

ev = tagging.load()
sp = [e["start"] for e in ev if e["code"] == "SET PIECES"]
print("%-16s %s | %-10s %-10s %s" % ("submission", " ".join("%+4ds" % o for o in OFFS), "att agree", "def agree",
                                     "D share"))
for path in sys.argv[1:]:
    lab = pd.read_csv(path)["label"].to_numpy()
    n = len(lab)
    prof = []
    for o in OFFS:
        idx = [s + 25 * o for s in sp if 0 <= s + 25 * o < n and lab[s + 25 * o] != "X"]
        prof.append(np.mean(lab[idx] == "D") if idx else np.nan)
    agree = {}
    for kind, codes in (("att", ATTACK), ("def", DEFEND)):
        hit = tot = 0
        for e in ev:
            if e["code"] not in codes or e["tags"].get("team") in (None, "N/A"):
                continue
            seg = lab[e["start"]:e["end"] + 1]
            seg = seg[(seg == "H") | (seg == "A")]
            want = "H" if (e["tags"]["team"] == HOME) == (kind == "att") else "A"
            hit += (seg == want).sum(); tot += len(seg)
        agree[kind] = hit / max(tot, 1)
    g = lab[lab != "X"]
    print("%-16s %s | %-10.3f %-10.3f %.3f" % (path.split("/")[-1][:16], " ".join("%5.2f" % p for p in prof),
                                               agree["att"], agree["def"], np.mean(g == "D")))
