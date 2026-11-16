"""Teammate's kickoff rule (football fact, not tuned): the ball is dead from the start of each graded half
up to that half's Smart Tagging "Kick Off (Start)" tag.

    python kickoff_rule.py submissions/v13_cpu.csv submissions/v14_cpu_kick.csv
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repo")
sys.path.insert(0, os.path.join(REPO, "starter")); sys.path.insert(0, REPO)
import load_smart_tagging  # noqa: E402

GRADED_HALVES = ((11185, 90652), (110808, 190198))  # [start, stop) video frames, DATA.md

lab = pd.read_csv(sys.argv[1])["label"].values.astype("<U1")
starts = sorted(e["start_frame"] for e in load_smart_tagging.load()
                if (e["tags"].get("Type") or "").lower() == "kick off (start)")
for lo, hi in GRADED_HALVES:
    k = [s for s in starts if lo <= s < hi]
    if k:
        print("half [%d, %d): %d frames before the kickoff tag set to D" % (lo, hi, int((lab[lo:k[0]] != "D").sum())))
        lab[lo:k[0]] = "D"
with open(sys.argv[2], "w", newline="\n") as f:
    f.write("frame,label\n")
    f.writelines("{},{}\n".format(i, l) for i, l in enumerate(lab))
print("wrote", sys.argv[2], "sha256", hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest())
