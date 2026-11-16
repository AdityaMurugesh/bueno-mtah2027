"""IDSSE (DFL Bundesliga open data) -> the same arrays as mtah.atd, plus frame labels.

Seven matches with clean optical tracking and per-frame BallStatus / BallPossession -- the same
label definition as the challenge (alive flag + owning team). Used ONLY as an external training and
validation set; nothing here touches the competition match.

Per match, cached as npz:
    players (n, slots, 3)  x, y in metres from the pitch corner (0..105, 0..68), speed m/s
    ball    (n, 3)
    team    (slots,)       0 home, 1 away
    labels  (n,)           'H' 'A' 'D', or 'X' in the gap between halves
    half    (n,)           1 / 2, 0 in the gap
"""

import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "ext", "idsse")
CACHE = os.path.join(HERE, "..", "cache", "idsse")
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
GAP = 250  # X frames inserted between halves so no possession spans halftime

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)"(?: Z="[-\d.]+")?[^>]* S="([-\d.]+)"'
                   r'(?:[^>]*BallPossession="(\d)" BallStatus="(\d)")?')


def _file(kind, match):
    return glob.glob(os.path.join(EXT, "DFL_{}_*{}.xml".format(kind, match)))[0]


def _home_away(match):
    root = ET.parse(_file("02_01_matchinformation", match)).getroot()
    roles = {}
    for team in root.iter("Team"):
        roles[team.get("Role")] = team.get("TeamId")
    return roles["home"], roles["guest"]


def parse(match):
    home, away = _home_away(match)
    sets = {}  # (section, person) -> (team, list of rows)
    current = None
    with open(_file("04_03_positions_raw_observed", match), encoding="utf-8") as handle:
        for line in handle:
            if "<Frame " in line:
                m = FRAME.search(line)
                current[1].append(m.groups())
            elif "<FrameSet" in line:
                section, team, person = FRAMESET.search(line).groups()
                current = (team, [])
                sets[(section, person)] = current
    halves = []
    for section in ("firstHalf", "secondHalf"):
        ball = [v for (s, p), v in sets.items() if s == section and v[0] == "BALL"][0][1]
        n0 = int(ball[0][0]); n = int(ball[-1][0]) - n0 + 1
        b = np.full((n, 3), np.nan, np.float32); lab = np.full(n, "X", "<U1")
        for row in ball:
            i = int(row[0]) - n0
            b[i] = float(row[1]), float(row[2]), float(row[3])
            if row[5] is not None:
                lab[i] = "D" if row[5] == "0" else ("H" if row[4] == "1" else "A")
        people = [(p, v) for (s, p), v in sets.items() if s == section and v[0] in (home, away)]
        P = np.full((n, len(people), 3), np.nan, np.float32)
        team = np.array([0 if v[0] == home else 1 for p, v in people], np.int8)
        ids = [p for p, v in people]
        for k, (p, v) in enumerate(people):
            rows = np.array([[float(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in v[1]])
            idx = rows[:, 0].astype(int) - n0
            ok = (idx >= 0) & (idx < n)
            P[idx[ok], k] = rows[ok, 1:]
        halves.append((P, b, team, lab, ids))
    # merge halves onto one timeline with a common slot list (person ids)
    all_ids = sorted(set(halves[0][4]) | set(halves[1][4]))
    slot = {p: i for i, p in enumerate(all_ids)}
    team = np.zeros(len(all_ids), np.int8)
    total = sum(len(h[1]) for h in halves) + GAP
    players = np.full((total, len(all_ids), 3), np.nan, np.float32)
    ballall = np.full((total, 3), np.nan, np.float32)
    labels = np.full(total, "X", "<U1"); half = np.zeros(total, np.int8)
    start = 0
    for h, (P, b, t, lab, ids) in enumerate(halves, 1):
        n = len(b)
        cols = [slot[p] for p in ids]
        players[start:start + n, cols] = P
        team[cols] = t
        ballall[start:start + n] = b; labels[start:start + n] = lab; half[start:start + n] = h
        start += n + GAP
    # DFL coordinates are centred; shift to the corner origin used by the ATD
    for arr in (players[..., 0], ballall[:, 0]):
        arr += 52.5
    for arr in (players[..., 1], ballall[:, 1]):
        arr += 34.0
    return {"players": players, "ball": ballall, "team": team, "labels": labels, "half": half}


def load(match, refresh=False):
    path = os.path.join(CACHE, match + ".npz")
    if os.path.exists(path) and not refresh:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    d = parse(match)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **d)
    return d


if __name__ == "__main__":
    import sys
    for m in (sys.argv[1:] or MATCHES):
        d = load(m, refresh=True)
        lab = d["labels"]; g = lab != "X"
        segs = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        bounds = np.concatenate(([0], segs, [len(lab)]))
        runs = [(lab[a], b - a) for a, b in zip(bounds[:-1], bounds[1:])]
        poss = [n for l, n in runs if l in "HA"]
        print(m, "frames", g.sum(), "share H/A/D", np.round([np.mean(lab[g] == c) for c in "HAD"], 3),
              "possessions", len(poss), "mean %.1fs" % (np.mean(poss) / 25),
              "<2s: %.2f" % np.mean(np.array(poss) < 50), "players/frame %.1f" % np.mean(np.sum(~np.isnan(d["players"][g, :, 0]), 1)),
              flush=True)
