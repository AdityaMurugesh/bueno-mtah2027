"""Smart Tagging SET PIECES as model input: real tags on BAR-ATH, simulated tags on the DFL bench.

Real tags (BAR-ATH, label-free aggregates only): 59 set pieces, each ~35 s long. Measured against predicted
restarts (v2_gru / v9), a tag starts a median 6.3 s before the restart (q10 3.0 s, q75 10.6 s), with per-type medians
throw-in 6.7, corner 5.8, goal kick 6.6, free kick 11.6, kick-off after goal 3.8 s. Against a typical match's
restart counts, Metrica tags most corners and kick-offs, about 60% of throw-ins and goal kicks and few free kicks.

Simulation (DFL): every dead->alive label transition is a restart; its type comes from the nearest DFL restart event
(within 5 s, else free kick). Each restart is kept with a per-type recall and its tag starts a random delay before
it; a few false tags are added. Recall, delay scale and false-tag rate are redrawn per copy (domain randomisation),
so the model learns "a restart tends to follow a tag start by a few seconds", not an exact offset.
"""

import numpy as np

from . import events, idsse, tagging

FPS = 25
CAP = 60.0  # seconds
GROUPS = ["ThrowIn", "CornerKick", "GoalKick", "FreeKick", "KickOff"]  # Penalty folds into KickOff (both central)
DELAY_S = {"ThrowIn": 6.7, "CornerKick": 5.8, "GoalKick": 6.6, "FreeKick": 11.6, "KickOff": 3.8, "Penalty": 8.0}
RECALL = {"ThrowIn": (0.45, 0.75), "CornerKick": (0.8, 1.0), "GoalKick": (0.45, 0.8), "FreeKick": (0.1, 0.35),
          "KickOff": (0.8, 1.0), "Penalty": (0.5, 1.0)}
NAMES = ["st_since", "st_until"] + ["st_last_" + g for g in GROUPS] + ["st_next_" + g for g in GROUPS]


def _group(kind):
    return GROUPS.index("KickOff" if kind == "Penalty" else kind)


def real():
    """(start frame, group index) of BAR-ATH set-piece tags. Kick Off (Start) opens a half before grading starts
    and has no dead->alive counterpart on the bench, so it is left out on both sides."""
    out = []
    for e in tagging.load():
        t = e["tags"].get("type", "")
        if e["code"] == "SET PIECES" and t in events.TAG_TYPE and t != "Kick Off (Start)":
            out.append((e["start"], _group(events.TAG_TYPE[t])))
    out.sort()
    return np.array([s for s, _ in out], np.int64), np.array([g for _, g in out], np.int8)


def simulate(match, labels, seed):
    """Simulated tag starts for one copy of a DFL match (labels are the copy's own, so swapped copies work)."""
    rng = np.random.default_rng(10_000 + seed * 100 + idsse.MATCHES.index(match))
    alive = np.isin(labels, ["H", "A"])
    up = np.flatnonzero(alive[1:] & (labels[:-1] == "D")) + 1
    ef, ek = events.restarts(match)
    recall = {k: rng.uniform(*r) for k, r in RECALL.items()}
    scale = rng.uniform(0.8, 1.25)
    sigma = rng.uniform(0.3, 0.55)
    starts, groups = [], []
    for f in up:
        j = np.searchsorted(ef, f)
        near = [i for i in (j - 1, j) if 0 <= i < len(ef) and abs(ef[i] - f) <= 5 * FPS]
        kind = events.TYPES[ek[min(near, key=lambda i: abs(ef[i] - f))]] if near else "FreeKick"
        if rng.random() > recall[kind]:
            continue
        delay = max(0.5, DELAY_S[kind] * scale * np.exp(rng.normal(0, sigma)))
        starts.append(f - int(round(delay * FPS))); groups.append(_group(kind))
    graded = np.flatnonzero(labels != "X")
    for _ in range(rng.poisson(rng.uniform(0.0, 0.12) * len(starts))):
        starts.append(int(rng.choice(graded))); groups.append(int(rng.integers(len(GROUPS))))
    order = np.argsort(starts)
    return np.array(starts, np.int64)[order], np.array(groups, np.int8)[order]


def features(starts, groups, n):
    """(n, 12): seconds since the last / until the next tag start (capped), and the type of each."""
    t = np.arange(n)
    X = np.zeros((n, len(NAMES)), np.float32)
    X[:, 0] = X[:, 1] = CAP
    if len(starts) == 0:
        return X
    j = np.searchsorted(starts, t, side="right")  # starts[j-1] <= t < starts[j]
    has_last, has_next = j > 0, j < len(starts)
    last = np.where(has_last, starts[np.maximum(j - 1, 0)], 0)
    nxt = np.where(has_next, starts[np.minimum(j, len(starts) - 1)], 0)
    since = np.where(has_last, (t - last) / FPS, CAP)
    until = np.where(has_next, (nxt - t) / FPS, CAP)
    X[:, 0] = np.minimum(since, CAP)
    X[:, 1] = np.minimum(until, CAP)
    k = len(GROUPS)
    ok = has_last & (since < CAP)
    X[np.flatnonzero(ok), 2 + groups[np.maximum(j - 1, 0)][ok]] = 1
    ok = has_next & (until < CAP)
    X[np.flatnonzero(ok), 2 + k + groups[np.minimum(j, len(starts) - 1)][ok]] = 1
    return X
