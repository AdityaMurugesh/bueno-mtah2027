"""Per-player token tensors for the set encoder (seq_set.py), cached next to the hand features.

Same degraded copies as dataset.external (same seeds), same direction normalisation as features.py.
Tokens per frame and slot: x, y, speed (NaN when the slot is not tracked); slot team; ball x, y.
"""

import os

import numpy as np

from . import atd, dataset, degrade, features, idsse

K = 48  # slots, padded (ATD has 46)
CACHE = os.path.join(dataset.HERE, "..", "cache", "raw")
EXT_CACHE = CACHE + ("_d{}".format(dataset.DEG_VERSION) if dataset.DEG_VERSION != 1 else "")


def _pack(d):
    P, B = features.normalise(d["players"], d["ball"], d["team"], d["half"])
    n, k = P.shape[:2]
    xy = np.full((n, K, 2), np.nan, np.float16)
    xy[:, :k] = P
    sp = np.full((n, K), np.nan, np.float16)
    v = features._velocity(P, 5)  # (n, k, 2) m/s
    sp[:, :k] = np.hypot(v[..., 0], v[..., 1])
    team = np.full(K, -1, np.int8)
    team[:k] = d["team"]
    return xy, sp, team, B.astype(np.float16)


def external(match, seed, swapped=False):
    path = os.path.join(dataset.ext_cache(CACHE, seed), "{}_s{}_{}.npz".format(match, seed, "sw" if swapped else "or"))
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d = idsse.load(match)
    if swapped:
        d = degrade.swap(d)
    d = dataset.degrade_fn(seed)(d, seed=seed * 1000 + idsse.MATCHES.index(match))
    xy, sp, team, B = _pack(d)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, xy=xy, sp=sp, team=team, ball=B)
    return {"xy": xy, "sp": sp, "team": team, "ball": B}


def challenge(swapped=False):
    path = os.path.join(CACHE, "BAR-ATH_sw.npz" if swapped else "BAR-ATH.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d = atd.load()
    if swapped:
        d = {**d, "team": 1 - d["team"]}
    _, half, _ = dataset.challenge()
    xy, sp, team, B = _pack({**d, "half": half})
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, xy=xy, sp=sp, team=team, ball=B)
    return {"xy": xy, "sp": sp, "team": team, "ball": B}
