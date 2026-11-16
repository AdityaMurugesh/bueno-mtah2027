"""Feature matrices for the external matches (degraded) and for the challenge match, cached."""

import os

import numpy as np

from . import atd, degrade, features, idsse, settags

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_VERSION = int(os.environ.get("MTAH_FEAT", "2"))
DEG_VERSION = int(os.environ.get("MTAH_DEG", "2"))  # degrade.degrade (1), degrade2 (2) or degrade3 (3)
# The held-out evaluation copies (seed 99) can use a different degradation, to test how a model trained under
# one noise model holds up under another (train d2 / evaluate d3 and the reverse).
EVAL_SEED = 99
EVAL_DEG_VERSION = int(os.environ.get("MTAH_EVAL_DEG", str(DEG_VERSION)))
CACHE = os.path.join(HERE, "..", "cache", "feat" if FEAT_VERSION == 1 else "feat_v{}".format(FEAT_VERSION))
EXT_CACHE = CACHE + ("_d{}".format(DEG_VERSION) if DEG_VERSION != 1 else "")


def deg_for(seed):
    return EVAL_DEG_VERSION if seed == EVAL_SEED else DEG_VERSION


def ext_cache(base, seed):
    v = deg_for(seed)
    return base + ("_d{}".format(v) if v != 1 else "")


def degrade_fn(seed=None):
    return {1: degrade.degrade, 2: degrade.degrade2, 3: degrade.degrade3, 4: degrade.degrade4, 5: degrade.degrade5, 6: degrade.degrade6, 7: degrade.degrade7, 8: degrade.degrade8}[deg_for(seed)]
CODE = {"H": 0, "A": 1, "D": 2, "X": -1}

# Feature columns to leave out (MTAH_DROP=kick|touch). Adversarial validation (adv.py) showed the kick
# features never fire on the smoothed ATD (q10 at the cap) but fire often on degraded copies.
KICK = ["h_since_kick", "h_until_kick", "a_since_kick", "a_until_kick"]
TOUCH = ["ball_dv", "h_touch", "h_since_touch", "h_until_touch", "h_touch_3s", "a_touch", "a_since_touch",
         "a_until_touch", "a_touch_3s", "last_touch_home", "next_touch_home"] + KICK
DROP = {"": [], "kick": KICK, "touch": TOUCH, "kickclock": KICK + ["half_clock"]}[os.environ.get("MTAH_DROP", "")]
# MTAH_TAGS=1 appends Smart Tagging set-piece features: real tags on BAR-ATH, simulated ones on the bench (settags.py)
TAGS = os.environ.get("MTAH_TAGS", "") == "1"


def _tags(X, names, starts_groups):
    if not TAGS:
        return X, names
    return np.hstack([X, settags.features(*starts_groups, len(X))]), names + settags.NAMES


def _drop(X, names):
    names = [str(n) for n in names]
    if not DROP:
        return X, names
    keep = [i for i, n in enumerate(names) if n not in DROP]
    return X[:, keep], [names[i] for i in keep]


def external(match, seed, swapped=False, strength=1.0, refresh=False):
    """(X, y, half, labels) for one degraded copy of an IDSSE match."""
    tag = "{}_s{}_{}_k{}".format(match, seed, "sw" if swapped else "or", strength)
    path = os.path.join(ext_cache(CACHE, seed), tag + ".npz")
    if os.path.exists(path) and not refresh:
        z = np.load(path)
        X, names = _drop(z["X"], z["names"])
        if TAGS:
            X, names = _tags(X, names, settags.simulate(match, z["labels"], seed))
        return X, z["y"], z["half"], z["labels"], names
    d = idsse.load(match)
    if swapped:
        d = degrade.swap(d)
    d = degrade_fn(seed)(d, seed=seed * 1000 + idsse.MATCHES.index(match), strength=strength)
    X, names = features.frame_features(d["players"], d["ball"], d["team"], d["half"])
    y = np.array([CODE[c] for c in d["labels"]], np.int8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=X, y=y, half=d["half"], labels=d["labels"], names=np.array(names))
    X, names = _drop(X, names)
    if TAGS:
        X, names = _tags(X, names, settags.simulate(match, d["labels"], seed))
    return X, y, d["half"], d["labels"], names


def challenge(refresh=False, swapped=False):
    """(X, half, names) for the BAR-ATH tactical ATD. half = 1/2 on graded frames, 0 elsewhere.

    swapped: home and away roles exchanged (test-time augmentation; H/A of the output must be swapped back)."""
    path = os.path.join(CACHE, "BAR-ATH_sw.npz" if swapped else "BAR-ATH.npz")
    if os.path.exists(path) and not refresh:
        z = np.load(path)
        X, names = _drop(z["X"], z["names"])
        if TAGS:
            X, names = _tags(X, names, settags.real())
        return X, z["half"], names
    d = atd.load()
    if swapped:
        d = {**d, "team": 1 - d["team"]}
    n = len(d["ball"])
    half = np.zeros(n, np.int8)
    half[11185:90652] = 1
    half[110808:190198] = 2
    X, names = features.frame_features(d["players"], d["ball"], d["team"], half)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, X=X, half=half, names=np.array(names))
    X, names = _drop(X, names)
    if TAGS:
        X, names = _tags(X, names, settags.real())
    return X, half, names
