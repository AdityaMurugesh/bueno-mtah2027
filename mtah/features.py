"""Per-frame features from team-level tracking. Same code for the challenge ATD and for IDSSE.

Inputs (see mtah.atd / mtah.idsse): players (n, slots, 3), ball (n, 3), team (slots,), half (n,).
Only positions are used; speeds are recomputed from positions so both sources agree.

Coordinates are normalised per half so that the HOME team always attacks towards +x. Everything is
expressed home-vs-away; mirror() swaps the roles for augmentation, which keeps the model symmetric.
"""

import numpy as np
import pandas as pd

L, W = 105.0, 68.0
FPS = 25
CONTACT_RADII = (1.5, 3.0)
CAP = 30 * FPS  # cap on time-since / time-until features


def attack_sign(players, team, half):
    """+1 if home attacks +x in that half, else -1, from the median team centroids."""
    signs = {}
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        cx = [np.nanmedian(np.nanmean(players[idx][:, team == t, 0], axis=1)) for t in (0, 1)]
        signs[int(h)] = 1.0 if cx[0] < cx[1] else -1.0
    return signs


def normalise(players, ball, team, half):
    """Flip x (and y, to keep handedness) on frames where home attacks -x."""
    signs = attack_sign(players, team, half)
    P = players[..., :2].copy()
    B = ball[:, :2].copy()
    flip = np.zeros(len(ball), bool)
    for h, s in signs.items():
        if s < 0:
            flip |= half == h
    P[flip, :, 0] = L - P[flip, :, 0]
    P[flip, :, 1] = W - P[flip, :, 1]
    B[flip, 0] = L - B[flip, 0]
    B[flip, 1] = W - B[flip, 1]
    return P, B


def _velocity(xy, smooth=5):
    """Central-difference velocity (m/s) over +/-smooth frames; NaN-tolerant."""
    v = np.full_like(xy, np.nan)
    v[smooth:-smooth] = (xy[2 * smooth:] - xy[:-2 * smooth]) * FPS / (2 * smooth)
    return v


def _since(event, cap=CAP):
    """Frames since the last True (inclusive); cap where none."""
    idx = np.where(event, np.arange(len(event)), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(event)) - last, cap).astype(np.float32)


def _until(event, cap=CAP):
    return _since(event[::-1], cap)[::-1]


def _roll(x, w, how="mean"):
    s = pd.Series(x)
    r = s.rolling(2 * w + 1, center=True, min_periods=1)
    return getattr(r, how)().values.astype(np.float32)


def frame_features(players, ball, team, half):
    n = len(ball)
    P, B = normalise(players, ball, team, half)
    home, away = team == 0, team == 1
    F = {}

    present = ~np.isnan(B[:, 0])
    F["ball_present"] = present.astype(np.float32)
    F["ball_since_seen"] = _since(present)
    F["ball_until_seen"] = _until(present)
    F["ball_x"], F["ball_y"] = B[:, 0], B[:, 1]
    out = np.maximum.reduce([-B[:, 0], B[:, 0] - L, -B[:, 1], B[:, 1] - W])
    F["ball_out_dist"] = out
    bv = _velocity(B, 3)
    F["ball_speed"] = np.hypot(bv[:, 0], bv[:, 1])
    F["ball_vx"] = bv[:, 0]

    # player kinematics
    pv = _velocity(P, 5)
    pspeed = np.hypot(pv[..., 0], pv[..., 1])
    tracked = ~np.isnan(P[..., 0])
    for name, m in (("h", home), ("a", away)):
        x, y = P[:, m, 0], P[:, m, 1]
        with np.errstate(all="ignore"), _quiet():
            F[name + "_n"] = tracked[:, m].sum(1).astype(np.float32)
            F[name + "_cx"] = np.nanmean(x, 1)
            F[name + "_cy"] = np.nanmean(y, 1)
            F[name + "_sx"] = np.nanstd(x, 1)
            F[name + "_sy"] = np.nanstd(y, 1)
            F[name + "_xmin"] = np.nanmin(x, 1)
            F[name + "_xmax"] = np.nanmax(x, 1)
            F[name + "_speed"] = np.nanmean(pspeed[:, m], 1)
            F[name + "_speed_max"] = np.nanmax(pspeed[:, m], 1)
            F[name + "_vx"] = np.nanmean(pv[:, m, 0], 1)
            F[name + "_in_opp_half"] = np.sum(x > L / 2, 1).astype(np.float32) if name == "h" \
                else np.sum(x < L / 2, 1).astype(np.float32)
        # distance from the ball
        dx = P[:, m, 0] - B[:, None, 0]
        dy = P[:, m, 1] - B[:, None, 1]
        d = np.hypot(dx, dy)
        dmin = np.where(np.isnan(d), np.inf, d).min(1)
        dmin[~present | np.isinf(dmin)] = np.nan
        F[name + "_dball"] = dmin
        with np.errstate(invalid="ignore"):
            F[name + "_n5"] = np.where(present, np.nansum(d < 5, 1), np.nan).astype(np.float32)
            for r in CONTACT_RADII:
                c = present & (dmin < r)
                F["{}_since_c{}".format(name, r)] = _since(c)
                F["{}_until_c{}".format(name, r)] = _until(c)
                F["{}_frac_c{}_1s".format(name, r)] = _roll(c.astype(np.float32), FPS)
                F["{}_frac_c{}_3s".format(name, r)] = _roll(c.astype(np.float32), 3 * FPS)

    # touches: the ball changes velocity sharply while a player is close. A kick or a reception is
    # what moves ownership, so these are the events the decoder most needs to see.
    acc = np.full(n, np.nan, np.float32)
    acc[3:-3] = np.hypot(bv[6:, 0] - bv[:-6, 0], bv[6:, 1] - bv[:-6, 1]) if n > 6 else acc[3:-3]
    F["ball_dv"] = acc
    peak = np.zeros(n, bool)
    with np.errstate(invalid="ignore"):
        a = np.nan_to_num(acc)
        peak[1:-1] = (a[1:-1] >= a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] > 3.0)
        for name in ("h", "a"):
            other = "a" if name == "h" else "h"
            near = (F[name + "_dball"] < 2.5) & ~(F[other + "_dball"] < F[name + "_dball"])
            t = peak & near
            F[name + "_touch"] = t.astype(np.float32)
            F[name + "_since_touch"] = _since(t)
            F[name + "_until_touch"] = _until(t)
            F[name + "_touch_3s"] = _roll(t.astype(np.float32), 3 * FPS, "sum")
            strong = t & (a > 8.0)
            F[name + "_since_kick"] = _since(strong)
            F[name + "_until_kick"] = _until(strong)
    F["last_touch_home"] = np.where(F["h_since_touch"] < F["a_since_touch"], 1.0,
                                    np.where(F["a_since_touch"] < F["h_since_touch"], -1.0, 0.0)).astype(np.float32)
    F["next_touch_home"] = np.where(F["h_until_touch"] < F["a_until_touch"], 1.0,
                                    np.where(F["a_until_touch"] < F["h_until_touch"], -1.0, 0.0)).astype(np.float32)

    with np.errstate(all="ignore"):
        F["dball_diff"] = F["h_dball"] - F["a_dball"]
        F["all_speed"] = np.nanmean(pspeed, 1)
        F["n_tracked"] = tracked.sum(1).astype(np.float32)
    # half clock
    t = np.zeros(n, np.float32)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        t[idx] = (idx - idx[0]) / FPS
    F["half_clock"] = t

    base = list(F)
    # temporal context
    for k in ["all_speed", "h_speed", "a_speed", "h_vx", "a_vx", "h_cx", "a_cx", "ball_present",
              "dball_diff", "h_sx", "a_sx", "h_xmax", "a_xmin", "ball_speed", "ball_x",
              "ball_out_dist", "h_n5", "a_n5"]:
        for w in (FPS, 3 * FPS, 10 * FPS):
            F["{}_m{}".format(k, w)] = _roll(F[k], w)
    for k in ["all_speed", "h_speed", "a_speed"]:
        F[k + "_min3s"] = _roll(F[k], 3 * FPS, "min")
        F[k + "_max3s"] = _roll(F[k], 3 * FPS, "max")
    # temporal derivatives of speeds (dead ball begins with a collective slow-down)
    for k in ["all_speed_m25", "h_cx_m25", "a_cx_m25"]:
        x = F[k]
        F[k + "_d2s"] = np.concatenate((np.full(50, np.nan), x[50:] - x[:-50])).astype(np.float32)
        F[k + "_f2s"] = np.concatenate((x[50:] - x[:-50], np.full(50, np.nan))).astype(np.float32)
    names = list(F)
    X = np.stack([np.asarray(F[k], np.float32) for k in names], 1)
    return X, names


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *a):
        self._w.__exit__(*a)
