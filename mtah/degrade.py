"""Make clean optical tracking look like Metrica's automatic, uncorrected ATD.

Statistics matched (measured on the challenge ATD, graded frames, aggregates only):
    ball missing ~41% of frames, in long blackouts: median 10.7 s, p90 34 s
    ball present runs: median 13.8 s, p90 62 s
    players tracked per frame: mean 20.5, p5 14; slot tracklets median ~35 s
Every parameter is drawn per copy from a range around those values, so the model sees a family of
noise levels rather than one guess.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

FPS = 25


def _lognormal_runs(rng, n, median_s, p90_s):
    sigma = np.log(p90_s / median_s) / 1.2816
    return lambda: max(1, int(rng.lognormal(np.log(median_s * FPS), sigma)))


def ball_mask(rng, n, miss_med, miss_p90, pres_med, pres_p90):
    """Alternating renewal process: True where the ball is kept."""
    keep = np.ones(n, bool)
    draw_miss = _lognormal_runs(rng, n, miss_med, miss_p90)
    draw_pres = _lognormal_runs(rng, n, pres_med, pres_p90)
    t = int(rng.integers(0, 10 * FPS))
    present = rng.random() < 0.6
    while t < n:
        length = draw_pres() if present else draw_miss()
        if not present:
            keep[t:t + length] = False
        t += length
        present = not present
    return keep


def degrade(d, seed, strength=1.0):
    """Return a degraded copy of an mtah.idsse match dict (labels untouched)."""
    rng = np.random.default_rng(seed)
    P = d["players"].copy()
    B = d["ball"].copy()
    team = d["team"].copy()
    n, k = P.shape[:2]

    # ball: long blackouts, jitter, rare outliers
    miss_med = rng.uniform(7, 15) * strength
    pres_med = rng.uniform(10, 18) / strength
    keep = ball_mask(rng, n, miss_med, miss_med * rng.uniform(2.5, 3.8),
                     pres_med, pres_med * rng.uniform(3.5, 5.0))
    B[~keep] = np.nan
    B[:, :2] += rng.normal(0, rng.uniform(0.15, 0.5), (n, 2)).astype(np.float32)
    out = rng.random(n) < 0.005
    B[out, :2] += rng.normal(0, 8, (out.sum(), 2)).astype(np.float32)

    # players: tracklet gaps per player
    for j in range(k):
        alive = ~np.isnan(P[:, j, 0])
        t = 0
        while t < n:
            t += max(FPS, int(rng.lognormal(np.log(35 * FPS), 1.0)))
            gap = max(1, int(rng.lognormal(np.log(1.5 * FPS), 1.0)))
            P[t:t + gap, j] = np.nan
            t += gap
        del alive
    # crowd dropouts: several players vanish together for a few seconds
    t = 0
    while t < n:
        t += int(rng.exponential(40 * FPS))
        length = int(rng.uniform(2, 10) * FPS)
        lost = rng.choice(k, size=min(k, int(rng.integers(3, 9))), replace=False)
        P[t:t + length, lost] = np.nan
        t += length
    P[..., :2] += rng.normal(0, rng.uniform(0.1, 0.35), P[..., :2].shape).astype(np.float32)
    # occasional wrong team on a slot
    flip = rng.random(k) < 0.03
    team[flip] = 1 - team[flip]
    return {"players": P, "ball": B, "team": team, "labels": d["labels"], "half": d["half"]}


def _nan_smooth(a, sigma):
    """Gaussian smoothing along time that leaves NaN gaps as gaps (normalised convolution)."""
    m = ~np.isnan(a)
    num = gaussian_filter1d(np.where(m, a, 0.0), sigma, axis=0, mode="nearest")
    den = gaussian_filter1d(m.astype(np.float64), sigma, axis=0, mode="nearest")
    out = (num / np.maximum(den, 1e-6)).astype(np.float32)
    out[~m] = np.nan
    return out


def _drift(rng, shape, amp, tau_s):
    """Slow, smooth position error: white noise low-passed over ~tau_s, rescaled to std amp."""
    z = gaussian_filter1d(rng.normal(0, 1, shape), tau_s * FPS, axis=0, mode="wrap")
    return (z / max(z.std(), 1e-9) * amp).astype(np.float32)


def degrade2(d, seed, strength=1.0):
    """v2, matched to more ATD statistics (aggregates over graded frames, no labels):
    - the ATD is smooth: median |second difference| 0.0014 m (players), ball p99 0.06 m. White jitter
      is wrong, so tracks are Gaussian-smoothed (players sigma ~2 frames, ball ~6) plus a slow drift.
    - ATD ball presence depends on activity: 6% when players are slow, ~90% at high tempo; overall 59%.
      So the ball is dropped mostly in dead-ball time: two renewal processes, one per state.
    """
    rng = np.random.default_rng(seed)
    P = d["players"].copy()
    B = d["ball"].copy()
    team = d["team"].copy()
    n, k = P.shape[:2]
    dead = d["labels"] == "D"

    B[:, :2] = _nan_smooth(B[:, :2].astype(np.float64), rng.uniform(4, 8))
    B[:, :2] += _drift(rng, (n, 2), rng.uniform(0.1, 0.4), rng.uniform(0.5, 2.0))
    keep_alive = ball_mask(rng, n, rng.uniform(3, 8) * strength, rng.uniform(15, 30) * strength,
                           rng.uniform(30, 60) / strength, rng.uniform(100, 200) / strength)
    keep_dead = ball_mask(rng, n, rng.uniform(20, 40), rng.uniform(60, 120), rng.uniform(1, 3), rng.uniform(4, 8))
    keep = np.where(dead, keep_dead, keep_alive)
    # the ball usually stays visible for a moment after play stops and while it is set for the restart
    edges = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        head, tail = int(rng.uniform(1, 6) * FPS), int(rng.uniform(0, 4) * FPS)
        keep[a:min(b, a + head)] = keep_alive[a:min(b, a + head)]
        keep[max(a, b - tail):b] = True
    B[~keep] = np.nan
    _players2(rng, P, team)
    return {"players": P, "ball": B, "team": team, "labels": d["labels"], "half": d["half"]}


def _players2(rng, P, team):
    """degrade2's player noise, in place: smoothing, per-slot gaps, group losses, drift, team flips."""
    n, k = P.shape[:2]
    P[..., :2] = _nan_smooth(P[..., :2].astype(np.float64), rng.uniform(1.5, 3))
    for j in range(k):
        t = 0
        while t < n:
            t += max(FPS, int(rng.lognormal(np.log(35 * FPS), 1.0)))
            gap = max(1, int(rng.lognormal(np.log(1.5 * FPS), 1.0)))
            P[t:t + gap, j] = np.nan
            t += gap
    t = 0
    while t < n:
        t += int(rng.exponential(40 * FPS))
        length = int(rng.uniform(2, 10) * FPS)
        lost = rng.choice(k, size=min(k, int(rng.integers(3, 9))), replace=False)
        P[t:t + length, lost] = np.nan
        t += length
    P[..., :2] += _drift(rng, (n, k, 2), rng.uniform(0.1, 0.35), rng.uniform(0.5, 2.0))
    flip = rng.random(k) < 0.03
    team[flip] = 1 - team[flip]


# ATD ball presence by collective player speed (3 s mean, m/s), measured label-free by calib.py
ATD_SPEED = np.array([0.3, 0.75, 1.05, 1.35, 1.75, 2.25, 3.0, 4.0])
ATD_PRESENT = np.array([0.06, 0.17, 0.34, 0.50, 0.75, 0.90, 0.89, 0.93])


def tempo_mask(rng, speed, tau_s=4.0, min_run_s=1.0, shift=0.0, present=None):
    """Ball kept where a persistent latent N(0,1) process falls below the presence quantile for the current tempo.

    Uses no labels: visibility follows what the camera sees (play speed), as in the ATD, rather than the
    ground-truth dead/alive state, which degrade2 used and the model could read straight off the mask."""
    from scipy.stats import norm
    n = len(speed)
    p = np.interp(np.nan_to_num(speed, nan=0.0), ATD_SPEED, ATD_PRESENT if present is None else present)
    thr = norm.ppf(np.clip(p, 1e-3, 1 - 1e-3)) + shift
    a = np.exp(-1.0 / (tau_s * FPS))
    e = rng.normal(size=n) * np.sqrt(1 - a * a)
    z = np.empty(n)
    z[0] = rng.normal()
    for t in range(1, n):
        z[t] = a * z[t - 1] + e[t]
    keep = z < thr
    # merge flickers: runs shorter than min_run_s take the state of the run before
    m = int(min_run_s * FPS)
    edges = np.flatnonzero(np.diff(keep.astype(np.int8))) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, n]
    for a0, b0 in zip(starts[1:], ends[1:]):
        if b0 - a0 < m:
            keep[a0:b0] = keep[a0 - 1]
    return keep


def degrade3(d, seed, strength=1.0, tau_s=None, players=None, present=None):
    """v3 = degrade2 with a label-free ball dropout (tempo_mask) instead of one keyed to the dead/alive labels.

    players: optional replacement for _players2, called as players(rng, P, team)."""
    rng = np.random.default_rng(seed)
    P = d["players"].copy()
    B = d["ball"].copy()
    team = d["team"].copy()
    xy = d["players"][..., :2].astype(np.float64)
    v = np.full(xy.shape, np.nan)
    v[5:] = (xy[5:] - xy[:-5]) * (FPS / 5.0)                      # m/s over 0.2 s, like features._velocity
    with np.errstate(all="ignore"):
        sp = np.nanmean(np.hypot(v[..., 0], v[..., 1]), 1)
    sp = np.convolve(np.nan_to_num(sp, nan=0.0), np.ones(3 * FPS) / (3 * FPS), mode="same")
    B[:, :2] = _nan_smooth(B[:, :2].astype(np.float64), rng.uniform(4, 8))
    B[:, :2] += _drift(rng, (len(B), 2), rng.uniform(0.1, 0.4), rng.uniform(0.5, 2.0))
    # tau 7-13 s, runs >= 2 s, shift -0.1: matches ATD presence 0.59, missing-run median/p90 10.7/33.8 s (calib.py --deg 3)
    keep = tempo_mask(rng, sp, tau_s if tau_s is not None else rng.uniform(7, 13), 2.0, -0.1, present)
    B[~keep] = np.nan
    out = (players or _players2)(rng, P, team)
    if out is not None:                                           # players may add slots
        P, team = out
    return {"players": P, "ball": B, "team": team, "labels": d["labels"], "half": d["half"]}


def swap(d):
    """Home <-> away: same match, roles exchanged. Labels follow."""
    lab = d["labels"].copy()
    lab[d["labels"] == "H"] = "A"
    lab[d["labels"] == "A"] = "H"
    return {**d, "team": 1 - d["team"], "labels": lab}


def handle_dead(rng, P, B, labels):
    """Replace DFL's frozen dead-ball position with a ball that is picked up, carried and placed (in place, on B).

    In the DFL data the ball stays frozen where it went out for the whole stoppage (speed 0, a goal kick's ball
    ~11 m from its restart spot, a kick-off's ~53 m) and jumps to the restart spot at the restart. Metrica's
    camera tracking follows the real ball instead: in the first seconds of BAR-ATH goal-kick tags the ball is
    seen 80% of the time, 4 m inside the pitch, moving ~1 m/s with a player ~3 m away. A frozen ball is a
    giveaway the model learns on the bench and cannot use on BAR-ATH.

    Per dead run ending in a restart at frame b (restart spot r = B[b]): the ball stays at the out spot (or is
    lost, e.g. a fresh ball from a ball-kid) until a pick-up time, is then carried by the player who takes the
    restart (nearest to r at b), and sits at r for the last 0.5-3 s. A few runs are left frozen."""
    n = len(labels)
    alive = np.isin(labels, ["H", "A"])
    dead = labels == "D"
    ups = np.flatnonzero(alive[1:] & dead[:-1]) + 1
    for b in ups:
        a = b - 1
        while a > 0 and dead[a - 1]:
            a -= 1
        L = b - a
        if L < FPS or rng.random() < 0.15 or np.isnan(B[b, 0]):
            continue
        r = B[b, :2].copy()
        d2 = np.hypot(P[b, :, 0] - r[0], P[b, :, 1] - r[1])
        if np.all(np.isnan(d2)):
            continue
        k = int(np.nanargmin(d2))
        place = min(int(rng.uniform(0.5, 3.0) * FPS), int(0.4 * L))
        tp = a + int(rng.uniform(0.1, 0.7) * (L - place))
        if rng.random() < 0.5:
            B[a:tp] = np.nan  # the ball that went out is not followed
        ang = rng.uniform(0, 2 * np.pi)
        off = rng.uniform(0.3, 0.8) * np.array([np.cos(ang), np.sin(ang)])
        seg = slice(tp, b - place)
        carried = P[seg, k, :2] + off
        ok = ~np.isnan(carried[:, 0])
        B[seg, :2] = np.where(ok[:, None], carried, np.nan)
        if B.shape[1] > 2:
            B[seg, 2] = np.where(ok, B[seg, 2], np.nan)
        B[b - place:b, :2] = r
    return B


def degrade4(d, seed, strength=1.0, tau_s=None):
    """v4 = degrade3 on a ball that is handled during stoppages (handle_dead) instead of frozen."""
    B = d["ball"].astype(np.float64).copy()
    handle_dead(np.random.default_rng(seed + 7_777_777), d["players"], B, d["labels"])
    return degrade3({**d, "ball": B.astype(d["ball"].dtype)}, seed, strength, tau_s)


# ATD players missing (22 - tracked) by collective speed (6 s mean, m/s): quantiles at ATD_MISS_Q per speed bin,
# measured label-free by calib_players.py. Losses are rare at match tempo and heavy (often most of the pitch,
# for up to ~20 s) when play is slow, i.e. mostly in stoppages; DFL tracking loses nobody, degrade2-4 lost
# ~1.4 players uniformly whatever the tempo.
ATD_MISS_EDGES = np.array([0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4])
ATD_MISS_Q = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98])
ATD_MISS = np.array([
    [0, 0, 1, 1, 2, 2, 2, 3, 6, 8, 14, 15, 17, 21],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 5, 11, 17],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 4, 7, 15],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 5, 18],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 5, 11],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 7]], np.float64)


def _ar(rng, shape, tau_s):
    """Stationary N(0,1) AR(1) process along axis 0 with time constant tau_s."""
    a = np.exp(-1.0 / (tau_s * FPS))
    e = rng.normal(size=shape) * np.sqrt(1 - a * a)
    z = np.empty(shape)
    z[0] = rng.normal(size=shape[1:])
    for t in range(1, shape[0]):
        z[t] = a * z[t - 1] + e[t]
    return z


def tempo_players(rng, P, scale=1.0):
    """Drop players like the ATD, in place: how many from the tempo (copula on a persistent latent, so heavy
    losses last), which ones at random but persistent. Not the farthest from the play: that removed keepers and
    the deepest/highest outfielders, and the ATD's team extents (h_xmin, a_xmax, ...) show those stay tracked."""
    from scipy.stats import norm
    n, k = P.shape[:2]
    xy = P[..., :2].astype(np.float64)
    v = np.full(xy.shape, np.nan)
    v[5:] = (xy[5:] - xy[:-5]) * (FPS / 5.0)
    with np.errstate(all="ignore"):
        sp = np.nanmean(np.hypot(v[..., 0], v[..., 1]), 1)
    w = 6 * FPS + 1
    sp = np.convolve(np.nan_to_num(sp, nan=0.0), np.ones(w) / w, mode="same")
    u = norm.cdf(_ar(rng, (n,), rng.uniform(10, 25)))      # long: ATD heavy losses last up to ~20 s
    q = np.r_[0.0, ATD_MISS_Q, 1.0]
    b = np.digitize(sp, ATD_MISS_EDGES)
    miss = np.zeros(n)
    for j in range(len(ATD_MISS)):
        m = b == j
        row = np.r_[ATD_MISS[j, 0], ATD_MISS[j], ATD_MISS[j, -1]]
        miss[m] = np.interp(u[m], q, row)
    dither = norm.cdf(_ar(rng, (n,), 2.0))                     # slow dither: fractional quantiles average out
    miss = median_filter(np.floor(miss * scale + dither), size=FPS, mode="nearest").astype(int)  # no flicker
    tracked = ~np.isnan(xy[..., 0])
    miss = np.clip(miss - (22 - tracked.sum(1)), 0, None)       # DFL already lacks a sent-off player etc.
    score = _ar(rng, (n, k), rng.uniform(3, 10))
    score[~tracked] = -np.inf
    rank = np.argsort(np.argsort(-score, 1), 1)                 # 0 = farthest from the play
    P[rank < miss[:, None]] = np.nan


def _players5(rng, P, team):
    """degrade2's smoothing, drift and team flips, with tempo_players instead of uniform gaps and group losses."""
    n, k = P.shape[:2]
    tempo_players(rng, P, rng.uniform(0.8, 1.2))
    P[..., :2] = _nan_smooth(P[..., :2].astype(np.float64), rng.uniform(1.5, 3))
    P[..., :2] += _drift(rng, (n, k, 2), rng.uniform(0.1, 0.35), rng.uniform(0.5, 2.0))
    flip = rng.random(k) < 0.03
    team[flip] = 1 - team[flip]


def degrade5(d, seed, strength=1.0, tau_s=None, present=None):
    """v5 = degrade4 with tempo-dependent, mostly off-camera player losses (tempo_players)."""
    B = d["ball"].astype(np.float64).copy()
    handle_dead(np.random.default_rng(seed + 7_777_777), d["players"], B, d["labels"])
    return degrade3({**d, "ball": B.astype(d["ball"].dtype)}, seed, strength, tau_s, players=_players5, present=present)


def keeper_depth(rng, d, pivot=6.0, lo=0.3, hi=1.0):
    """Pull each keeper toward his own goal line, in place on a copy: depth' = pivot + (depth - pivot) * f beyond
    pivot metres, f ~ U(lo, hi) per team and half. ATD keepers stand ~9 m off their line in open play, DFL's
    ~15 m (adv_val.py), and a 6 m shift cost the d4 LightGBM 0.034 spellF1 on the bench (sens_keeper.py), so
    the model must not read keeper depth in absolute metres. Near the line (goal kicks, saves) is unchanged."""
    from mtah import features
    P = d["players"].copy()
    signs = features.attack_sign(P, d["team"], d["half"])
    for h, s in signs.items():
        idx = np.flatnonzero(d["half"] == h)
        for t in (0, 1):
            own_left = (s > 0) == (t == 0)
            slots = np.flatnonzero(d["team"] == t)
            x = P[idx][:, slots, 0].astype(np.float64)
            depth = x if own_left else features.L - x
            with np.errstate(all="ignore"):
                med = np.nanmedian(depth, 0)
            if np.all(np.isnan(med)):
                continue
            j = int(np.nanargmin(med))
            dep = depth[:, j]
            f = rng.uniform(lo, hi)
            dep = np.where(dep > pivot, pivot + (dep - pivot) * f, dep)
            P[idx, slots[j], 0] = dep if own_left else features.L - dep
    return {**d, "players": P}


def degrade6(d, seed, strength=1.0, tau_s=None, present=None):
    """v6 = degrade5 on players whose keepers stand at a random depth (keeper_depth)."""
    d = keeper_depth(np.random.default_rng(seed + 5_555_555), d)
    return degrade5(d, seed, strength, tau_s, present)


# Ball presence table for degrade7, on the ATD_SPEED grid. ATD_PRESENT was measured with a different tempo window,
# and through the degrade pipeline (3 s clean-player tempo, run merging, handled dead ball) the bench came out ~0.1
# below the ATD at 0.75-2 m/s on the feature-level curve (all_speed_m75 vs ball_present, all frames, label-free;
# calib_ball2.py). Iterated with calib_ball2.py --fit until the bench curve matches BAR-ATH's.
ATD_PRESENT7 = np.array([0.04, 0.205, 0.434, 0.669, 0.862, 0.98, 0.894, 0.913])


def degrade7(d, seed, strength=1.0, tau_s=None):
    """v7 = degrade6 with the ball presence recalibrated (ATD_PRESENT7)."""
    return degrade6(d, seed, strength, tau_s, ATD_PRESENT7)


def transient_flips(rng, P, team, spare=3):
    """ATD team mix-ups are brief: a team shows >= 12 players in ~6% of frames, in runs of median ~3 s (p90 ~15 s),
    ~1.25 per minute (calib on BAR-ATH h_n/a_n, label-free; with tempo losses taking a share of the moved
    frames, rate 1.2-2.4/min and median 3-5 s reproduce the ATD's h_n/a_n distribution). d2-d7 flipped ~3% of slots for the whole match, which
    left 6 of 7 eval copies 12 v 10 most of the time (ATD: 11 v 11 in 42% of frames). Each event moves one tracked
    player's frames into a spare slot of the other team. Returns the widened (P, team)."""
    n, k = P.shape[:2]
    P2 = np.concatenate([P, np.full((n, 2 * spare) + P.shape[2:], np.nan, P.dtype)], 1)
    team2 = np.r_[team, np.repeat([0, 1], spare)].astype(team.dtype)
    free = {0: [k + i for i in range(spare)], 1: [k + spare + i for i in range(spare)]}
    busy = {}                                                      # spare slot -> frame it frees up
    rate = rng.uniform(1.2, 2.4) / (60 * FPS)
    med = rng.uniform(3.0, 5.0) * FPS
    t = int(rng.exponential(1 / rate))
    while t < n:
        tracked = np.flatnonzero(~np.isnan(P2[t, :k, 0]))
        if len(tracked):
            j = int(rng.choice(tracked))
            other = 1 - int(team2[j])
            for sl, until in list(busy.items()):
                if until <= t:
                    free[int(team2[sl])].append(sl); del busy[sl]
            if free[other]:
                sl = free[other].pop()
                L = max(FPS // 2, int(rng.lognormal(np.log(med), 1.0)))
                P2[t:t + L, sl] = P2[t:t + L, j]
                P2[t:t + L, j] = np.nan
                busy[sl] = t + L
        t += int(rng.exponential(1 / rate)) + 1
    return P2, team2


def _players8(rng, P, team):
    """_players5 with transient team mix-ups (transient_flips) instead of whole-match ones."""
    n, k = P.shape[:2]
    tempo_players(rng, P, rng.uniform(0.8, 1.2))
    P[..., :2] = _nan_smooth(P[..., :2].astype(np.float64), rng.uniform(1.5, 3))
    P[..., :2] += _drift(rng, (n, k, 2), rng.uniform(0.1, 0.35), rng.uniform(0.5, 2.0))
    return transient_flips(rng, P, team.copy())


def degrade8(d, seed, strength=1.0, tau_s=None):
    """v8 = degrade7 with transient team mix-ups (_players8)."""
    d = keeper_depth(np.random.default_rng(seed + 5_555_555), d)
    B = d["ball"].astype(np.float64).copy()
    handle_dead(np.random.default_rng(seed + 7_777_777), d["players"], B, d["labels"])
    return degrade3({**d, "ball": B.astype(d["ball"].dtype)}, seed, strength, tau_s, players=_players8,
                    present=ATD_PRESENT7)
