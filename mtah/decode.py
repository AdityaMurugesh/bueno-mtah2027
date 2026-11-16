"""Turn per-frame class probabilities into possession segments.

Viterbi over three states (H, A, D) with a fixed cost for every change of state: a possession is
an interval, so the decoder pays to open a new one. Two costs, because a hand-over between teams and
a stoppage are different events with different evidence.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "repo", "evaluation"))
import possession_metrics as pm  # noqa: E402

STATES = np.array(["H", "A", "D"])


CMIN = 0.5  # floor on a switch cost when the boundary head lowers it


def viterbi(logp, cost_team, cost_dead, bonus=None):
    """logp (n, 3) log-scores for H, A, D. Returns the best state path (n,) as ints.

    bonus (n,) optional: subtracted from every switch cost into frame t (floored at CMIN), so a
    boundary detector can make changes cheap where it fires and dear where it is sure there is none."""
    n = len(logp)
    trans = np.array([[0, cost_team, cost_dead],
                      [cost_team, 0, cost_dead],
                      [cost_dead, cost_dead, 0]], dtype=np.float64)
    back = np.zeros((n, 3), np.int8)
    score = logp[0].astype(np.float64).copy()
    lp = logp.astype(np.float64)
    offd = trans > 0
    for t in range(1, n):
        tr = trans if bonus is None else np.where(offd, np.maximum(trans - bonus[t], CMIN), 0.0)
        cand = score[:, None] - tr             # from i to j
        best = cand.argmax(0)
        back[t] = best
        score = cand[best, np.arange(3)] + lp[t]
    path = np.empty(n, np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(proba, half, cost_team=40.0, cost_dead=40.0, prior=None, prior_weight=0.0, beta=0.0):
    """Labels (n,) over the whole array; 'X' where half == 0. Each half decoded separately.

    proba (n, 3) or (n, 4); a 4th column is the boundary probability, used when beta > 0."""
    eps = 1e-6
    bonus = None
    if proba.shape[1] > 3 and beta:
        b = np.clip(proba[:, 3], 1e-4, 1 - 1e-4)
        bonus = beta * np.log(b / (1 - b))
    proba = proba[:, :3]
    logp = np.log(np.clip(proba, eps, 1))
    if prior is not None and prior_weight:
        logp = logp - prior_weight * np.log(prior)[None]
    labels = np.full(len(proba), "X", "<U1")
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        labels[idx] = STATES[viterbi(logp[idx], cost_team, cost_dead, None if bonus is None else bonus[idx])]
    return labels


def score(gt_labels, pred_labels):
    r = pm.evaluate(pm.label_codes(gt_labels), pm.label_codes(pred_labels))
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "n_pred": r["n_pred_spells"],
            "n_gt": r["n_gt_spells"], "alive_dead_f1": r["alive_dead_f1"],
            "team_acc": r["team_acc_both_alive"]}
