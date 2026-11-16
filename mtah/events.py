"""DFL open-data events -> restart frames on the merged idsse timeline, and simulated Smart Tagging set pieces.

Restarts (throw-in, corner, goal kick, free kick, kick-off, penalty) are what Metrica's Smart Tagging marks as
SET PIECES on the challenge match. The DFL event files give the same restarts for the 7 bench matches, so we can
build tag-like inputs there and measure what they are worth before trusting the real tags.
"""

import os
import re
from datetime import datetime

import numpy as np

from . import idsse

FPS = 25
TYPES = ["ThrowIn", "CornerKick", "GoalKick", "FreeKick", "KickOff", "Penalty"]
# Smart Tagging "Type" values for the same restarts
TAG_TYPE = {"Throw-in": "ThrowIn", "Corner Kick": "CornerKick", "Goal Kick": "GoalKick", "Free Kick": "FreeKick",
            "Kick Off (After Goal)": "KickOff", "Kick Off (Start)": "KickOff", "Penalty": "Penalty"}
CACHE = os.path.join(idsse.CACHE, "..", "events")

EVENT = re.compile(r'<Event [^>]*EventTime="([^"]+)"')
KIND = re.compile(r"<(%s)[ >/]" % "|".join(TYPES))
BALLSET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="BALL"')
FRAME_T = re.compile(r'<Frame N="(\d+)" T="([^"]+)"')


def _t(s):
    return datetime.fromisoformat(s).timestamp()


def _half_clock(match):
    """[(first N, time of first N, n frames)] for the ball track of each half, as idsse.parse lays them out."""
    out, cur, first, last = {}, None, None, None
    with open(idsse._file("04_03_positions_raw_observed", match), encoding="utf-8") as handle:
        for line in handle:
            if "<FrameSet" in line:
                if cur:
                    out[cur] = (int(first[0]), _t(first[1]), int(last[0]) - int(first[0]) + 1)
                m = BALLSET.search(line)
                cur, first, last = (m.group(1) if m else None), None, None
            elif cur and "<Frame " in line:
                m = FRAME_T.search(line)
                if first is None:
                    first = m.groups()
                last = m.groups()
    if cur:
        out[cur] = (int(first[0]), _t(first[1]), int(last[0]) - int(first[0]) + 1)
    return [out["firstHalf"], out["secondHalf"]]


def restarts(match, refresh=False):
    """(frame, type index) of every restart event, frame on the merged idsse timeline."""
    path = os.path.join(CACHE, match + ".npz")
    if os.path.exists(path) and not refresh:
        z = np.load(path)
        return z["frame"], z["kind"]
    clock = _half_clock(match)
    offs = [0, clock[0][2] + idsse.GAP]
    frames, kinds = [], []
    text = open(idsse._file("03_02_events_raw", match), encoding="utf-8").read()
    for block in text.split("<Event ")[1:]:
        k = KIND.search(block)
        if not k:
            continue
        t = _t(EVENT.search("<Event " + block).group(1))
        for h, (n0, t0, n) in enumerate(clock):
            f = int(round((t - t0) * FPS))
            if -FPS * 60 <= f < n + FPS * 60:  # events can sit just outside the tracked span
                frames.append(offs[h] + min(max(f, 0), n - 1)); kinds.append(TYPES.index(k.group(1)))
                break
    order = np.argsort(frames)
    frame, kind = np.array(frames, np.int64)[order], np.array(kinds, np.int8)[order]
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, frame=frame, kind=kind)
    return frame, kind
