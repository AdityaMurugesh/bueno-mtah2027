"""Parse a Metrica FIFA/EPTS ATD pair into dense numpy arrays, cached as .npz.

Arrays are indexed by video frame (ATD frame_id - 1), in metres, origin at the pitch corner:
    players  (n_frames, n_slots, 3)  x, y, speed  -- NaN where the slot is not tracked
    ball     (n_frames, 3)           x, y, speed  -- NaN where the ball is not tracked
    team     (n_slots,)              0 = home, 1 = away
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], dtype=np.int8)


def parse_raw(raw_path, n_frames, n_slots, pitch):
    length, width = pitch
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            groups = parts[1].split(";")
            players[frame, :len(groups)] = [[float(v) for v in g.split(",")] for g in groups]
            ball[frame] = [float(v) for v in parts[2].split(",")[:3]]
    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return players, ball


def load(source="tactical", game="BAR-ATH", refresh=False):
    """Dict with players, ball, team. Parsed once, then read from cache."""
    import sys
    repo = os.path.join(os.path.dirname(CACHE_DIR), "..", "repo")
    sys.path.insert(0, os.path.abspath(repo))
    from games import GAMES, asset_paths, source_frames

    cache = os.path.join(CACHE_DIR, "{}_{}_atd.npz".format(game, source))
    if os.path.exists(cache) and not refresh:
        data = np.load(cache)
        return {k: data[k] for k in data.files}

    paths = asset_paths(game=game, source=source)
    team = slot_teams(paths["atd_meta"])
    players, ball = parse_raw(paths["atd_raw"], source_frames(source, game), len(team),
                              GAMES[game]["pitch"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, players=players, ball=ball, team=team)
    return {"players": players, "ball": ball, "team": team}


if __name__ == "__main__":
    d = load(refresh=True)
    p, b = d["players"], d["ball"]
    print("players", p.shape, "ball", b.shape, "slots home/away", np.bincount(d["team"]))
    print("ball present %.1f%%" % (100 * np.mean(~np.isnan(b[:, 0]))))
    print("mean tracked players %.1f" % np.mean(np.sum(~np.isnan(p[..., 0]), axis=1)))
