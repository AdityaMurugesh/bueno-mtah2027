"""One command from the data bundle to the submission file (CPU, deterministic).

    python make_submission.py --team <team_name>        # -> out/<team_name>.csv, then the official validator

Steps: tracking features + per-player tokens (cached in cache/), the ensemble on CPU (predict_cpu.py,
weights/ trained on open IDSSE data only), then the kickoff rule (kickoff_rule.py).
Expects the official challenge repo, with the data bundle unpacked in it, at ../repo.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--team", required=True)
a = ap.parse_args()
env = dict(os.environ, MTAH_FEAT="2", MTAH_DROP="kickclock", CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore")
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
raw_csv = os.path.join(HERE, "out", "ensemble_raw.csv")
out = os.path.join(HERE, "out", a.team + ".csv")


def run(*cmd):
    subprocess.run([sys.executable, *cmd], cwd=HERE, env=env, check=True)


run("predict_cpu.py", raw_csv)
run("kickoff_rule.py", raw_csv, out)
run(os.path.join(HERE, "..", "repo", "evaluation", "validate_submission.py"), out)
