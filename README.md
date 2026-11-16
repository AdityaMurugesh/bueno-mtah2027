# Ball possession from automatic tracking (More than a Hack 2027, Phase 1)

Per-frame `H` / `A` / `D` / `X` labels for FC Barcelona vs Athletic Club from the tactical automatic tracking.
Method, data and limitations: **[TECHNICAL.md](TECHNICAL.md)**.

## Reproduce the submission

```
git clone https://github.com/metrica-sports/more-than-a-hack-2027 repo      # unpack the data bundle into repo/data (its DATA.md)
git clone https://github.com/AdityaMurugesh/bueno-mtah2027 possession                                                # must sit next to repo/
cd possession
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt                                # .venv/bin/pip on Linux / macOS
.venv/Scripts/python make_submission.py --team bueno                     # -> out/bueno.csv (+ official validator)
```

CPU only, deterministic, about 15-20 minutes. Expected `sha256(out/bueno.csv)` = `b86c6d8d07c7a310ff4337e7dfe84de1d90fefac001ef4b3f7c2b85e2bb3effc`.

| step | file |
|---|---|
| parse the tracking, 131 features per frame, per-player tokens | `mtah/atd.py`, `mtah/features.py`, `mtah/raw.py` |
| 6 set-encoder BiGRUs + LightGBM, home/away-swap TTA, Viterbi (3 / 20) | `predict_cpu.py`, `seq_set.py`, `mtah/decode.py` |
| kickoff rule (Smart Tagging "Kick Off (Start)") | `kickoff_rule.py` |
| trained weights (open IDSSE data only) | `weights/` |

## Retraining (optional, GPU)

The weights were trained on the 7 open IDSSE Bundesliga matches, degraded to look like the challenge tracking.
Place the IDSSE XML files in `ext/idsse/`, then run the commands in `hpc/d7_finals.sh` and `hpc/lgbm8_fin_d7c.sh`
(`seq_set.py --final ...`, `final_lgbm.py --train ...`, with `MTAH_DEG=7 MTAH_DROP=kickclock MTAH_FEAT=2`).
Leave-one-match-out validation: `seq_set.py --cv`, `cv.py`. Noise-model calibration and checks: `tools/`.

## Data and licences

- No match data is in this repository. The BAR-ATH assets are licensed to registered teams only.
- Training data: IDSSE, Bassek, M., Rein, R., Weber, H., Memmert, D. (2025), *An integrated dataset of
  spatiotemporal and event data in elite soccer*, Scientific Data. CC-BY 4.0. The weights in `weights/` are
  derived from it.
- The noise model uses only aggregate, label-free statistics of the challenge tracking (presence-vs-tempo curves,
  quantile tables) written as constants in `mtah/degrade.py`. No frames.
