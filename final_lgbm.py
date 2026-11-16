"""LightGBM on all seven external matches. Training needs no BAR-ATH data, so it can run on the HPC.

    python final_lgbm.py --train --tag _v2      # -> cache/lgbm_final_v2.txt
    python final_lgbm.py --predict --tag _v2    # laptop: -> cache/bar_ath_proba_lgbm_v2.npy
"""

import argparse
import os

import lightgbm as lgb
import numpy as np

from cv import PARAMS, ROUNDS, training_set
from mtah import dataset, idsse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--tag", default="_v2")
    args = ap.parse_args()
    path = os.path.join("cache", "lgbm_final%s.txt" % args.tag)
    if args.train:
        Xtr, ytr, names = training_set(idsse.MATCHES)
        model = lgb.train({**PARAMS, "seed": 0, "deterministic": True, "force_col_wise": True},
                          lgb.Dataset(Xtr, ytr, feature_name=list(names)), ROUNDS)
        model.save_model(path)
        print("saved", path)
    if args.predict:
        model = lgb.Booster(model_file=path)
        X, half, names = dataset.challenge()
        assert model.feature_name() == list(names), "feature mismatch: check MTAH_FEAT"
        np.save(os.path.join("cache", "bar_ath_proba_lgbm%s.npy" % args.tag), model.predict(X))
        print("predicted")
