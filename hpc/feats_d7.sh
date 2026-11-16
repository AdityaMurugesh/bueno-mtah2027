#!/bin/bash
# d7 caches (d6 + recalibrated ball presence): hand features and per-player tokens, train seeds 0-7 + eval seed 99
export MTAH_DEG=7
python gen_feats.py --seeds 0 1 2 3 4 5 6 7 --workers 16
python gen_feats.py --seeds 0 1 2 3 4 5 6 7 --workers 16 --raw
