"""Build the degraded-copy feature caches for the external matches in parallel (HPC helper).

    MTAH_FEAT=2 python gen_feats.py --seeds 0 1 2 3 --workers 24
"""

import argparse
import itertools
import time
from multiprocessing import Pool

from mtah import dataset, idsse, raw

EVAL_SEED = 99


def one(job):
    m, s, sw, want_raw = job
    t0 = time.time()
    (raw.external if want_raw else dataset.external)(m, s, sw)
    return "%s s%d %s %.0fs" % (m, s, "sw" if sw else "or", time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--raw", action="store_true", help="per-player token caches instead of hand features")
    args = ap.parse_args()
    jobs = [(m, s, sw, args.raw) for m, s, sw in itertools.product(idsse.MATCHES, args.seeds, (False, True))]
    jobs += [(m, EVAL_SEED, sw, args.raw) for m in idsse.MATCHES for sw in (False, True)]  # swapped: for TTA
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(one, jobs):
            print(r, flush=True)
