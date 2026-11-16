"""Deterministic CPU rebuild of the v13 submission from the saved checkpoints (no GPU, fixed threads).

    MTAH_FEAT=2 MTAH_DROP=kickclock python predict_cpu.py submissions/v13_cpu.csv

Same recipe as v13: 5 set-encoder inits (aux head) + 1 set-encoder + LightGBM, team-swap TTA on the
set encoders, probabilities averaged, Viterbi with switch costs 3 (team) / 20 (dead).
"""
import hashlib
import os
import sys

import numpy as np
import torch

import seq_set
from mtah import dataset, decode, raw
from seq import Norm

torch.use_deterministic_algorithms(True)
torch.set_num_threads(8)
seq_set.DEV = "cpu"
SET_TAGS = ["_seta8nkc_i{}_d7".format(i) for i in range(5)] + ["_setnkc_d7"]
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
os.makedirs("cache", exist_ok=True)

X0, half, names = dataset.challenge()
X1 = dataset.challenge(swapped=True)[0]
r0, r1 = raw.challenge(), raw.challenge(swapped=True)
probs = []
for tag in SET_TAGS:
    ck = torch.load(os.path.join(WEIGHTS, "seq_final%s.pt" % tag), weights_only=False, map_location="cpu")
    norm = Norm(); norm.med, norm.scale = ck["med"], ck["scale"]
    net = seq_set.SetNet(X0.shape[1], **ck["arch"]); net.load_state_dict(ck["state"]); net.eval()
    p = seq_set.predict(net, norm, X0, r0, ck["T"])
    q = seq_set.predict(net, norm, X1, r1, ck["T"])
    p = (p + q[:, [1, 0] + list(range(2, q.shape[1]))]) / 2
    np.save(os.path.join("cache", "cpu_proba%s.npy" % tag), p)
    probs.append(p[:, :3])
    print("done", tag, flush=True)
import lightgbm as lgb  # noqa: E402
m = lgb.Booster(model_file=os.path.join(WEIGHTS, "lgbm_final_8nkc_d7.txt"))
assert m.feature_name() == list(names)
probs.append(m.predict(X0, num_threads=1)[:, :3])
labels = decode.decode(np.mean(probs, 0), half, 3.0, 20.0)
out = sys.argv[1]
with open(out, "w", newline="\n") as f:
    f.write("frame,label\n")
    f.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
print("wrote", out, "sha256", hashlib.sha256(open(out, "rb").read()).hexdigest())
