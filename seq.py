"""Bidirectional GRU over the per-frame features -> per-frame H/A/D probabilities.

Same features, same degraded external matches, same leave-one-match-out protocol as cv.py, so the
two are directly comparable. Out-of-fold probabilities go to cache/oof_seq/.

    python seq.py --cv                  # leave-one-match-out
    python seq.py --final --tag _v2     # train on all 7, save cache/seq_final_v2.pt (fine on the HPC)
    python seq.py --predict --tag _v2   # laptop: BAR-ATH probabilities -> cache/bar_ath_proba_seq_v2.npy
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from mtah import dataset, decode, idsse

TRAIN_SEEDS = (0, 1)
EVAL_SEED = 99
T = 1000          # window length in frames (40 s)
BATCH = 32
STEPS = 3000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOUND_K = 6      # boundary target is smeared +-6 frames (0.24 s) around each change of label
AUX_W = 0.5


class Net(nn.Module):
    def __init__(self, f, h=128, layers=2, aux=0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2 * f, h), nn.GELU(), nn.Dropout(0.1))
        self.conv = nn.Conv1d(h, h, 5, padding=2)
        self.gru = nn.GRU(h, h, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * h, 3 + aux)

    def forward(self, x):
        z = self.inp(x)
        z = z + torch.relu(self.conv(z.transpose(1, 2))).transpose(1, 2)
        z, _ = self.gru(z)
        return self.out(z)


class Norm:
    def fit(self, X):
        self.med = np.nanmedian(X[::50], 0)
        q = np.nanpercentile(X[::50], [10, 90], 0)
        self.scale = np.where(q[1] - q[0] > 1e-6, q[1] - q[0], 1.0)
        return self

    def __call__(self, X):
        Z = (X - self.med) / self.scale
        miss = np.isnan(Z)
        Z = np.clip(np.nan_to_num(Z, nan=0.0), -10, 10)
        return np.concatenate([Z, miss.astype(np.float32)], 1).astype(np.float32)


def load_copies(matches, seeds=TRAIN_SEEDS):
    out = []
    for m in matches:
        for s in seeds:
            for sw in (False, True):
                X, y, half, lab, names = dataset.external(m, s, sw)
                out.append((X, y.astype(np.int64)))
    return out


def boundary_target(y):
    """1.0 within BOUND_K frames of a change between two graded labels, -1 where ungraded."""
    y = np.asarray(y)
    ch = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0))
    b = np.zeros(len(y), np.float32)
    for c in ch:
        b[max(0, c - BOUND_K + 1):c + BOUND_K + 1] = 1.0
    b[y < 0] = -1.0
    return b


def train(copies, steps=STEPS, seed=0, hidden=128, layers=2, aux=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = Norm().fit(np.concatenate([c[0][::7] for c in copies]))
    data = []
    while copies:
        X, y = copies.pop()
        data.append((torch.from_numpy(norm(X)).half(), torch.from_numpy(y), torch.from_numpy(boundary_target(y))))
        del X
    net = Net(data[0][0].shape[1] // 2, hidden, layers, aux).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    net.train()
    for step in range(steps):
        xb, yb, bb = [], [], []
        for _ in range(BATCH):
            X, y, b = data[rng.integers(len(data))]
            s = int(rng.integers(0, len(y) - T))
            xb.append(X[s:s + T]); yb.append(y[s:s + T]); bb.append(b[s:s + T])
        xb = torch.stack(xb).to(DEV).float(); yb = torch.stack(yb).to(DEV)
        out = net(xb)
        loss = lossf(out[..., :3].reshape(-1, 3), yb.reshape(-1))
        if aux:
            bb = torch.stack(bb).to(DEV)
            m = bb >= 0
            loss = loss + AUX_W * nn.functional.binary_cross_entropy_with_logits(out[..., 3][m], bb[m])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0:
            print("  step", step, "loss %.4f" % loss.item(), flush=True)
    net.eval()
    return net, norm


@torch.no_grad()
def predict(net, norm, X, hop=500):
    Z = torch.from_numpy(norm(X))
    n = len(Z)
    acc = torch.zeros(n, net.out.out_features); cnt = torch.zeros(n, 1)
    # triangular weights so window edges (little context) count less
    w = torch.from_numpy(np.minimum(np.arange(1, T + 1), np.arange(T, 0, -1)).astype(np.float32))[:, None]
    starts = list(range(0, max(1, n - T + 1), hop))
    if starts[-1] != max(0, n - T):
        starts.append(max(0, n - T))
    for i in range(0, len(starts), 16):
        batch = starts[i:i + 16]
        xb = torch.stack([Z[s:s + T] for s in batch]).to(DEV)
        o = net(xb)
        p = torch.cat([torch.softmax(o[..., :3], -1), torch.sigmoid(o[..., 3:])], -1).cpu()
        for s, ps in zip(batch, p):
            acc[s:s + T] += ps * w[:len(ps)]
            cnt[s:s + T] += w[:len(ps)]
    return (acc / cnt.clamp(min=1e-6)).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--matches", nargs="*", default=idsse.MATCHES)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(TRAIN_SEEDS))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--aux", type=int, default=0, help="1 = extra head predicting label changes")
    ap.add_argument("--tta", action="store_true", help="average with the home/away-swapped view")
    ap.add_argument("--init", type=int, default=0, help="model init / batch-sampling seed")
    ap.add_argument("--T", type=int, default=T)
    ap.add_argument("--tag", default="", help="suffix for the OOF directory, e.g. _v2")
    args = ap.parse_args()
    T = args.T
    arch = dict(hidden=args.hidden, layers=args.layers, aux=args.aux)
    if args.cv:
        oof = os.path.join("cache", "oof_seq" + args.tag)
        os.makedirs(oof, exist_ok=True)
        if args.tta:
            os.makedirs(oof + "_plain", exist_ok=True)
        for held in args.matches:
            t0 = time.time()
            net, norm = train(load_copies([m for m in idsse.MATCHES if m != held], args.seeds), args.steps, seed=args.init, **arch)
            X, y, half, lab, _ = dataset.external(held, EVAL_SEED)
            proba = predict(net, norm, X)
            if args.tta:
                p = predict(net, norm, dataset.external(held, EVAL_SEED, True)[0])
                plain, proba = proba, (proba + p[:, [1, 0] + list(range(2, p.shape[1]))]) / 2
                np.savez(os.path.join(oof + "_plain", held + ".npz"), proba=plain, half=half, labels=lab)
            np.savez(os.path.join(oof, held + ".npz"), proba=proba, half=half, labels=lab)
            r = decode.score(lab, decode.decode(proba[:, :3], half, 15, 20))
            print(held, "%.0fs" % (time.time() - t0), {k: round(v, 3) for k, v in r.items()}, flush=True)
    if args.final:
        net, norm = train(load_copies(idsse.MATCHES, args.seeds), args.steps, seed=args.init, **arch)
        torch.save({"state": net.state_dict(), "med": norm.med, "scale": norm.scale, "arch": arch, "T": T,
                    "feat": dataset.FEAT_VERSION}, os.path.join("cache", "seq_final%s.pt" % args.tag))
        print("saved")
    if args.predict:
        # BAR-ATH inference runs on the laptop only; the checkpoint may come from the HPC
        ck = torch.load(os.path.join("cache", "seq_final%s.pt" % args.tag), weights_only=False)
        assert ck["feat"] == dataset.FEAT_VERSION, "set MTAH_FEAT to the checkpoint's feature version"
        T = ck["T"]
        norm = Norm(); norm.med, norm.scale = ck["med"], ck["scale"]
        X, half, _ = dataset.challenge()
        net = Net(X.shape[1], ck["arch"]["hidden"], ck["arch"]["layers"], ck["arch"].get("aux", 0)).to(DEV)
        net.load_state_dict(ck["state"]); net.eval()
        proba = predict(net, norm, X)
        if args.tta:
            p = predict(net, norm, dataset.challenge(swapped=True)[0])
            proba = (proba + p[:, [1, 0] + list(range(2, p.shape[1]))]) / 2
        np.save(os.path.join("cache", "bar_ath_proba_seq%s%s.npy" % (args.tag, "_tta" if args.tta else "")), proba)
        print("predicted")
