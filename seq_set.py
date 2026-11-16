"""Set encoder over every tracked player + the hand features -> BiGRU -> per-frame H/A/D.

The hand features summarise each team; the set encoder lets the model see who exactly is near the
ball and how fast they move, which is what short possessions hinge on. Same protocol as seq.py.

    python seq_set.py --cv --tag _set            # leave-one-match-out, OOF -> cache/oof_seq_set
    python seq_set.py --final --tag _set         # train on all 7 (HPC is fine), save cache/seq_final_set.pt
    python seq_set.py --predict --tag _set       # laptop: BAR-ATH -> cache/bar_ath_proba_seq_set.npy
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from mtah import dataset, decode, idsse, raw
from seq import AUX_W, Norm, boundary_target

EVAL_SEED = 99
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, W = 105.0, 68.0
AMP = os.environ.get("MTAH_AMP", "1") == "1"


class SetNet(nn.Module):
    def __init__(self, f, h=128, layers=2, d=64, aux=0):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.att = nn.Linear(d, 1)
        self.inp = nn.Sequential(nn.Linear(2 * f + 6 * d + 2, h), nn.GELU(), nn.Dropout(0.1))
        self.conv = nn.Conv1d(h, h, 5, padding=2)
        self.gru = nn.GRU(h, h, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * h, 3 + aux)

    def pool(self, xy, sp, team, ball):
        """xy (B,T,K,2), sp (B,T,K), team (B,K), ball (B,T,2) -> (B,T,6d+2)."""
        bpres = ~torch.isnan(ball[..., :1])                          # (B,T,1)
        pres = ~torch.isnan(xy[..., 0]) & (team[:, None, :] >= 0)    # (B,T,K)
        rel = (xy - ball[:, :, None, :]) / 10.0
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        bp = bpres[:, :, None, :].expand_as(dist).float()
        t = torch.cat([xy[..., :1] / L - 0.5, xy[..., 1:] / W - 0.5, rel, dist, torch.exp(-dist * 2),
                       sp[..., None] / 5.0, bp], -1)
        t = torch.nan_to_num(t, nan=0.0).clamp(-10, 10)
        e = self.tok(t)                                              # (B,T,K,d)
        a = self.att(e).squeeze(-1)                                  # (B,T,K)
        outs, counts = [], []
        for side in (0, 1):
            m = pres & (team[:, None, :] == side)
            mf = m[..., None].float()
            cnt = mf.sum(2)
            mean = (e * mf).sum(2) / cnt.clamp(min=1)
            mx = e.masked_fill(~m[..., None], -1e4).max(2).values * (cnt > 0)
            w = torch.softmax(a.masked_fill(~m, -1e4), -1)[..., None] * mf
            outs += [mean, mx, (e * w).sum(2)]
            counts.append(cnt / 11.0)
        return torch.cat(outs + counts, -1)

    def forward(self, x, xy, sp, team, ball):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP and x.is_cuda):
            z = self.inp(torch.cat([x, self.pool(xy, sp, team, ball)], -1))
            z = z + torch.relu(self.conv(z.transpose(1, 2))).transpose(1, 2)
        z, _ = self.gru(z.float())  # cuDNN GRU in fp32: bf16 RNNs fall off the fast path
        return self.out(z)


def to_dev(r):
    return (torch.from_numpy(r["xy"]).to(DEV), torch.from_numpy(r["sp"]).to(DEV),
            torch.from_numpy(r["team"].astype(np.int64)).to(DEV), torch.from_numpy(r["ball"]).to(DEV))


def load_copies(matches, seeds):
    out = []
    for m in matches:
        for s in seeds:
            for sw in (False, True):
                X, y, half, lab, names = dataset.external(m, s, sw)
                out.append((X, y.astype(np.int64), raw.external(m, s, sw)))
    return out


def train(copies, steps, T, seed=0, batch=32, net=None, norm=None, lr=2e-3, target=(), target_frac=0.0, **arch):
    """Train a SetNet (or keep training `net` with its `norm`). `target`: extra (X, y, raw) copies, e.g. pseudo-labelled,
    drawn with probability `target_frac` per window."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if norm is None:
        norm = Norm().fit(np.concatenate([c[0][::7] for c in copies]))

    def load(cs):
        out = []
        while cs:
            X, y, r = cs.pop()
            out.append((torch.from_numpy(norm(X)).half().to(DEV), torch.from_numpy(y).to(DEV)) + to_dev(r)
                       + (torch.from_numpy(boundary_target(y)).to(DEV),))
            del X
        return out
    data, tdata = load(copies), load(list(target))
    if net is None:
        net = SetNet(data[0][0].shape[1] // 2, **arch).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    net.train()
    for step in range(steps):
        xs, ys, xys, sps, teams, balls, bs = [], [], [], [], [], [], []
        for _ in range(batch):
            src = tdata if tdata and rng.random() < target_frac else data
            X, y, xy, sp, team, ball, b = src[rng.integers(len(src))]
            s = int(rng.integers(0, len(y) - T))
            xs.append(X[s:s + T]); ys.append(y[s:s + T]); xys.append(xy[s:s + T]); sps.append(sp[s:s + T])
            teams.append(team); balls.append(ball[s:s + T]); bs.append(b[s:s + T])
        logits = net(torch.stack(xs).float(), torch.stack(xys).float(), torch.stack(sps).float(),
                     torch.stack(teams), torch.stack(balls).float())
        logits = logits.float()
        loss = lossf(logits[..., :3].reshape(-1, 3), torch.stack(ys).reshape(-1))
        if logits.shape[-1] > 3:
            bb = torch.stack(bs)
            m = bb >= 0
            loss = loss + AUX_W * nn.functional.binary_cross_entropy_with_logits(logits[..., 3][m], bb[m])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0:
            print("  step", step, "loss %.4f" % loss.item(), flush=True)
    net.eval()
    return net, norm


@torch.no_grad()
def predict(net, norm, X, r, T, hop=500):
    Z = torch.from_numpy(norm(X)).to(DEV)
    xy, sp, team, ball = to_dev(r)
    n = len(Z)
    acc = torch.zeros(n, net.out.out_features, device=DEV); cnt = torch.zeros(n, 1, device=DEV)
    w = torch.from_numpy(np.minimum(np.arange(1, T + 1), np.arange(T, 0, -1)).astype(np.float32))[:, None].to(DEV)
    starts = list(range(0, max(1, n - T + 1), hop))
    if starts[-1] != max(0, n - T):
        starts.append(max(0, n - T))
    for i in range(0, len(starts), 16):
        b = starts[i:i + 16]
        o = net(torch.stack([Z[s:s + T] for s in b]), torch.stack([xy[s:s + T] for s in b]).float(),
                torch.stack([sp[s:s + T] for s in b]).float(), team[None].expand(len(b), -1),
                torch.stack([ball[s:s + T] for s in b]).float()).float()
        p = torch.cat([torch.softmax(o[..., :3], -1), torch.sigmoid(o[..., 3:])], -1)
        for s, ps in zip(b, p):
            acc[s:s + T] += ps * w[:len(ps)]
            cnt[s:s + T] += w[:len(ps)]
    return (acc / cnt.clamp(min=1e-6)).cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--matches", nargs="*", default=idsse.MATCHES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--aux", type=int, default=0, help="1 = extra head predicting label changes")
    ap.add_argument("--tta", action="store_true", help="average with the home/away-swapped view")
    ap.add_argument("--init", type=int, default=0, help="model init / batch-sampling seed")
    ap.add_argument("--tag", default="_set")
    ap.add_argument("--pseudo_steps", type=int, default=0, help="CV: self-train on the held-out match's own decoded labels")
    ap.add_argument("--pseudo_conf", type=float, default=0.0, help="mask pseudo-labels where max p < this")
    args = ap.parse_args()
    arch = dict(h=args.hidden, layers=args.layers, aux=args.aux)
    if args.cv:
        oof = os.path.join("cache", "oof_seq" + args.tag)
        os.makedirs(oof, exist_ok=True)
        if args.tta:
            os.makedirs(oof + "_plain", exist_ok=True)
        for held in args.matches:
            t0 = time.time()
            net, norm = train(load_copies([m for m in idsse.MATCHES if m != held], args.seeds), args.steps, args.T, seed=args.init, **arch)
            X, y, half, lab, _ = dataset.external(held, EVAL_SEED)
            proba = predict(net, norm, X, raw.external(held, EVAL_SEED), args.T)
            if args.tta:
                p = predict(net, norm, dataset.external(held, EVAL_SEED, True)[0], raw.external(held, EVAL_SEED, True), args.T)
                plain, proba = proba, (proba + p[:, [1, 0] + list(range(2, p.shape[1]))]) / 2
                np.savez(os.path.join(oof + "_plain", held + ".npz"), proba=plain, half=half, labels=lab)
            np.savez(os.path.join(oof, held + ".npz"), proba=proba, half=half, labels=lab)
            r = decode.score(lab, decode.decode(proba[:, :3], half, 10, 80))
            print(held, "%.0fs" % (time.time() - t0), {k: round(v, 3) for k, v in r.items()}, flush=True)
            if args.pseudo_steps:
                # self-training: the held-out match's own decoded labels (no ground truth), both team views
                yp = np.array([dataset.CODE[c] for c in decode.decode(proba[:, :3], half, 3, 20)], np.int64)
                yp[proba[:, :3].max(1) < args.pseudo_conf] = -1
                ysw = np.where(yp == 0, 1, np.where(yp == 1, 0, yp))
                target = [(X, yp, raw.external(held, EVAL_SEED)),
                          (dataset.external(held, EVAL_SEED, True)[0], ysw, raw.external(held, EVAL_SEED, True))]
                mix = load_copies([m for m in idsse.MATCHES if m != held], args.seeds[:2])
                net, _ = train(mix, args.pseudo_steps, args.T, seed=args.init + 100, net=net, norm=norm, lr=5e-4,
                               target=target, target_frac=0.5)
                p1 = predict(net, norm, X, raw.external(held, EVAL_SEED), args.T)
                p2 = predict(net, norm, dataset.external(held, EVAL_SEED, True)[0], raw.external(held, EVAL_SEED, True), args.T)
                pp = (p1 + p2[:, [1, 0] + list(range(2, p2.shape[1]))]) / 2
                os.makedirs(oof + "_pl", exist_ok=True)
                np.savez(os.path.join(oof + "_pl", held + ".npz"), proba=pp, half=half, labels=lab)
                r = decode.score(lab, decode.decode(pp[:, :3], half, 3, 20))
                print("  after self-training", {k: round(v, 3) for k, v in r.items()}, flush=True)
            del net
            torch.cuda.empty_cache()
    if args.final:
        net, norm = train(load_copies(idsse.MATCHES, args.seeds), args.steps, args.T, seed=args.init, **arch)
        torch.save({"state": net.state_dict(), "med": norm.med, "scale": norm.scale, "arch": arch, "T": args.T,
                    "feat": dataset.FEAT_VERSION}, os.path.join("cache", "seq_final%s.pt" % args.tag))
        print("saved")
    if args.predict:
        ck = torch.load(os.path.join("cache", "seq_final%s.pt" % args.tag), weights_only=False)
        assert ck["feat"] == dataset.FEAT_VERSION, "set MTAH_FEAT to the checkpoint's feature version"
        norm = Norm(); norm.med, norm.scale = ck["med"], ck["scale"]
        X, half, _ = dataset.challenge()
        net = SetNet(X.shape[1], **ck["arch"]).to(DEV)
        net.load_state_dict(ck["state"]); net.eval()
        proba = predict(net, norm, X, raw.challenge(), ck["T"])
        if args.tta:
            p = predict(net, norm, dataset.challenge(swapped=True)[0], raw.challenge(swapped=True), ck["T"])
            proba = (proba + p[:, [1, 0] + list(range(2, p.shape[1]))]) / 2
        np.save(os.path.join("cache", "bar_ath_proba_seq%s%s.npy" % (args.tag, "_tta" if args.tta else "")), proba)
        print("predicted")
