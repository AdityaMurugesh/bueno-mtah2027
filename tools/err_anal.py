import glob, sys, numpy as np
from mtah import decode
pm = decode.pm
d = sys.argv[1] if len(sys.argv) > 1 else "cache/oof"
ct, cd = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (3, 80)
bins = [0, 25, 50, 125, 250, 500, 10**9]; hit = np.zeros(len(bins)-1); tot = np.zeros(len(bins)-1)
fp_len = []
for f in sorted(glob.glob(d + "/*.npz")):
    z = np.load(f); gt = pm.label_codes(z["labels"]); pr = pm.label_codes(decode.decode(z["proba"], z["half"], ct, cd))
    runs = pm.valid_runs(gt); gs = pm.possession_spells(gt, runs); ps = pm.possession_spells(pr, runs)
    ov = pm.spell_overlaps(gs, ps); flags = pm.matched_gt_flags(ov, len(gs), 0.5)
    matched_p = set()
    # recompute pred matched
    used=set(); mg=set()
    for iou,i,j in ov:
        if iou<0.5: break
        if i in mg or j in used: continue
        mg.add(i); used.add(j)
    for k,(a,b,c) in enumerate(gs):
        L = b-a; idx = np.searchsorted(bins, L, side="right")-1
        tot[idx]+=1; hit[idx]+=flags[k]
    fp_len += [b-a for j,(a,b,c) in enumerate(ps) if j not in used]
for i in range(len(tot)):
    print("gt len %5.1f-%5.1fs  n %4d  recall@0.5 %.2f" % (bins[i]/25, min(bins[i+1],99999)/25, tot[i], hit[i]/tot[i]))
fp_len = np.array(fp_len); print("unmatched preds", len(fp_len), "median len %.1fs" % (np.median(fp_len)/25), "<2s %.2f" % np.mean(fp_len<50))
