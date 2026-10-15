"""C6 reconciliation: does the streamline pipeline's own load + basic-gate + n_norm>=800 filter,
applied to RAW ovrbench/oddbench npz, reproduce the manifest counts (ovr 131, odd 107)?
Cheap test: no severity/n_eff. Just load, split normals 60/40 (cap 6000), count."""
import os, glob, numpy as np
ROOT = r"E:\Projects\Submitted\ADRank"
MIN_NORM = 800

def tab_split_counts(Xn, Xa):
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); k = int(0.6 * len(idx))
    n_tr = k; n_hold = len(idx) - k
    return n_tr, n_hold, len(Xa)

for corp, avail in [("ovrbench", 754), ("oddbench", 187)]:
    files = sorted(glob.glob(os.path.join(ROOT, "data", corp, "*.npz")))
    loaded = basic_ok = s5_ok = 0
    reasons = {}
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]))
            y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
        except Exception as e:
            reasons["load_error"] = reasons.get("load_error", 0) + 1; continue
        loaded += 1
        Xn, Xa = X[y == 0], X[y == 1]
        n_tr, n_hold, n_a = tab_split_counts(Xn, Xa)
        if n_tr < 40 or n_hold < 20 or n_a < 5:
            reasons["basic_gate(too_small)"] = reasons.get("basic_gate(too_small)", 0) + 1; continue
        basic_ok += 1
        if n_hold < MIN_NORM:
            reasons["n_norm<800"] = reasons.get("n_norm<800", 0) + 1; continue
        s5_ok += 1
    print(f"\n=== {corp}: available(npz)={len(files)} (Table1={avail}) ===")
    print(f"  loaded ok           : {loaded}")
    print(f"  pass basic size gate: {basic_ok}")
    print(f"  pass n_norm>=800(S5): {s5_ok}   <-- compare to manifest (ovr 131 / odd 107)")
    print(f"  drop reasons: {reasons}")
