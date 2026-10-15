# -*- coding: utf-8 -*-
"""Do deep's REAL-anomaly failures sit BETWEEN clusters (in inter-mode voids)?
Pick datasets where a deep detector loses badly to the best classical (from debias ap). Refit AE
(reconstruction) + DeepSVDD (if available) + LOF locally on standardized train normals. For each REAL
test anomaly, get each detector's rank; classify the anomaly geometry as 'between-modes' vs 'other'.
Test: among anomalies LOF catches, are the ones DEEP MISSES concentrated between modes?

Betweenness: for anomaly a, take its two nearest train-cluster centroids c1,c2 (k by silhouette).
t = <a-c1, c2-c1>/|c2-c1|^2 ; perp = |a - (c1 + t(c2-c1))|. 'between' if 0.25<=t<=0.75 AND
perp <= 0.5*|c2-c1| (on the bridge, not off to one side). Also report nearest-normal distance.
"""
import json, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from pyod.models.lof import LOF
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2")
DEEP = lambda d: d.startswith("D_")
# --- load debias ap to find where deep loses to classical ---
J = {}
for fn in glob.glob(os.path.join(S, "streamline", "debias_results", "**", "shard_*.jsonl"), recursive=True):
    for ln in open(fn):
        try: r = json.loads(ln)
        except: continue
        if r.get("ap"): J[r["key"]] = r
lb = pd.read_csv(os.path.join(S, "streamline", "STREAM_V3_LEADERBOARD.csv"))
keep = set(lb[(lb.oracle_ap > 0.05) & (lb.reg_random > 0.02) & (lb.n_h >= 100)].key)
cand = []
for k, r in J.items():
    if k not in keep: continue
    ap = r["ap"]; dcls = [v for d, v in ap.items() if not DEEP(d)]; ddeep = [v for d, v in ap.items() if DEEP(d)]
    if not dcls or not ddeep: continue
    cand.append((k, max(dcls) - max(ddeep), max(dcls), max(ddeep)))
cand.sort(key=lambda x: -x[1])                         # biggest classical-over-deep gap first
picks = [c[0] for c in cand[:6]]
print("datasets where deep loses most to classical (best_cls_ap - best_deep_ap):")
for k, g, c, d in cand[:6]: print(f"  {k[:40]:40s} gap={g:.3f}  cls={c:.3f} deep={d:.3f}")

def deep_scores(Ztr, Zall):
    out = {}
    try:
        from pyod.models.auto_encoder import AutoEncoder
        m = AutoEncoder(epoch_num=50, verbose=0); m.fit(Ztr); out["AE"] = m.decision_function(Zall)
    except Exception as e: print("   AE failed:", e)
    try:
        from deepod.models import DeepSVDD
        import contextlib, io
        m = DeepSVDD(epochs=40, random_state=0, device="cpu")
        with contextlib.redirect_stdout(io.StringIO()): m.fit(Ztr)
        out["DeepSVDD"] = m.decision_function(Zall)
    except Exception as e: print("   DeepSVDD failed:", e)
    return out

def pct(s):  # percentile rank 0..1 (high = looks anomalous)
    s = np.asarray(s, float); order = s.argsort().argsort(); return order / max(len(s) - 1, 1)

rows = []
for key in picks:
    f = os.path.join(BUN, key + ".npz")
    if not os.path.exists(f): continue
    z = np.load(f); Xtr = z["Xtr"].astype("float64"); Xtn = z["Xtn"].astype("float64"); Xh = z["Xh"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr = ss.transform(Xtr); Ztn = ss.transform(Xtn); Zh = ss.transform(Xh)
    Zall = np.vstack([Ztn, Zh]); n_norm = len(Ztn)
    # cluster train normals
    best_k, best_s = 2, -1
    for k in range(2, 7):
        if len(Ztr) < k * 10: break
        lab = KMeans(k, n_init=5, random_state=0).fit_predict(Ztr)
        try: sc = silhouette_score(Ztr, lab)
        except Exception: sc = -1
        if sc > best_s: best_s, best_k = sc, k
    cen = KMeans(best_k, n_init=5, random_state=0).fit(Ztr).cluster_centers_
    nn = NearestNeighbors(n_neighbors=1).fit(Ztr)
    d_nn_all = nn.kneighbors(Zall, return_distance=True)[0][:, 0]
    d_nn_norm_med = np.median(d_nn_all[:n_norm]) + 1e-9
    # detectors
    lof = LOF(n_neighbors=20).fit(Ztr); lof_s = pct(lof.decision_function(Zall))
    dss = deep_scores(Ztr, Zall);
    if not dss: continue
    # betweenness for each anomaly
    def betweenness(a):
        d2 = np.linalg.norm(cen - a, axis=1); o = d2.argsort()[:2]
        if len(o) < 2: return 0
        c1, c2 = cen[o[0]], cen[o[1]]; seg = c2 - c1; L2 = seg @ seg + 1e-12
        t = (a - c1) @ seg / L2; proj = c1 + t * seg; perp = np.linalg.norm(a - proj)
        return int(0.25 <= t <= 0.75 and perp <= 0.5 * np.sqrt(L2))
    Za = Zall[n_norm:]                                  # real anomalies only
    btw = np.array([betweenness(a) for a in Za])
    dnn_a = d_nn_all[n_norm:] / d_nn_norm_med           # anomaly off-manifold dist (in normal-median units)
    lof_a = lof_s[n_norm:]
    for name, sc in dss.items():
        deep_a = pct(sc)[n_norm:]
        missed_by_deep = deep_a < 0.5                    # deep ranks it in lower (normal-looking) half
        caught_by_lof = lof_a >= 0.5
        A = missed_by_deep & caught_by_lof               # deep fails where LOF succeeds
        B = ~missed_by_deep                              # deep catches it
        rows.append(dict(key=key[:30], deep=name, k=best_k, n_anom=len(Za),
                         deep_miss_lof_catch=int(A.sum()), deep_catch=int(B.sum()),
                         btw_frac_A=round(btw[A].mean(), 2) if A.sum() else None,
                         btw_frac_B=round(btw[B].mean(), 2) if B.sum() else None,
                         dnn_A=round(dnn_a[A].mean(), 2) if A.sum() else None,
                         dnn_B=round(dnn_a[B].mean(), 2) if B.sum() else None))
df = pd.DataFrame(rows)
print("\n=== per (dataset, deep model): geometry of deep's failures ===")
print("btw_frac = fraction of anomalies sitting BETWEEN two modes; dnn = nearest-normal dist (normal-median units)")
print("A = anomalies DEEP MISSES but LOF CATCHES ; B = anomalies DEEP CATCHES")
print(df.to_string(index=False))
# pooled test
print("\n=== pooled ===")
for name in df.deep.unique():
    d = df[df.deep == name]
    a = d.btw_frac_A.dropna(); b = d.btw_frac_B.dropna()
    print(f"{name}: mean between-modes fraction  deep-misses={a.mean():.2f}  deep-catches={b.mean():.2f}  "
          f"(A more between-modes? {'YES' if a.mean() > b.mean() else 'no'})")
