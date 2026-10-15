"""Quantify how SPARC's synthetic anomalies differ from the REAL test anomalies, using only a normal-
data density model (kNN distance to train normals in standardized space). Per task, score normals
(Xval), synthetic probe (regenerated), and real anomalies (Xh). Report, aggregated over 151 tasks:
- accidental-normal rate: synthetic points that look normal (score below the 95th pct of normals)
- too-easy rate: synthetic vs real points far past the normal range (above 99th pct of normals)
- difficulty gap: are synthetic points systematically more extreme (easier) than real anomalies?
Design-time analysis to guide SPARC improvement; SPARC itself never sees the real anomalies."""
import glob, os, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2")

def _gen_beta(Xn, ns, beta=1.0, frac=0.4, seed=0):
    rng = np.random.default_rng(seed); n, d = Xn.shape; out = np.empty((ns, d)); Hs = []
    for j in range(d):
        col = Xn[:, j]; cnt, edges = np.histogram(col, bins=30); dens = cnt/max(cnt.sum(),1)
        lo, hi = np.percentile(col,2), np.percentile(col,98); ctr = (edges[:-1]+edges[1:])/2
        allowed = (cnt>0)&(ctr>=lo)&(ctr<=hi); w = np.where(allowed, np.maximum(dens,1e-6)**beta, 0.0)
        wn = w/w.sum() if w.sum()>0 else None; binidx = np.clip(np.digitize(col, edges[1:-1]),0,len(cnt)-1)
        Hs.append((wn,col,binidx))
    for r in range(ns):
        base = Xn[rng.integers(n)].copy()
        for j in rng.choice(d, max(1,int(frac*d)), replace=False):
            wn,col,binidx = Hs[j]
            base[j] = Xn[rng.integers(n),j] if wn is None else (lambda c: c[rng.integers(len(c))] if len(c) else Xn[rng.integers(n),j])(col[binidx==rng.choice(len(wn),p=wn)])
        out[r] = base
    return out

acc_norm, too_easy_s, too_easy_r = [], [], []
pct_syn, pct_real = [], []   # median percentile of synth / real within the normal score distribution
sep_syn, sep_real = [], []
for f in sorted(glob.glob(os.path.join(BUN, "*.npz"))):
    z = np.load(f); Xtr, Xval, Xh = z["Xtr"].astype("float64"), z["Xval"].astype("float64"), z["Xh"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr, Zval, Zh = ss.transform(Xtr), ss.transform(Xval), ss.transform(Xh)
    Zsyn = ss.transform(_gen_beta(Xval, 200))
    k = min(5, len(Ztr)-1)
    nn = NearestNeighbors(n_neighbors=k).fit(Ztr)
    s = lambda X: nn.kneighbors(X)[0].mean(1)   # kNN-distance normality score (higher = more anomalous)
    sn, sy, sr = s(Zval), s(Zsyn), s(Zh)
    p95, p99 = np.percentile(sn, 95), np.percentile(sn, 99)
    acc_norm.append(float((sy <= p95).mean()))          # synthetic that look normal
    too_easy_s.append(float((sy > p99).mean()))          # synthetic far past normals
    too_easy_r.append(float((sr > p99).mean()))          # real far past normals
    # percentile of each syn/real point within the normal score CDF
    pct_syn.append(float(np.mean([np.mean(sn <= v) for v in sy])))
    pct_real.append(float(np.mean([np.mean(sn <= v) for v in sr])))
    sep_syn.append(float(np.median(sy)/(np.median(sn)+1e-9)))
    sep_real.append(float(np.median(sr)/(np.median(sn)+1e-9)))
acc_norm, too_easy_s, too_easy_r = map(np.array, (acc_norm, too_easy_s, too_easy_r))
pct_syn, pct_real = np.array(pct_syn), np.array(pct_real)
print(f"tasks: {len(acc_norm)}  (normality = mean distance to 5 nearest train normals)")
print(f"\n=== accidental-normal rate: SPARC synthetic points that look normal (below 95th pct of normals) ===")
print(f"  mean {100*acc_norm.mean():.1f}% of the 200 synthetic points look normal (median {100*np.median(acc_norm):.1f}%)")
print(f"  on {100*(acc_norm>0.2).mean():.0f}% of tasks, >20% of the synthetic probe looks normal")
print(f"\n=== too-easy rate: points far past the normal range (above 99th pct of normals) ===")
print(f"  synthetic: mean {100*too_easy_s.mean():.1f}%   real anomalies: mean {100*too_easy_r.mean():.1f}%")
print(f"\n=== difficulty: median percentile of each point within the normal score distribution ===")
print(f"  synthetic median-percentile {100*np.mean(pct_syn):.1f}   real median-percentile {100*np.mean(pct_real):.1f}")
print(f"  (higher = more extreme/easier vs normals; compares how far off-normal synth vs real sit)")
print(f"\n=== separation ratio (median anomaly score / median normal score) ===")
print(f"  synthetic {np.mean(sep_syn):.2f}x   real {np.mean(sep_real):.2f}x normal")
print(f"\ntakeaways: acc-normal rate -> filter candidate; too-easy gap -> whether synth is more extreme than real")
