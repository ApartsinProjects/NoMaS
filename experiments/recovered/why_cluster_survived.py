"""Why did SpectralImaging's cluster anomalies survive the (linear) hardening filter?
Compare how separable the REAL anomalies are by: (a) best single raw feature, (b) best PCA-whitened
component, (c) the hardening severity statistic itself (frac flagged at 5% FP), (d) kNN distance to
normals (non-linear / neighbor). Hypothesis: (a)(b)(c) low (linear filter can't flag) but (d) high
(neighbor detector separates), i.e. non-linearly separable, linearly subtle -> correctly retained."""
import os, sys, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2")
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1]-tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right")/m
        tail = -np.log(np.maximum(2*np.minimum(fb, 1-fb), 1.0/(2*m)))
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt/max(cnt.sum(),1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins-1)
        sev = np.maximum(sev, np.maximum(tail, -np.log(dens[b]+1e-9)))
    return sev
def best_feat_auc(Xn, Xa):
    y = np.r_[np.zeros(len(Xn)), np.ones(len(Xa))]; best = 0.5
    for j in range(Xn.shape[1]):
        v = np.r_[Xn[:, j], Xa[:, j]]
        try: a = max(roc_auc_score(y, v), roc_auc_score(y, -v))
        except Exception: a = 0.5
        best = max(best, a)
    return best
for key in sys.argv[1:]:
    z = np.load(os.path.join(BUN, key+".npz"))
    Xtr, Xval, Xh = z["Xtr"].astype("float64"), z["Xval"].astype("float64"), z["Xh"].astype("float64")
    ss = StandardScaler().fit(Xtr); Ztr, Zn, Za = ss.transform(Xtr), ss.transform(Xval), ss.transform(Xh)
    # (a) best single raw feature
    a_feat = best_feat_auc(Xval, Xh)
    # (b) best PCA-whitened component
    pca = PCA(whiten=True, random_state=0).fit(Ztr); Pn, Pa = pca.transform(Zn), pca.transform(Za)
    a_pca = best_feat_auc(Pn, Pa)
    # (c) hardening severity: frac of real anomalies flagged at the filter's 5% FP threshold (orig OR pca)
    so_n, so_a = severity(Xtr, Xval), severity(Xtr, Xh); tho = np.quantile(so_n, 0.95)
    Ptr = pca.transform(Ztr); sp_n, sp_a = severity(Ptr, Pn), severity(Ptr, Pa); thp = np.quantile(sp_n, 0.95)
    flagged = ((so_a > tho) | (sp_a > thp)).mean()
    # (d) kNN distance to normals (neighbor / non-linear)
    nn = NearestNeighbors(n_neighbors=min(5, len(Ztr)-1)).fit(Ztr)
    sc = lambda X: nn.kneighbors(X)[0].mean(1)
    y = np.r_[np.zeros(len(Zn)), np.ones(len(Za))]; a_knn = roc_auc_score(y, np.r_[sc(Zn), sc(Za)])
    print(f"\n=== {key} (d={Xtr.shape[1]}, n_anom={len(Xh)}) ===")
    print(f"  (a) best SINGLE-FEATURE AUC        : {a_feat:.3f}")
    print(f"  (b) best PCA-WHITENED comp AUC     : {a_pca:.3f}")
    print(f"  (c) hardening severity: frac real anomalies flagged at 5% FP : {100*flagged:.0f}%  (survive => NOT flagged)")
    print(f"  (d) kNN-DISTANCE (neighbor) AUC    : {a_knn:.3f}")
    print(f"  -> {'CONFIRMS: non-linearly separable (kNN high) but linearly subtle (feat/PCA/severity low)' if a_knn-max(a_feat,a_pca)>0.1 else 'not a clean non-linear-only case'}")
