"""(1) Scatter each dataset by best-neighbor-detector ap vs best-marginal-detector ap -> do the two
anomaly geometries form two blobs? (2) Push for a stronger geometry signal: engineer more normal-only
structure features and cross-validate a predictor of the marginal-vs-neighbor oracle."""
import json, glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import skew, kurtosis
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad"
BUN = os.path.join(S, "streamline", "bundle_v2"); DEEP = lambda d: d.startswith("D_")
def ld(d):
    J={}
    for fn in glob.glob(os.path.join(d,"shard_*.jsonl")):
        for ln in open(fn):
            try: r=json.loads(ln)
            except: continue
            if "key" in r: J[r["key"]]=r
    return J
clstd = ld(os.path.join(S,"streamline","deepres_v2_clstd")); clfix = ld(os.path.join(S,"streamline","deepres_v2_clfix"))
lb = pd.read_csv(os.path.join(S,"streamline","STREAM_V3_LEADERBOARD.csv"))
keep = lb[(lb.oracle_ap>0.05)&(lb.reg_random>0.02)&(lb.n_h>=100)].set_index("key")
MARG = lambda d: d.startswith(("HBOS","ECOD","COPOD")); NEIGH = lambda d: d.startswith(("LOF","KNN"))
rows=[]
for f in sorted(glob.glob(os.path.join(BUN,"*.npz"))):
    key=os.path.basename(f)[:-4]
    if key not in keep.index or key not in clstd: continue
    ap={d:v for d,v in clstd[key]["ap"].items() if not DEEP(d)}
    m=[ap[d] for d in ap if MARG(d)]; ne=[ap[d] for d in ap if NEIGH(d)]
    if not m or not ne: continue
    bm, bn = max(m), max(ne)
    z=np.load(f); Xtr=z["Xtr"].astype("float64"); Zt=StandardScaler().fit_transform(Xtr); n,d=Zt.shape
    pca=PCA().fit(Zt); cum=np.cumsum(pca.explained_variance_ratio_); effdim=(np.searchsorted(cum,0.90)+1)/d
    C=np.corrcoef(Zt.T); mc=np.nanmean(np.abs(C[np.triu_indices(d,1)])) if d>1 else 0.0
    nn=NearestNeighbors(n_neighbors=min(6,n-1)).fit(Zt); dd=nn.kneighbors(Zt)[0][:,1:].mean(1); cv=dd.std()/(dd.mean()+1e-9)
    a1=[v for x,v in clfix[key]["a1"].items() if not DEEP(x)]; dep_sep=max(a1) if a1 else 0.5
    sk=np.nanmean(np.abs(skew(Zt,axis=0))); ku=np.nanmean(kurtosis(Zt,axis=0))
    disc=np.mean([len(np.unique(Xtr[:,j]))<20 for j in range(d)])   # frac near-discrete features
    top1=pca.explained_variance_ratio_[0]                            # concentration on 1st PC
    rows.append(dict(key=key, bm=bm, bn=bn, G=bm-bn, marg_oracle=int(bm>bn),
                     effdim=effdim, meancorr=mc, knn_cv=cv, dep_sep=dep_sep, skew=sk, kurt=ku,
                     disc=disc, top1=top1, logd=np.log10(d)))
df = pd.DataFrame(rows)
print(f"tasks: {len(df)}  marginal-oracle: {df.marg_oracle.sum()}  neighbor-oracle: {(1-df.marg_oracle).sum()}")
# (1) scatter
fig,ax=plt.subplots(1,2,figsize=(12,5.2))
c=np.where(df.marg_oracle==1,"#c0392b","#1b6ca8")
ax[0].scatter(df.bn, df.bm, c=c, s=26, alpha=.75, edgecolors="none")
lim=[min(df.bn.min(),df.bm.min())-.02, max(df.bn.max(),df.bm.max())+.02]
ax[0].plot(lim,lim,"k--",lw=.8,alpha=.5); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("best NEIGHBOR detector ap_norm (LOF/KNN)"); ax[0].set_ylabel("best MARGINAL detector ap_norm (HBOS/ECOD/COPOD)")
ax[0].set_title("each dataset: marginal vs neighbor detectability\n(red=marginal oracle, blue=neighbor oracle)")
ax[1].scatter(df.effdim, df.G, c=c, s=26, alpha=.75, edgecolors="none")
ax[1].axhline(0,color="k",lw=.8,alpha=.5); ax[1].set_xlabel("normal effective dimensionality (label-free)")
ax[1].set_ylabel("geometry G = marginal ap - neighbor ap"); ax[1].set_title("normal structure vs anomaly geometry")
fig.tight_layout(); out=os.path.join(S,"TWO_BLOBS.png"); fig.savefig(out,dpi=140); print("saved",out)
# (2) stronger signal: cross-validated predictor of marginal-vs-neighbor oracle
feats=["effdim","meancorr","knn_cv","dep_sep","skew","kurt","disc","top1","logd"]
X=df[feats].fillna(df[feats].median()).values; y=df.marg_oracle.values
Xs=StandardScaler().fit_transform(X)
auc=cross_val_score(LogisticRegression(max_iter=1000,class_weight="balanced"),Xs,y,cv=5,scoring="roc_auc")
print(f"\n=== stronger signal: 5-fold CV predicting marginal-oracle from {len(feats)} normal-structure features ===")
print(f"  CV ROC-AUC = {auc.mean():.3f} +/- {auc.std():.3f}   (0.5=no signal; univariate best effdim was ~0.60)")
lr=LogisticRegression(max_iter=1000,class_weight="balanced").fit(Xs,y)
print("  feature weights:", {f:round(w,2) for f,w in sorted(zip(feats,lr.coef_[0]),key=lambda x:-abs(x[1]))})
