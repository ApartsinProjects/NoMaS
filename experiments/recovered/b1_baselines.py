"""B1: promote always-IForest and best-fixed-detector to primary baselines.
Per-modality (ADBench+DAMI) and the OddBench all-nine-fixed table."""
import pandas as pd, numpy as np, json
from scipy.stats import wilcoxon

TRUE = pd.concat([pd.read_parquet("results/raw/modal_true_regimes.parquet"),
                  pd.read_parquet("results/raw/modal_true_dami.parquet")], ignore_index=True)
TRUE["seed"]=TRUE.seed.astype(int)
vf = pd.read_parquet("results/vs_fixed_per_dataset.parquet")  # modality,dataset,adrank,iforest,random
DETS=["CBLOF","COPOD","ECOD","HBOS","IForest","KNN","LODA","LOF","PCA"]

# per-dataset per-detector mean-over-seeds regret
def det_regrets(mod):
    tm=TRUE[TRUE.modality==mod]; rows=[]
    for ds in tm.dataset.unique():
        tt=tm[tm.dataset==ds].groupby("detector").true_auc.mean()
        if len(tt)<3: continue
        best=tt.max()
        rows.append({d: best-tt.get(d,np.nan) for d in DETS} | {"dataset":ds})
    return pd.DataFrame(rows).set_index("dataset")

print("=== B1a per-modality: ADRank vs always-IForest vs best-fixed vs random ===")
print(f"{'mod':8} {'nds':>3} {'ADRank':>7} {'IForest':>7} {'best-fixed':>16} {'random':>7} {'ADRvsIFO':>16} {'ADRvsBest':>16}")
for mod in ["tabular","cv","nlp","ts","dami"]:
    dr=det_regrets(mod)
    a=vf[vf.modality==mod].set_index("dataset").adrank.reindex(dr.index)
    detmean=dr.mean().sort_values()
    best=detmean.index[0]; brg=dr[best]
    ifo=dr["IForest"]; rnd=vf[vf.modality==mod].set_index("dataset")["random"].reindex(dr.index)
    def wl(x,y):  # x=adrank,y=baseline; wins=adrank lower
        w=(x<y).sum(); l=(x>y).sum(); t=len(x)-w-l
        try: p=wilcoxon(x,y).pvalue
        except Exception: p=np.nan
        return f"{w}/{l}/{t} p={p:.1e}"
    print(f"{mod:8} {len(dr):3d} {a.mean():7.4f} {ifo.mean():7.4f} {best+' '+format(brg.mean(),'.4f'):>16} "
          f"{rnd.mean():7.4f} {wl(a,ifo):>16} {wl(a,brg):>16}")

print("\n=== B1b OddBench (187 unseen): mean regret of every fixed detector vs ADRank ===")
ob=pd.read_parquet("results/oddbench_hash_ok.parquet")
recs=[]
for _,r in ob.iterrows():
    tr=json.loads(r.true_auc_json); best=max(tr.values())
    recs.append({d: best-tr.get(d,np.nan) for d in DETS} | {"dataset":r.dataset,"adrank":r.regret})
G=pd.DataFrame(recs).groupby("dataset").mean()
order=G[DETS].mean().sort_values()
print(f"  ADRank            {G.adrank.mean():.4f}   <- lowest of any label-free strategy")
for d,v in order.items():
    p=wilcoxon(G.adrank,G[d]).pvalue
    w=(G.adrank<G[d]).sum(); l=(G.adrank>G[d]).sum()
    print(f"  always-{d:8s}  {v:.4f}   ADRank vs: {w}W/{l}L  p={p:.2e}")
