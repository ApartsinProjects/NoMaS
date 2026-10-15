"""Compute, from ONE consistent artifact (regimes), for every modality:
   - ADRank (frozen discriminative bank: smallest+random+hard x K30,50) regret
   - always-IForest regret, best-oracle-single-detector regret, random regret
   - paired Wilcoxon ADRank vs each.
Also recompute the corrected NLP ensemble-default number (smallest+random).
"""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon

TRUE = pd.concat([pd.read_parquet("results/raw/modal_true_regimes.parquet"),
                  pd.read_parquet("results/raw/modal_true_dami.parquet")], ignore_index=True)
PS   = pd.concat([pd.read_parquet("results/raw/modal_pseudo_regimes.parquet"),
                  pd.read_parquet("results/raw/modal_pseudo_dami.parquet")], ignore_index=True)
TRUE["seed"]=TRUE.seed.astype(int); PS["seed"]=PS.seed.astype(int); PS["K"]=PS.K.astype(int)

def agg_disc(pp):
    ms = pp.dropna(subset=["pseudo_auc"]).groupby(["regime","detector"]).pseudo_auc.mean()
    regs = pp.regime.unique(); w={}
    for r in regs:
        v = pp[pp.regime==r].dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean()
        w[r]=float(v.var())
    sden=sum(w.values()) or 1.0; w={r:w[r]/sden for r in regs}
    dets=ms.index.get_level_values(1).unique()
    out=pd.Series(0.0,index=dets); ws=0.0
    for r in ms.index.get_level_values(0).unique():
        out=out.add(ms.loc[r].rank(ascending=True)*w.get(r,0.0),fill_value=0.0); ws+=w.get(r,0.0)
    return out/(ws or 1.0)

def ens_smallest_random(pp):  # the paper's stated default (K=30 smallest+random ensemble)
    sub=pp[(pp.selection.isin(["smallest","random"]))&(pp.K==30)]
    return sub.dropna(subset=["pseudo_auc"]).groupby("detector").pseudo_auc.mean()

rows=[]; nlp_ens=[]
for mod in ["tabular","cv","nlp","ts","dami"]:
    tmod=TRUE[TRUE.modality==mod]; pmod=PS[PS.modality==mod]
    for ds in tmod.dataset.unique():
        radr=[]; rens=[]; per_seed_true=[]
        for s in sorted(tmod.seed.unique()):
            tr=tmod[(tmod.dataset==ds)&(tmod.seed==s)].set_index("detector").true_auc.dropna()
            if len(tr)<3: continue
            bank=pmod[(pmod.dataset==ds)&(pmod.seed==s)]
            if bank.empty: continue
            sc=agg_disc(bank); sc=sc[sc.index.isin(tr.index)]
            if sc.empty: continue
            pick=sc.sort_values(ascending=False).index[0]
            radr.append(tr.max()-tr[pick])
            # ensemble-default pick
            e=ens_smallest_random(bank); e=e[e.index.isin(tr.index)]
            if not e.empty:
                epick=e.sort_values(ascending=False).index[0]; rens.append(tr.max()-tr[epick])
            per_seed_true.append(tr)
        if not radr: continue
        # detector-wise mean true auc over seeds for fixed-baseline regrets
        tt=pd.concat(per_seed_true,axis=1).mean(axis=1)
        best=tt.max()
        rows.append(dict(modality=mod,dataset=ds,
                         adrank=float(np.mean(radr)),
                         iforest=float(best-tt.get("IForest",np.nan)),
                         random=float(best-tt.mean()),
                         true_by_det=tt.to_dict()))
        if mod=="nlp" and rens: nlp_ens.append(np.mean(rens))
g=pd.DataFrame(rows)

# best oracle-single detector PER MODALITY (commit to one detector for the whole modality)
print(f"{'modality':8} {'nds':>4} {'ADRank':>8} {'IForest':>8} {'best-1det':>9} {'random':>8}  {'ADRvsIFO':>9} {'ADRvsBest1':>11}")
for mod,gm in g.groupby("modality"):
    detmean={}
    for det in ["CBLOF","COPOD","ECOD","HBOS","IForest","KNN","LODA","LOF","PCA"]:
        reg=[row["true_by_det"].get(det) for row in gm.to_dict("records")]
        reg=[ (row_best:=max(row["true_by_det"].values())) - row["true_by_det"].get(det,np.nan) for row in gm.to_dict("records")]
        detmean[det]=np.nanmean(reg)
    best1=min(detmean,key=detmean.get)
    best1reg=np.array([max(r["true_by_det"].values())-r["true_by_det"].get(best1,np.nan) for r in gm.to_dict("records")])
    a=gm.adrank.values; ifo=gm.iforest.values; rnd=gm.random.values
    def pv(x,y):
        try: return wilcoxon(x,y).pvalue
        except Exception: return np.nan
    print(f"{mod:8} {len(gm):4d} {a.mean():8.4f} {ifo.mean():8.4f} {best1reg.mean():9.4f} {rnd.mean():8.4f}  "
          f"p={pv(a,ifo):7.1e} p={pv(a,best1reg):9.1e}  (best1={best1})")

print()
if nlp_ens:
    print(f"CORRECTED NLP ensemble-default (smallest+random, K30) regret = {np.mean(nlp_ens):.4f} "
          f"(paper's inflated 'fixed' = 0.065; agent found 0.022)")
g.drop(columns=["true_by_det"]).to_parquet("results/vs_fixed_per_dataset.parquet")
print("saved results/vs_fixed_per_dataset.parquet")
