"""B1-B4 analyses on the 151-task per-dataset regret table.
B1: direct paired SPARC-vs-Goswami test (+ effect size, CI)
B2: bootstrap CIs on mean regret per selector (+ paired-diff plot)
B3: median + IQR + tails per selector
B4: source-clustered inference (cluster bootstrap by parent dataset)
Nothing here is folded into the paper; results are printed and a figure saved.
"""
import numpy as np, pandas as pd, re, json
from scipy import stats
from pathlib import Path

RES = Path(r"E:/Projects/Submitted/zenodo_build/results/model_selection_per_dataset.csv")
OUT = Path(r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad")
df = pd.read_csv(RES)
n = len(df)
print(f"tasks: {n}")

SEL = {
    "SPARC": "regret_SPARC",
    "Goswami": "regret_Goswami",
    "Best fixed (LOF k=10)": "regret_best_fixed_detector",
    "Entropy": "regret_entropy",
    "Mass-volume": "regret_massvolume",
    "Consensus": "regret_consensus",
    "Model-centrality": "regret_modelcentrality",
    "HITS": "regret_HITS",
    "Random": "regret_random",
}
for k, c in SEL.items():
    assert c in df.columns, c

rng = np.random.default_rng(20260906)

def cohen_dz(diff):
    # paired effect size
    return diff.mean() / diff.std(ddof=1)

def boot_ci_mean(x, B=10000, alpha=0.05):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return x.mean(), lo, hi

def cluster_boot_ci_mean(x, groups, B=10000, alpha=0.05):
    # resample whole parent-groups with replacement, then all rows in chosen groups
    x = np.asarray(x, float)
    uniq = pd.unique(groups)
    gidx = {g: np.where(groups.values == g)[0] for g in uniq}
    means = np.empty(B)
    for b in range(B):
        chosen = uniq[rng.integers(0, len(uniq), size=len(uniq))]
        rows = np.concatenate([gidx[g] for g in chosen])
        means[b] = x[rows].mean()
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return x.mean(), lo, hi

def cluster_perm_paired(a, b, groups, B=20000):
    # paired diff d = a-b; sign-flip whole groups (cluster-level sign permutation)
    d = (a - b).values
    uniq = pd.unique(groups)
    gidx = {g: np.where(groups.values == g)[0] for g in uniq}
    obs = d.mean()
    cnt = 0
    for _ in range(B):
        signs = rng.choice([-1.0, 1.0], size=len(uniq))
        dd = d.copy()
        for s, g in zip(signs, uniq):
            dd[gidx[g]] *= s
        if abs(dd.mean()) >= abs(obs) - 1e-15:
            cnt += 1
    return obs, (cnt + 1) / (B + 1)

# parent grouping: strip the trailing task specifier. keys look like
# "ovrbench__<parent>__<class>", "oddbench__<parent>", "ts__<parent>__win..", "adbench__<n>_<name>"
def parent(key):
    s = str(key)
    parts = s.split("__")
    if len(parts) >= 2:
        # corpus + parent (drop any 3rd+ chunk = class/window)
        return parts[0] + "__" + parts[1]
    return s

df["parent"] = df["dataset"].map(parent)
df["corpus_norm"] = df["corpus"]
n_parents = df["parent"].nunique()
print(f"parent groups: {n_parents}; corpora: {sorted(df['corpus'].unique())}")

# ---------------- B1: SPARC vs Goswami ----------------
print("\n=== B1: SPARC vs Goswami (paired) ===")
a = df["regret_SPARC"]; b = df["regret_Goswami"]
diff = b - a  # positive => SPARC lower regret (better)
w = stats.wilcoxon(a, b, zero_method="wilcoxon", alternative="two-sided")
w1 = stats.wilcoxon(a, b, alternative="less")  # SPARC < Goswami
nz = (a != b).sum()
print(f"mean SPARC={a.mean():.4f}  Goswami={b.mean():.4f}  mean diff(Gos-SPARC)={diff.mean():+.4f}")
print(f"median SPARC={a.median():.4f}  Goswami={b.median():.4f}  median diff={diff.median():+.4f}")
print(f"nonzero pairs: {nz}/{n}  wins SPARC(lower)={ (a<b).sum() }  wins Goswami={ (b<a).sum() }  ties={ (a==b).sum() }")
print(f"Wilcoxon two-sided p={w.pvalue:.4f}; one-sided(SPARC<Gos) p={w1.pvalue:.4f}")
print(f"Cohen dz(paired)={cohen_dz(diff):+.3f}")
m, lo, hi = boot_ci_mean(diff.values); print(f"mean diff 95% CI (row boot): {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
m, lo, hi = cluster_boot_ci_mean(diff.values, df["parent"]); print(f"mean diff 95% CI (parent-cluster boot): {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
obs, pcl = cluster_perm_paired(a, b, df["parent"]); print(f"cluster sign-perm p (by parent)={pcl:.4f}")

# ---------------- B3: median/IQR/tails ----------------
print("\n=== B3: distribution of regret per selector ===")
rowsB3 = []
for k, c in SEL.items():
    x = df[c]
    rowsB3.append(dict(selector=k, mean=x.mean(), median=x.median(),
                       q25=x.quantile(.25), q75=x.quantile(.75),
                       p90=x.quantile(.90), p95=x.quantile(.95), max=x.max()))
b3 = pd.DataFrame(rowsB3)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print(b3.to_string(index=False))
b3.to_csv(OUT/"B3_regret_distribution.csv", index=False)

# ---------------- B2: bootstrap CIs vs best-fixed ----------------
print("\n=== B2: mean regret + 95% bootstrap CI (row-level) ===")
fixed = df["regret_best_fixed_detector"]
rowsB2 = []
for k, c in SEL.items():
    m, lo, hi = boot_ci_mean(df[c].values)
    # paired diff vs fixed (selector - fixed): negative => selector better
    d = (df[c] - fixed)
    md, dlo, dhi = boot_ci_mean(d.values)
    pw = stats.wilcoxon(df[c], fixed).pvalue if (df[c]!=fixed).any() else np.nan
    rowsB2.append(dict(selector=k, mean=m, ci_lo=lo, ci_hi=hi,
                       diff_vs_fixed=md, dci_lo=dlo, dci_hi=dhi, wilcoxon_p_vs_fixed=pw))
b2 = pd.DataFrame(rowsB2)
print(b2.to_string(index=False))
b2.to_csv(OUT/"B2_mean_ci.csv", index=False)

# ---------------- B4: source-clustered inference ----------------
print("\n=== B4: source-clustered (parent) inference vs best-fixed ===")
rowsB4 = []
for k, c in SEL.items():
    if k.startswith("Best fixed"): continue
    d = df[c] - fixed  # selector - fixed
    md, lo, hi = cluster_boot_ci_mean(d.values, df["parent"])
    _, pcl = cluster_perm_paired(df[c], fixed, df["parent"])
    # source-balanced mean: mean over corpora of within-corpus mean regret
    sb = df.groupby("corpus")[c].mean().mean()
    rowsB4.append(dict(selector=k, mean_regret=df[c].mean(), source_balanced_mean=sb,
                       diff_vs_fixed=md, cl_ci_lo=lo, cl_ci_hi=hi, cluster_perm_p=pcl))
b4 = pd.DataFrame(rowsB4)
print(b4.to_string(index=False))
b4.to_csv(OUT/"B4_clustered.csv", index=False)
print(f"\n[note] task-level n={n}; parent-group n={n_parents}; corpus n={df['corpus'].nunique()}")

# ---------------- B2 figure: mean regret with bootstrap CI ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
order = ["SPARC","Goswami","Best fixed (LOF k=10)","Entropy","Mass-volume","Consensus","Model-centrality","HITS","Random"]
b2i = b2.set_index("selector").loc[order]
fig, ax = plt.subplots(figsize=(7.2,3.6))
y = np.arange(len(order))[::-1]
means = b2i["mean"].values
err = np.vstack([means-b2i["ci_lo"].values, b2i["ci_hi"].values-means])
colors = ["#1b6ca8" if s=="SPARC" else "#c44e52" if s.startswith("Best fixed") else "#888" for s in order]
ax.errorbar(means, y, xerr=err, fmt="o", capsize=3, color="#333", ecolor="#333", zorder=3, linestyle="none")
for yi, s in zip(y, order):
    ax.plot(b2i.loc[s,"mean"], yi, "o", color="#1b6ca8" if s=="SPARC" else "#c44e52" if s.startswith("Best fixed") else "#555", zorder=4)
ax.axvline(b2i.loc["Best fixed (LOF k=10)","mean"], color="#c44e52", ls="--", lw=1, alpha=.7, zorder=1)
ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlabel("mean regret (95% bootstrap CI)")
ax.set_title("Selector mean regret with bootstrap confidence intervals")
ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(OUT/"B2_mean_regret_ci.png", dpi=150)
print(f"\nsaved figure -> {OUT/'B2_mean_regret_ci.png'}")
