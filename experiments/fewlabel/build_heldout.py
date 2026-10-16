# -*- coding: utf-8 -*-
"""Build the HELD-OUT confirmation pool for the pre-registered few-label study
(experiments/PREREGISTRATION_fewlabel.md). Tabular only (OvrBench, OddBench).

Candidates = raw source datasets NEVER screened during development (absent from
experiments/streamline/STREAM_FINAL2_ALL.csv) and sharing no original source family with any screened
development dataset. Family = OddBench github_link (repo path) / tablib_origin batch; for OvrBench, which
has no provenance, a normalized name stem (lowercase, strip BNG(...)/version suffixes and digits).

Screening = the frozen detector-free 5-stage pipeline (functions copied verbatim from the recovered
pipeline_final.py): stage 1 drop if the two-stage hardening rule flags >= 90% of anomalies; stage 5 keep
>= 800 held-out normals; stage 3 keep >= 100 DISTINCT hard anomalies; stage 4 fingerprint dedup (also
against development fingerprints). Bundling = the recovered bundle_v2.py rules: drop anomalies identical to
a normal, dedup anomalies, grouped leak-free 60/20/20 split, harden, base-rate cap 0.5, then the
detector-free floor of >= 100 test anomalies. NO detector-dependent floors.
Output: artifacts/heldout/bv2/<key>.npz (+ _index.csv with family ids), artifacts/heldout/SCREEN.csv.
"""
import os, re, glob, hashlib, warnings, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "artifacts", "heldout"); OUTB = os.path.join(OUT, "bv2"); os.makedirs(OUTB, exist_ok=True)
Q = 0.05; TRIV1 = 0.90; MIN_HARD = 100; MIN_NORM = 800; BASE_CAP = 0.5; MIN_TEST_ANOM = 100

# ---------------- frozen functions (verbatim from recovered pipeline_final.py / bundle_v2.py) -------------
def severity(Xtr, X, bins=30):
    n, d = X.shape; sev = np.zeros(n)
    for j in range(d):
        tr = np.sort(Xtr[:, j])
        if tr[-1] - tr[0] < 1e-12: continue
        m = len(tr); fb = np.searchsorted(tr, X[:, j], side="right") / m
        tail = -np.log(np.maximum(2 * np.minimum(fb, 1 - fb), 1.0 / (2 * m)))
        cnt, edges = np.histogram(Xtr[:, j], bins=bins); dens = cnt / max(cnt.sum(), 1)
        b = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, bins - 1); sev = np.maximum(sev, np.maximum(tail, -np.log(dens[b] + 1e-9)))
    return sev

def two_stage_trivial(Xtr, Xhold, Xq):
    so_h, so_q = severity(Xtr, Xhold), severity(Xtr, Xq); tho = np.quantile(so_h, 1 - Q)
    try:
        sc = StandardScaler().fit(Xtr); pca = PCA(n_components=0.95, whiten=True, random_state=0).fit(sc.transform(Xtr))
        Ph, Pq, Ptr = pca.transform(sc.transform(Xhold)), pca.transform(sc.transform(Xq)), pca.transform(sc.transform(Xtr))
        sp_h, sp_q = severity(Ptr, Ph), severity(Ptr, Pq); thp = np.quantile(sp_h, 1 - Q)
    except Exception:
        sp_q = np.zeros(len(Xq)); thp = np.inf
    return (so_q > tho) | (sp_q > thp)

def n_eff(Xtr, Xh):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Zt = (Xtr - mu) / sd; Zh = (Xh - mu) / sd
    rng = np.random.default_rng(0); sub = Zt[rng.choice(len(Zt), min(len(Zt), 1500), replace=False)]
    r = np.median(NearestNeighbors(n_neighbors=2).fit(sub).kneighbors(sub)[0][:, 1]) + 1e-9
    if len(Zh) > 600: Zh = Zh[rng.choice(len(Zh), 600, replace=False)]
    nn = NearestNeighbors(radius=r).fit(Zh); cov = np.zeros(len(Zh), bool); c = 0
    for i in range(len(Zh)):
        if cov[i]: continue
        c += 1; cov[nn.radius_neighbors(Zh[i:i + 1], return_distance=False)[0]] = True
    return c

def fingerprint(Xtr, Xh_norm):
    Z = np.vstack([Xtr, Xh_norm]); return "%d|%s|%s" % (Z.shape[1], hashlib.sha1(np.round(np.sort(Z.mean(0)), 3).tobytes()).hexdigest()[:10], hashlib.sha1(np.round(np.sort(Z.std(0)), 3).tobytes()).hexdigest()[:10])

def _tab_split(Xn, Xa):
    r = np.random.default_rng(0)
    if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn)); r.shuffle(idx); k = int(0.6 * len(idx)); return Xn[idx[:k]], Xn[idx[k:]], Xa

def key32(A): return np.round(A.astype(np.float32).astype(np.float64), 6)

def grouped_split(Xn, seed=0):
    r = np.random.default_rng(seed)
    uniq, inv = np.unique(key32(Xn), axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)); order = np.arange(len(uniq)); r.shuffle(order)
    cum = np.cumsum(counts[order]); tot = cum[-1]
    trg = set(order[cum <= 0.6 * tot].tolist()); vag = set(order[(cum > 0.6 * tot) & (cum <= 0.8 * tot)].tolist())
    assign = np.where(np.isin(inv, list(trg)), 0, np.where(np.isin(inv, list(vag)), 1, 2))
    return Xn[assign == 0], Xn[assign == 1], Xn[assign == 2]

def harden(Xtr, Xhold, Xa):
    return Xa[~two_stage_trivial(Xtr, Xhold, Xa)]
# ------------------------------------------------------------------------------------------------------------

def load_raw(corp, name):
    d = np.load(os.path.join(ROOT, "data", corp, name + ".npz"), allow_pickle=True)
    X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]))
    y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    return X, y, str(d["github_link"]), str(d["tablib_origin"])

def family(corp, name, link, tablib):
    if link and "github.com" in link:
        m = re.search(r"github\.com/([^/]+/[^/]+)", link); return "gh:" + (m.group(1).lower() if m else link)
    if tablib: return "tl:" + tablib.split("_part")[0].lower()
    s = name.lower(); s = re.sub(r"^bng\((.*)\)$", r"\1", s); s = re.sub(r"[\(\)_\-\.\d]+", "", s)
    return "nm:" + s

def screen_and_bundle(arg):
    corp, name = arg
    try:
        X, y, link, tablib = load_raw(corp, name)
        Xn, Xa = X[y == 0], X[y == 1]
        Xtr0, Xhold, Xa0 = _tab_split(Xn, Xa)
        if len(Xtr0) < 40 or len(Xhold) < 20 or len(Xa0) < 5: return dict(corpus=corp, dataset=name, status="too_small")
        triv = two_stage_trivial(Xtr0, Xhold, Xa0); hard = Xa0[~triv]
        ne = n_eff(Xtr0, hard) if len(hard) >= 5 else len(hard)
        row = dict(corpus=corp, dataset=name, family=family(corp, name, link, tablib), n_norm=len(Xhold), n_anom=len(Xa0),
                   n_hard=int((~triv).sum()), frac_triv=float(triv.mean()), n_eff=int(ne), fp=fingerprint(Xtr0, Xhold))
        if not (row["frac_triv"] < TRIV1 and row["n_norm"] >= MIN_NORM and row["n_eff"] >= MIN_HARD):
            row["status"] = "screened_out"; return row
        # ---- bundle (bundle_v2 rules) ----
        r = np.random.default_rng(0)
        if len(Xn) > 6000: Xn = Xn[r.choice(len(Xn), 6000, replace=False)]
        nset = set(map(tuple, key32(Xn)))
        Xa = Xa[np.array([tuple(x) not in nset for x in key32(Xa)], dtype=bool)] if len(Xa) else Xa
        if len(Xa): _, ia = np.unique(key32(Xa), axis=0, return_index=True); Xa = Xa[np.sort(ia)]
        Xtr, Xval, Xtn = grouped_split(Xn, seed=0)
        Xh = harden(Xtr, np.vstack([Xval, Xtn]), Xa)
        cap_n = int(len(Xtn) * BASE_CAP / (1 - BASE_CAP))
        if len(Xh) > cap_n: Xh = Xh[np.random.default_rng(1).choice(len(Xh), cap_n, replace=False)]
        if len(Xh) < MIN_TEST_ANOM:
            row["status"] = "few_test_anomalies"; row["n_h"] = len(Xh); return row
        key = f"{corp}__{str(name)[:36]}".replace("/", "_").replace(" ", "_")
        np.savez_compressed(os.path.join(OUTB, key + ".npz"), Xtr=Xtr.astype(np.float32), Xval=Xval.astype(np.float32),
                            Xtn=Xtn.astype(np.float32), Xh=Xh.astype(np.float32))
        row.update(status="bundled", key=key, d=Xtr.shape[1], n_tr=len(Xtr), n_val=len(Xval), n_tn=len(Xtn), n_h=len(Xh),
                   base_rate=round(len(Xh) / (len(Xtn) + len(Xh)), 3))
        return row
    except Exception as e:
        return dict(corpus=corp, dataset=name, status=f"error:{str(e)[:60]}")


def main():
    from multiprocessing import Pool
    dev = pd.read_csv(os.path.join(ROOT, "experiments", "streamline", "STREAM_FINAL2_ALL.csv"))
    dev_names = set(zip(dev.corpus, dev.dataset.astype(str).str[:40]))
    dev_fp = set(dev.fp.astype(str))
    # development families
    dev_fams = set()
    for corp, name in dev_names:
        if corp not in ("ovrbench", "oddbench"): continue
        p = os.path.join(ROOT, "data", corp, name + ".npz")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True); dev_fams.add(family(corp, name, str(d["github_link"]), str(d["tablib_origin"])))
    cands, fam_excl = [], 0
    for corp in ("ovrbench", "oddbench"):
        for f in sorted(glob.glob(os.path.join(ROOT, "data", corp, "*.npz"))):
            name = os.path.basename(f)[:-4]
            if (corp, name[:40]) in dev_names: continue
            d = np.load(f, allow_pickle=True)
            if family(corp, name, str(d["github_link"]), str(d["tablib_origin"])) in dev_fams: fam_excl += 1; continue
            cands.append((corp, name))
    print(f"uninspected candidates: {len(cands)} (excluded {fam_excl} sharing a development family)", flush=True)
    rows = []
    with Pool(int(os.environ.get("NW", "3"))) as p:
        for i, r in enumerate(p.imap_unordered(screen_and_bundle, cands, chunksize=1)):
            rows.append(r)
            if (i + 1) % 50 == 0: print(f"  {i+1}/{len(cands)}  bundled so far {sum(x.get('status')=='bundled' for x in rows)}", flush=True)
    df = pd.DataFrame(rows)
    # stage 4: fingerprint dedup within held-out and against development
    b = df[df.status == "bundled"].copy()
    clash = b.fp.isin(dev_fp); b = b[~clash].sort_values("n_eff", ascending=False).drop_duplicates("fp")
    drop = set(df[(df.status == "bundled")].key) - set(b.key)
    for k in drop:
        try: os.remove(os.path.join(OUTB, k + ".npz"))
        except OSError: pass
    df.loc[df.key.isin(drop), "status"] = "dedup"
    df.to_csv(os.path.join(OUT, "SCREEN.csv"), index=False)
    b.to_csv(os.path.join(OUTB, "_index.csv"), index=False)
    print("\nstatus counts:", df.status.value_counts().to_dict())
    print(f"HELD-OUT POOL: {len(b)} tasks from {b.family.nunique()} families | by corpus {b.corpus.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
