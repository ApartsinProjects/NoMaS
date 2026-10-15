"""C6 FULL ledger: all 1,192 source datasets -> fate. Merges PART A (exact construction+floor
per dataset) with PART B (upstream curation: raw source -> pipeline input).
Reconciles to Table 1 (1192 available / 174 candidate / 151 in-bench). Writes C6_FULL_LEDGER.csv."""
import os, glob, re, numpy as np, pandas as pd
ROOT = r"E:\Projects\Submitted\ADRank"
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d"
SS = os.path.join(S, "scratchpad")
MIN_NORM = 800

def norm_key(key):
    key = str(key); m = re.match(r'(tsbad_m__\d+_[A-Za-z0-9]+_id_\d+)', key)
    if m: return m.group(1)
    return re.sub(r'\.(csv|txt|npz)$', '', key)[:40]

partA = pd.read_csv(os.path.join(SS, "C6_LEDGER.csv"))       # 462 in-scope pipeline datasets, exact stage
partA_by_nkey = partA.set_index("nkey")

# manifest membership (datasets that ENTERED the pipeline for tabular ovr/odd)
man = {}
for corp in ("ovrbench", "oddbench"):
    m = pd.read_csv(os.path.join(S, f"hadb_{corp}.csv"))
    man[corp] = set(m.dataset.astype(str))

def size_pass(f):
    try:
        d = np.load(f, allow_pickle=True)
        if "X" in d:   # adbench/dami format
            X = np.nan_to_num(np.asarray(d["X"], float)); y = np.asarray(d["y"], int).ravel()
        else:          # ovr/odd train/test format
            X = np.nan_to_num(np.vstack([np.asarray(d["train"], float), np.asarray(d["test"], float)]))
            y = np.concatenate([np.asarray(d["train_labels"]).ravel(), np.asarray(d["test_labels"]).ravel()]).astype(int)
    except Exception:
        return None, None, None
    Xn, Xa = X[y == 0], X[y == 1]
    r = np.random.default_rng(0); Xn2 = Xn if len(Xn) <= 6000 else Xn[r.choice(len(Xn), 6000, replace=False)]
    idx = np.arange(len(Xn2)); r.shuffle(idx); k = int(0.6 * len(idx))
    return k, len(idx) - k, len(Xa)   # n_tr, n_hold(=n_norm), n_anom

rows = []
used_nkeys = set()   # guard: assign each partA row to exactly ONE raw file (avoid truncation-collision double-count)
# ---- tabular sources read from raw npz: ovr, odd, adbench, dami ----
NPZ = {"ovrbench": 754, "oddbench": 187, "adbench": 35, "dami": 16}
for corp in NPZ:
    for f in sorted(glob.glob(os.path.join(ROOT, "data", corp, "*.npz"))):
        name = os.path.basename(f)[:-4]
        nk = norm_key(f"{corp}__{name}")
        if nk in partA_by_nkey.index and nk not in used_nkeys:
            used_nkeys.add(nk)
            r = partA_by_nkey.loc[nk]
            rows.append(dict(corpus=corp, dataset=name, nkey=nk, stage=r["stage"],
                             in_candidate=bool(r["in_candidate"]), in_benchmark=bool(r["in_benchmark"])))
            continue
        if nk in partA_by_nkey.index:   # collided with an already-matched pipeline row (40-char key clash)
            rows.append(dict(corpus=corp, dataset=name, nkey=nk + f"~dup{name[:6]}",
                             stage="truncation_collision_40char", in_candidate=False, in_benchmark=False))
            continue
        # not in pipeline -> upstream drop. Reason:
        if corp in ("ovrbench", "oddbench"):
            if name in man[corp]:
                stage = "curation_in_manifest_but_load_mismatch"  # should be rare
            else:
                n_tr, n_norm, n_a = size_pass(f)
                if n_tr is None: stage = "raw_load_error"
                elif n_tr < 40 or n_norm < 20 or n_a < 5: stage = "raw_basic_gate(too_small)"
                elif n_norm < MIN_NORM: stage = f"curation_size(n_norm<{MIN_NORM})"
                else: stage = "curation_triviality/usability_prefilter"
        else:  # adbench/dami read raw by the pipeline
            n_tr, n_norm, n_a = size_pass(f)
            if n_tr is None: stage = "raw_load_error"
            elif n_tr < 40 or n_norm < 20 or n_a < 5: stage = "raw_basic_gate(too_small)"
            elif n_norm < MIN_NORM: stage = f"S5_too_few_normals(<{MIN_NORM})"
            else: stage = "raw_dropped_pre_metrics"
        rows.append(dict(corpus=corp, dataset=name, nkey=nk, stage=stage, in_candidate=False, in_benchmark=False))

# ---- tsbad_m: enumerate the REAL 200 source series (all csv entries in TSB-AD-M.zip; load_mts takes all).
import zipfile
zpath = os.path.join(ROOT, "data", "tsbad", "TSB-AD-M.zip")
ts_names = sorted(os.path.basename(n) for n in zipfile.ZipFile(zpath).namelist() if n.lower().endswith(".csv"))
partA_ts = {r["nkey"]: r for _, r in partA[partA.corpus == "tsbad_m"].iterrows()}
for name in ts_names:                       # name like 097_SVDB_id_14_Medical_tr_1031_1st_1131.csv
    nk = norm_key(f"tsbad_m__{name}")
    if nk in partA_ts and nk not in used_nkeys:
        used_nkeys.add(nk); r = partA_ts[nk]
        rows.append(dict(corpus="tsbad_m", dataset=name[:-4], nkey=nk, stage=r["stage"],
                         in_candidate=bool(r["in_candidate"]), in_benchmark=bool(r["in_benchmark"])))
    else:                                   # not loaded: too short for windowing / basic gate at load
        rows.append(dict(corpus="tsbad_m", dataset=name[:-4], nkey=nk if nk not in used_nkeys else nk + f"~{name[:6]}",
                         stage="raw_too_short_or_basic_gate", in_candidate=False, in_benchmark=False))

df = pd.DataFrame(rows)
OUT = os.path.join(SS, "C6_FULL_LEDGER.csv"); df.to_csv(OUT, index=False)
print(f"=== C6 FULL LEDGER: {len(df)} rows (Table1 available=1192) ===")
print(f"candidates: {df.in_candidate.sum()} (174)   in-benchmark: {df.in_benchmark.sum()} (151)")
print("\n--- per-corpus available / candidate / in-bench ---")
g = df.groupby("corpus").agg(available=("dataset","size"), candidate=("in_candidate","sum"), in_bench=("in_benchmark","sum"))
print(g.to_string()); print("TOTAL:", dict(available=len(df), candidate=int(df.in_candidate.sum()), in_bench=int(df.in_benchmark.sum())))
print("\n--- exclusion reasons (full funnel) ---")
print(df.stage.value_counts().to_string())
