"""C6 curation ledger. Two parts:
  PART A (exact, per-dataset): every dataset the streamline construction pipeline processed, its fate
     through the detector-free gates (S0 point-TS, S1 OR-trivial, S5 too-few-normals, S3 too-few-hard,
     S4 duplicate) and the three post-scoring floors (separability, min-test-anomalies, selection-trivial).
  PART B (source-level): the upstream curation (raw source -> pipeline input) with the size component
     reason-coded and the triviality pre-filter remainder counted.
Reconciles to Table 1 (available 1192 / candidate 174 / in-bench 151). Writes C6_LEDGER.csv + summary.
"""
import os, glob, re, numpy as np, pandas as pd

def norm_key(key):
    """Stable join id across construction (full name+ext) and scoring (bundle-truncated) stages.
    TSB-AD long filenames get two different [:40] truncations; reduce both to 'tsbad_m__NNN_SRC_id_M'."""
    key = str(key)
    m = re.match(r'(tsbad_m__\d+_[A-Za-z0-9]+_id_\d+)', key)
    if m: return m.group(1)
    return re.sub(r'\.(csv|txt|npz)$', '', key)[:40]   # cap at bundle truncation boundary
S = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/streamline"
ROOT = r"E:\Projects\Submitted\ADRank"
ZEN = r"E:/Projects/Submitted/zenodo_build/results"
TRIV1, MIN_HARD, MIN_NORM = 0.90, 100, 800

allrows = pd.read_csv(os.path.join(S, "STREAM_FINAL2_ALL.csv"))     # loaded candidates w/ metrics
cand = pd.read_csv(os.path.join(S, "STREAM_FINAL2_SET.csv"))         # 174 survived construction
lb = pd.read_csv(os.path.join(S, "STREAM_V3_LEADERBOARD.csv"))       # 174 scored: oracle_ap, n_h, reg_random
final = pd.read_csv(os.path.join(ZEN, "model_selection_per_dataset.csv"))  # 151 in-bench
INSCOPE = {"ovrbench", "oddbench", "tsbad_m", "adbench", "dami"}
allrows = allrows[allrows.corpus.isin(INSCOPE)].copy()

cand_keys = set((cand.corpus + "__" + cand.dataset).map(norm_key))
final_keys = set(final.dataset.map(norm_key))
lb["nkey"] = lb.key.map(norm_key)
lb_idx = lb.drop_duplicates("nkey").set_index("nkey")

def fate_construction(r):
    """first-failing construction gate in pipeline order (S0->S1->S5->S3->S4)."""
    if bool(r.point_anom): return "S0_point_or_short_TS"
    if r.frac_triv >= TRIV1: return f"S1_OR_trivial(>={int(TRIV1*100)}%)"
    if r.n_norm < MIN_NORM: return f"S5_too_few_normals(<{MIN_NORM})"
    if r.n_eff < MIN_HARD: return f"S3_too_few_hard_anomalies(<{MIN_HARD})"
    return "passed_construction"   # S4 dedup handled separately (needs fp uniqueness)

# S4 dedup: among passed rows, keep max n_eff per fingerprint; others are duplicates
def build():
    rows = []
    passed = []
    for _, r in allrows.iterrows():
        key = f"{r.corpus}__{r.dataset}"; nk = norm_key(key)
        f = fate_construction(r)
        rows.append(dict(corpus=r.corpus, dataset=r.dataset, key=key, nkey=nk,
                         n_norm=int(r.n_norm), n_anom=int(r.n_anom), n_hard=int(r.n_hard),
                         frac_triv=round(float(r.frac_triv), 3), n_eff=int(r.n_eff),
                         point_anom=bool(r.point_anom), fp=r.fp, _fate=f))
        if f == "passed_construction": passed.append((nk, r.fp, int(r.n_eff)))
    # S4 dedup among passed
    dfp = pd.DataFrame(passed, columns=["nkey", "fp", "n_eff"]).sort_values("n_eff", ascending=False)
    keep = set(dfp.drop_duplicates("fp").nkey)
    df = pd.DataFrame(rows)
    # assign candidate/dup
    def stage(row):
        if row._fate != "passed_construction":
            return row._fate
        if row.nkey not in keep:
            return "S4_duplicate_fingerprint"
        # candidate: now apply floors from leaderboard
        k = row.nkey
        if k not in lb_idx.index:
            return "candidate_unscored"        # 174->173 (1 fails to score)
        L = lb_idx.loc[k]
        if float(L.oracle_ap) <= 0.05: return "floor_separability(oracle_ap<=0.05)"
        if int(L.n_h) < 100:           return "floor_min_test_anomalies(<100)"
        if float(L.reg_random) <= 0.02: return "floor_selection_trivial(reg_random<=0.02)"
        return "IN_BENCHMARK"
    df["stage"] = df.apply(stage, axis=1)
    df["in_candidate"] = df.nkey.isin(cand_keys)
    df["in_benchmark"] = df.nkey.isin(final_keys)
    return df

df = build()
OUT = os.path.join(S, "..", "C6_LEDGER.csv"); df.drop(columns=["_fate"], errors="ignore").to_csv(OUT, index=False)
print(f"=== PART A: construction+floor ledger ({len(df)} in-scope loaded datasets) ===")
print(df.stage.value_counts().to_string())
print(f"\nreconcile: in_candidate={df.in_candidate.sum()} (Table1=174)  in_benchmark={df.in_benchmark.sum()} (Table1=151)")
print(f"IN_BENCHMARK stage rows={ (df.stage=='IN_BENCHMARK').sum() }")
# per-corpus candidate/in-bench
print("\nper-corpus candidate / in-benchmark:")
print(df.groupby('corpus')[['in_candidate','in_benchmark']].sum().to_string())
print(f"\nwrote {os.path.abspath(OUT)}")
