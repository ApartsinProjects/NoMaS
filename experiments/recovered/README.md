# Recovered experiment code (2026-10-15)

The final experiment scripts for the ADReal paper lived only in a session scratchpad, which was wiped
in mid-September 2026; they had never been committed. They were rebuilt from the Claude Code session
transcript by replaying every recorded Write, Edit, and full-file Read in order
(`_recover_from_transcript.py`).

- `RECOVERY_REPORT.txt` lists every file with its replay history
  (`W` write, `E` edit, `R` full read; `R=` read matched the replayed state, `R~` a later read differed
  and was taken as ground truth). 276 of 282 files replayed cleanly.
- Six files are marked `SUSPECT` (an edit could not be replayed): `hadb_ts_final.py`, `run_tsbad.py`,
  `streamline/eda_spectrum.py`, `streamline/fig_final_leaderboard.py`, `streamline/fig_leaderboard2.py`,
  `streamline/fig_pipeline.py`. A committed reference copy of `hadb_ts_final.py` is in `hadb/`.
- Paths inside the scripts still point at the old scratchpad
  (`E:/tmp/claude/.../scratchpad`); they need re-pointing before they will run.

Key scripts behind the paper: `streamline/bundle_v2.py` (task construction), `streamline/finalize_v3.py`
(leaderboard), `rescore_classical_valuni.py`, `streamline/modal_probe1000.py`, `streamline/modal_valscores.py`
(scoring harnesses), `uoms_recompute.py`, `udr_recompute.py`, `tier2_stats.py`, `tier3_recompute.py`
(selector statistics), `make_fig6_threepool.py` (Figure 6).

## Data that survived

Downloaded from Modal into `artifacts/modal/`: the 174 task bundles (`bv2/`, not in git), per-task test
scores for all 32 detectors (`probe1000/`, `valscores/`), and validation scores (`valscores/`, not in git).

## Known issues

- The paper's final leaderboard was built from an earlier scoring run (`deepres_v2`, `deepres_v2_clstd`)
  that did not survive. Rebuilding from the surviving runs (`artifacts/verify_paper_numbers.py`) gives
  150 tasks instead of 151: two OddBench tasks have an oracle score just under the 0.05 cut (0.0499,
  0.0494), and one of them must have been just over it in the lost run. Fixed-detector regret comes out
  0.128 (paper 0.130). The two surviving runs agree with each other exactly except for LODA.
- LODA is constructed without a seed in the scoring harnesses, so its scores change between runs
  (10 tasks differ). It needs a fixed seed before the pipeline is released.
