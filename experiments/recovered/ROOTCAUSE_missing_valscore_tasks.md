# Root cause: "9 missing tasks" in the val-scores Modal run (2026-09-07)

## Symptom
After the `modal_valscores.py` run, downloading the results volume and intersecting with the 151-task
keep set showed only **142/151** datasets, with 9 always missing (VoteOutcome, BatchProcessingEvents,
LandShapeAnomaly, GamingBehavior, SpotifyHitPredictor, CourseRegistration, seismic-bumps,
EmailSpamDetection, TreadmillStatus). The 9 clustered in the tails of shards 10 and 11, which looked
like a mid-shard worker crash.

## Investigation
1. Mapped the 9 to shard partitions (`keys[i::12]`): all in the **tails of shards 10 and 11** (positions
   >=6), consistent with a crash that lost each shard's tail.
2. Reproduced the two apparent crash datasets locally (`crash_diag.py`): all **25 classical fits pass**,
   and (background) all **7 deep fits pass** too. No detector crashes on this data.
3. Read the Modal client log: **every one of the 173 datasets printed "32 detectors, nval=..."**,
   including all 9 "missing" ones, ending with **"DONE: 173 datasets"**. The only traceback was a benign
   `StopAsyncIteration` from the synchronicity/`.map()` iterator teardown, not a data error.
4. Clean re-download of ALL shard files -> **151/151 present, 0 missing.**

## Root cause (NOT a compute failure)
No task ever failed; all 173 scored in the first run. The "missing 9" was a **download/sharding
artifact** with two compounding causes:

1. **Non-deterministic shard filenames.** `run_shard` named its output
   `shard_{abs(hash(tuple(keys))) % 10**8}.jsonl`. Python's builtin `hash()` is **per-process
   randomized** (PYTHONHASHSEED), so the original run and the (idempotent-intended) fill re-run wrote the
   SAME partitions to **different filenames** -> 22 shard files in the volume instead of 12, and the
   "resume" fill did not resume, it re-scored into new files.
2. **Download during an in-flight write.** The per-file `modal volume get` loop ran while the fill run
   was still committing its (differently-named) shards, so it captured a partial 142-of-151 snapshot.
   Modal volume commits are also subject to a short sync lag (see gpu2modal notes).

## Fix
- `modal_valscores.py`, `modal_probe1000.py`, `modal_deepdebias.py`: shard filename is now
  `shard_{hashlib.md5(",".join(sorted(keys)))[:10]}.jsonl` -> **deterministic across processes**, so a
  re-run resumes the same shard files instead of duplicating them.
- Operational lesson: after a `.map()` run reports "DONE: N", trust the app-log count over an immediate
  volume download; re-download once the app has fully completed, and verify the expected task count
  before treating any task as "failed".

## Takeaway for future "missing task" panics
Check the app log for a per-item success print and the terminal "DONE: N" BEFORE assuming a crash. A
`StopAsyncIteration` traceback at the end of a `modal run ... .map()` is normal client teardown, not an
error.
