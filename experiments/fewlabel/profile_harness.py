# Profile one task / one rep of the few-label harness to find the bottleneck.
import cProfile, pstats, sys, io, os, glob, importlib.util
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
key = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, "experiments", "pivot", "scores", "oddbench*.npz")))[0]
sys.argv = ["x", "--bundles", os.path.join(ROOT, "artifacts", "modal", "bv2"), "--scores", os.path.join(ROOT, "experiments", "pivot", "scores"),
            "--out", os.path.join(os.environ.get("TEMP", "."), "prof.jsonl"), "--phase", "core", "--reps", "1", "--workers", "1", "--keys", key]
spec = importlib.util.spec_from_file_location("h", os.path.join(ROOT, "experiments", "fewlabel", "run_fewlabel.py"))
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
pr = cProfile.Profile(); pr.enable(); h.main(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15); print(s.getvalue()[-3500:])
