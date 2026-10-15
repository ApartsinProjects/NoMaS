# -*- coding: utf-8 -*-
"""Build a CODE-ONLY curated Zenodo deposit for NoMaS into <repo>/../zenodo_build/.

No third-party corpora (DAMI/ADBench .npz), no model checkpoints, no heavy raw
parquet or diagnostic runs. Ships: source package, reproduction scripts, paper
source + built PDF, curated final-result CSVs/figures, license, requirements,
and external-data retrieval instructions.
"""
import hashlib, zipfile
from pathlib import Path

REPO = Path(r"E:\Projects\Submitted\ADRank")
OUT = REPO.parent / "zenodo_build"
ARC = "nomas"  # top folder inside the zip

# curated final results that back the paper's tables/figures (flat files under results/)
RESULTS_KEEP = [
    "summary.csv", "correlations.csv", "pivot_rho.csv",
    "oddbench.csv", "contamination.csv",
    "reconcile.csv", "reconcile_pulled.csv", "b2_scaleup_pulled.csv",
    "panel_robustness.csv", "regime_calibration.csv",
    "multimodal_summary.csv", "multimodal_per_dataset.csv", "multimodal_ci_summary.csv",
    "multiseed_per_dataset.csv", "multiseed_per_seed.csv", "multiseed_per_seed_agg.csv",
    "ci_regret_summary.csv", "ci_regret_per_seed.csv",
    "vade_full_dami.csv", "baseline_consensus_correlations.csv",
]
RESULTS_DIRS = ["plots"]  # figure PNGs used in the paper
CODE_DROP = {"__pycache__", ".idea", ".ipynb_checkpoints"}


def skip(p: Path) -> bool:
    parts = set(p.parts)
    return (p.name.startswith("~$") or p.suffix in {".pyc"}
            or bool(parts & CODE_DROP))


def add_tree(zf, base: Path, arc_sub: str):
    if not base.exists():
        print(f"  WARN missing: {base}")
        return 0
    n = 0
    for p in sorted(base.rglob("*")):
        if p.is_file() and not skip(p):
            zf.write(p, f"{ARC}/{arc_sub}/{p.relative_to(base)}")
            n += 1
    return n


def add_file(zf, src: Path, arc_name: str):
    if src.exists():
        zf.write(src, f"{ARC}/{arc_name}")
        return 1
    print(f"  WARN missing: {src}")
    return 0


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    zpath = OUT / "nomas_code.zip"
    counts = {}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, 6) as zf:
        counts["src"] = add_tree(zf, REPO / "src", "src")
        counts["scripts"] = add_tree(zf, REPO / "scripts", "scripts")
        # paper source (md + build scripts + latex), excluding built docs/
        counts["paper"] = add_tree(zf, REPO / "paper", "paper")
        # the authoritative built paper PDF
        counts["paper_pdf"] = add_file(zf, REPO / "docs" / "nomas-2col-latex.pdf", "paper.pdf")
        # curated final results
        rc = 0
        for f in RESULTS_KEEP:
            rc += add_file(zf, REPO / "results" / f, f"results/{f}")
        for d in RESULTS_DIRS:
            rc += add_tree(zf, REPO / "results" / d, f"results/{d}")
        counts["results"] = rc
        # repo readme + generated deposit docs
        counts["readme"] = add_file(zf, REPO / "README.md", "README.md")
        for name in ["DEPOSIT_README.md", "EXTERNAL_DATA.md", "LICENSE", "requirements.txt"]:
            counts[name] = add_file(zf, OUT / name, name)
    # manifest + checksums
    size_mb = zpath.stat().st_size / 1e6
    (OUT / "SHA256SUMS.txt").write_text(f"{sha256(zpath)}  {zpath.name}\n")
    man = [f"{zpath.name:24s} {size_mb:8.2f} MB", ""]
    man += [f"  {k:16s} {v} files" for k, v in counts.items()]
    (OUT / "MANIFEST.txt").write_text(
        "NoMaS Zenodo deposit (code-only) - file manifest\n" + "=" * 52 + "\n"
        + "\n".join(man) + "\n")
    print("\n".join(man))
    print(f"\nwrote {zpath}")


if __name__ == "__main__":
    main()
