# -*- coding: utf-8 -*-
"""Move Ablations / Panel-robustness / Per-dataset-scatter / Deep-detectors to an
appendix (A/B), renumber all floats and sections collision-safely, in adrank.md."""
import io, re

P = r'E:\Projects\Submitted\ADRank\paper\adrank.md'
t = io.open(P, encoding='utf-8').read()

def cut(a, b):
    global t
    i = t.index(a); j = t.index(b)
    blk = t[i:j]; t = t[:i] + t[j:]
    return blk

# 1. Extract the four moving blocks (order in source; cut sequentially).
abl = cut("### 5.2 Ablations", "### 5.3 Panel robustness")
panel = cut("### 5.3 Panel robustness", "### 5.4 Per-dataset scatter")
scatter = cut("### 5.4 Per-dataset scatter", "### 5.5 Cross-modality")
deep = cut("**Deep detectors.**", "**A structural ceiling, and candidate fixes.**")

# strip the original section-heading line from each section block (keep the body)
def strip_head(block):
    return block.split("\n", 1)[1].lstrip("\n")
abl_body = strip_head(abl); panel_body = strip_head(panel); scatter_body = strip_head(scatter)

# 2. Section renumber in the remaining main text (headings + Section/§ refs).
#    ascending order; single pass each; targets don't collide with pending sources.
for old, new in [("5.5", "5.2"), ("5.6", "5.3"), ("5.7", "5.4"), ("5.8", "5.5"), ("5.9", "5.6")]:
    t = t.replace(f"### {old} ", f"### {new} ")
    t = t.replace(f"Section {old}", f"Section {new}").replace(f"\u00a7{old}", f"\u00a7{new}")
# the 5.4 heading (was 5.7) still mentions "deep detectors"; drop that word
t = t.replace("### 5.4 A second benchmark, deep detectors, and a structural ceiling",
              "### 5.4 A second benchmark and a structural ceiling")

# 3. Float renumber, MAIN text, collision-safe (old -> TOK -> new).
main_map = [("Table 5b", "Table 4"), ("Table 4", "Table 2"), ("Table 5", "Table 3"),
            ("Table 7", "Table 5"), ("Table 8", "Table 6"), ("Table 9", "Table 7"),
            ("Figure 4", "Figure 3"), ("Figure 5", "Figure 4")]
# pass A: old -> token (longest label first so "Table 5b" beats "Table 5")
for k, (old, _) in enumerate(sorted(main_map, key=lambda p: -len(p[0]))):
    t = re.sub(re.escape(old) + r"(?![0-9A-Za-z])", f"\x00M{k}\x00", t)
# map token index back to new (index by the sorted order used above)
order = sorted(main_map, key=lambda p: -len(p[0]))
for k, (_, new) in enumerate(order):
    t = t.replace(f"\x00M{k}\x00", new)

# 4. Appendix float renumber (each block isolated -> no collision).
abl_body = re.sub(r"Table 2(?![0-9A-Za-z])", "Table A.1", abl_body)
scatter_body = re.sub(r"Figure 3(?![0-9A-Za-z])", "Figure A.1", scatter_body)
panel_body = re.sub(r"Table 3(?![0-9A-Za-z])", "Table B.1", panel_body)
deep = re.sub(r"Table 6(?![0-9A-Za-z])", "Table B.2", deep)

# 5. Build the appendix and insert before the References section.
appendix = (
    "\n## Appendix A. Ablations and per-dataset behavior\n\n"
    "### A.1 Cluster-selection strategy and rank aggregation\n\n" + abl_body.rstrip() + "\n\n"
    "### A.2 Per-dataset predicted-versus-true rank\n\n" + scatter_body.rstrip() + "\n\n"
    "## Appendix B. Robustness checks\n\n"
    "### B.1 Leave-one-detector-out panel robustness\n\n" + panel_body.rstrip() + "\n\n"
    "### B.2 Deep detectors\n\n" + deep.rstrip() + "\n\n")

# References heading: source uses "## References" (an <h2>) or a div; find it.
mref = re.search(r"\n(<h2>References</h2>|## References|<div class=\"references\">)", t)
assert mref, "References anchor not found"
t = t[:mref.start()] + "\n" + appendix + t[mref.start():]

io.open(P, "w", encoding="utf-8").write(t)
print("restructured. appendix chars:", len(appendix))
