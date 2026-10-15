# -*- coding: utf-8 -*-
"""Caption audit, math-insensitive: compare only alphabetic word runs so that
$\rho$ vs rho, entities and ligatures cannot cause false mismatches."""
import io, re, sys
import fitz

PDF = r"E:\Projects\Submitted\ADRank\docs\nomas-2col-latex.pdf"
SRC = r"E:\Projects\Submitted\ADRank\paper\nomas.md"


def words(s):
    """Lowercase alphabetic words of length>=3, math/markup stripped."""
    s = re.sub(r"\$[^$]*\$", " ", s)        # inline math
    s = re.sub(r"<[^>]+>", " ", s)          # tags
    s = re.sub(r"&[a-z]+;", " ", s)         # entities
    return [w for w in re.findall(r"[a-zA-Z]{3,}", s.lower())]


md = io.open(SRC, encoding="utf-8").read()
caps = []
for m in re.finditer(r"<b>((?:Table|Figure) [A-Z0-9.]+)\.</b>\s*(.*?)</(?:div|figcaption)>", md, re.S):
    caps.append((m.group(1), m.group(2)))

doc = fitz.open(PDF)
pdf_words = words(" ".join(p.get_text() for p in doc))
pdf_join = " ".join(pdf_words)

bad = 0
for label, body in caps:
    w = words(body)
    head, tail = " ".join(w[:8]), " ".join(w[-8:])
    ok_h, ok_t = head in pdf_join, tail in pdf_join
    # coverage: fraction of caption words found in a contiguous run
    full = " ".join(w)
    ok_full = full in pdf_join
    status = "OK " if (ok_h and ok_t and ok_full) else "BAD"
    if status == "BAD":
        bad += 1
    print(f"{status} {label:12s} words={len(w):3d} head={'Y' if ok_h else 'N'} "
          f"tail={'Y' if ok_t else 'N'} verbatim={'Y' if ok_full else 'N'}")
    if status == "BAD":
        if not ok_h:
            print(f"      head not found: {head!r}")
        if not ok_t:
            print(f"      tail not found: {tail!r}")

print(f"\ncaptions failing: {bad}/{len(caps)}")
sys.exit(1 if bad else 0)
