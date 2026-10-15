# -*- coding: utf-8 -*-
"""Audit the rendered PDF: every float caption present exactly once, intact,
and on the same page as its float."""
import io, re, sys
import fitz

PDF = r"E:\Projects\Submitted\ADRank\docs\nomas-2col-latex.pdf"
SRC = r"E:\Projects\Submitted\ADRank\paper\nomas.md"

md = io.open(SRC, encoding="utf-8").read()
# source captions: label + full text
caps = []
for m in re.finditer(r"<b>((?:Table|Figure) [A-Z0-9.]+)\.</b>\s*(.*?)</(?:div|figcaption)>", md, re.S):
    label, body = m.group(1), m.group(2)
    body = re.sub(r"<[^>]+>", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    caps.append((label, body))

doc = fitz.open(PDF)
pages = [re.sub(r"\s+", " ", p.get_text()) for p in doc]
full = " ".join(pages)

print(f"source floats: {len(caps)}   pdf pages: {len(pages)}\n")
bad = 0
for label, body in caps:
    lab = label + "."
    n = full.count(lab)
    where = [i + 1 for i, p in enumerate(pages) if lab in p]
    # first 8 words of the caption body should follow the label on that page
    head = " ".join(body.split()[:8])
    intact = head in full
    # tail of the caption (last 6 words) also present -> not truncated
    tail = " ".join(body.split()[-6:])
    whole = tail in full
    flag = ""
    if n != 1:
        flag += f" !!occurrences={n}"
    if not intact:
        flag += " !!head-missing"
    if not whole:
        flag += " !!tail-missing"
    if flag:
        bad += 1
    print(f"{label:12s} page {str(where):10s} label_x{n} head={'Y' if intact else 'N'} "
          f"tail={'Y' if whole else 'N'}{flag}")

print(f"\nfloats with problems: {bad}/{len(caps)}")
sys.exit(1 if bad else 0)
