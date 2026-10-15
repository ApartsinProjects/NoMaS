# -*- coding: utf-8 -*-
import io, re, fitz


def words(s):
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"&[a-z]+;", " ", s)
    return [w for w in re.findall(r"[a-zA-Z]{3,}", s.lower())]


md = io.open(r"E:\Projects\Submitted\ADRank\paper\nomas.md", encoding="utf-8").read()
m = [x for x in re.finditer(
    r"<b>((?:Table|Figure) [A-Z0-9.]+)\.</b>\s*(.*?)</(?:div|figcaption)>", md, re.S)
     if x.group(1) == "Table 4"][0]
w = words(m.group(2))
doc = fitz.open(r"E:\Projects\Submitted\ADRank\docs\nomas-2col-latex.pdf")
pdf = " ".join(words(" ".join(p.get_text() for p in doc)))

lo, hi = 0, len(w)
while lo < hi:
    mid = (lo + hi + 1) // 2
    if " ".join(w[:mid]) in pdf:
        lo = mid
    else:
        hi = mid - 1
print(f"longest matching prefix: {lo} of {len(w)} words")
print("context before break :", " ".join(w[max(0, lo - 8):lo]))
print("source continues     :", " ".join(w[lo:lo + 10]))
# what does the PDF have right after that prefix?
idx = pdf.find(" ".join(w[max(0, lo - 8):lo]))
if idx >= 0:
    print("pdf continues        :", pdf[idx:idx + 260])
