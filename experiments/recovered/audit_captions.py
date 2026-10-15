# -*- coding: utf-8 -*-
"""Audit float/caption structure in the generated main.tex."""
import io, re

P = r"E:\Projects\Submitted\ADRank\paper\latex\main.tex"
lines = io.open(P, encoding="utf-8").read().split("\n")

B = "\\"
pat = [
    (re.compile(re.escape(B) + r"begin\{table\*?\}"), "TAB-BEGIN"),
    (re.compile(re.escape(B) + r"end\{table\*?\}"), "TAB-END"),
    (re.compile(re.escape(B) + r"begin\{figure\*?\}"), "FIG-BEGIN"),
    (re.compile(re.escape(B) + r"end\{figure\*?\}"), "FIG-END"),
    (re.compile(re.escape(B) + r"caption"), "CAPTION"),
    (re.compile(re.escape(B) + r"textbf\{(Table|Figure) "), "BOLDLABEL"),
    (re.compile(re.escape(B) + r"begin\{longtable\}"), "LONGTAB-BEGIN"),
    (re.compile(re.escape(B) + r"end\{longtable\}"), "LONGTAB-END"),
]

for i, l in enumerate(lines, 1):
    for rx, tag in pat:
        if rx.search(l):
            print(f"{i:5d} {tag:14s} {l.strip()[:72]}")
            break
