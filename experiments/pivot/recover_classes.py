# -*- coding: utf-8 -*-
"""Recover the original class of each anomaly in the ADBench CV/NLP embedding families (CIFAR10_c,
FashionMNIST_c, 20news_c, agnews_c). In family F, file F_c uses class c as normal and draws anomalies from
the other classes. If the same items were embedded throughout, an anomaly's vector appears among the
normals of the file for its true class; an exact (float32) match recovers the class -> semantic anomaly
types for leave-type-out evaluation."""
import numpy as np, glob, os, re, collections, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
for sub in ["cv", "nlp"]:
    fams = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(D, sub, "*.npz")):
        m = re.match(r"(.+)_(\d+)$", os.path.basename(f)[:-4])
        if m: fams[m.group(1)][int(m.group(2))] = f
    print(sub, {k: len(v) for k, v in fams.items()}, flush=True)
    for fam, files in sorted(fams.items()):
        if len(files) < 2: continue
        lut = {}
        for c, f in files.items():
            z = np.load(f); X = np.ascontiguousarray(z["X"].astype(np.float32)); y = z["y"]
            for v in X[y == 0]: lut[v.tobytes()] = c
        hit = tot = 0; ntypes = []
        for c, f in sorted(files.items()):
            z = np.load(f); X = np.ascontiguousarray(z["X"].astype(np.float32)); y = z["y"]
            got = collections.Counter()
            for v in X[y == 1]:
                k = lut.get(v.tobytes()); tot += 1
                if k is not None and k != c: hit += 1; got[k] += 1
            ntypes.append(len([t for t, n in got.items() if n >= 20]))
        print(f"  {sub}/{fam}: {len(files)} files | anomalies {tot} | class recovered {hit} ({hit/max(tot,1):.0%}) |"
              f" types with >=20 anomalies per file: {ntypes}", flush=True)
