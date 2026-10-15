
# ----------------------------- DAMI fetch (ARFF) ----------------------------
from scipy.io import arff
DAMI_BASE = "https://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/"

def _pick_arff(names):
    cand = [n for n in names if n.endswith(".arff")]
    def sc(n):
        b = os.path.basename(n)
        pct = re.findall(r"_(\d{2})(?:_v\d+)?\.arff$", b)
        return ("withoutdupl" in b, "_norm" in b, re.search(r"_v\d+\.arff$", b) is None,
                -(int(pct[0]) if pct else 99))
    cand.sort(key=sc, reverse=True)
    return cand[0] if cand else None

def fetch_dami(name):
    raw = urllib.request.urlopen(DAMI_BASE + name + ".tar.gz", timeout=120).read()
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    an = _pick_arff(tf.getnames()); b = tf.extractfile(an).read()
    data, meta = arff.loadarff(io.StringIO(b.decode("utf-8", errors="replace")))
    names = list(meta.names())
    lab = next(c for c in names if c.lower() == "outlier")
    idc = next((c for c in names if c.lower() == "id"), None)
    feat = [c for c in names if c not in (lab, idc)]
    X = np.column_stack([np.asarray(data[c], dtype=np.float64) for c in feat])
    y = np.array([1 if str(v).lower().find("yes") >= 0 else 0 for v in data[lab]], dtype=int)
    return Dataset(name=name, X=X, y=y)

# ----------------------------- VaDE configs ---------------------------------
def _vade_custom(X, K, seed, alpha_mult=1.0, latent=None, pre=40, joint=60):
    import torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    torch.manual_seed(seed); np.random.seed(seed); dev = "cuda"
    Xs = StandardScaler().fit_transform(X).astype(np.float32); n, d = Xs.shape
    K = min(K, max(2, n // 20)); lat = latent or LATENT
    alpha = (1.0 / max(1, d // 16)) * alpha_mult
    Xt = torch.from_numpy(Xs).to(dev); model = VaDE(d, K, latent=lat).to(dev)
    opt = torch.optim.Adam(list(model.enc.parameters()) + list(model.mu.parameters())
                           + list(model.dec.parameters()), lr=2e-3)
    bs = min(256, n)
    for _ in range(pre):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            xb = Xt[perm[i:i+bs]]; z = model.mu(model.enc(xb)); loss = F.mse_loss(model.dec(z), xb)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Zpre = model.mu(model.enc(Xt)).cpu().numpy()
    gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=seed,
                          reg_covar=1e-4, n_init=3).fit(Zpre)
    model.pi.data = torch.log(torch.from_numpy(gmm.weights_).float() + 1e-9).to(dev)
    model.mu_c.data = torch.from_numpy(gmm.means_).float().to(dev)
    model.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).float() + 1e-6).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(joint):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            xb = Xt[perm[i:i+bs]]; xh, mu, logvar, z = model(xb)
            loss = vade_loss(model, xb, xh, mu, logvar, z, alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
    with torch.no_grad():
        mu, _ = model.encode(Xt); g = model.gamma(mu)
        return mu.cpu().numpy(), g.argmax(1).cpu().numpy()

CONFIGS = {
    "kmeans": None,
    "vade_default":      lambda X, K, s: _vade_custom(X, K, s),
    "vade_joint150":     lambda X, K, s: _vade_custom(X, K, s, joint=150),
    "vade_pre100j150":   lambda X, K, s: _vade_custom(X, K, s, pre=100, joint=150),
    "vade_protectclust": lambda X, K, s: _vade_custom(X, K, s, alpha_mult=0.25),
    "vade_boostclust":   lambda X, K, s: _vade_custom(X, K, s, alpha_mult=0.1, joint=150),
    "vade_latent16":     lambda X, K, s: _vade_custom(X, K, s, latent=16),
}
KS = [30, 50]; SELS = ["smallest", "random", "hard"]; M = 10; SEEDS = [0, 1, 2]
CEILING = ["Annthyroid", "WDBC", "Cardiotocography", "PageBlocks"]

def regret_cfg(ds, seed, fn):
    parts = []; effK = []
    for K in KS:
        pre = fn(ds.X, K, seed) if fn else None
        if pre is not None:
            effK.append(len(np.unique(pre[1])))
        for sel in SELS:
            p = pseudo_auc_for_dataset(ds, K=K, M=M, selection=sel, seed=seed, precomputed=pre)
            p["regime"] = f"{sel}_K{K}"; parts.append(p)
    bank = pd.concat(parts, ignore_index=True)
    tr = true_rank_from_labels(ds, seed=seed).set_index("detector").true_auc.dropna()
    score = _agg_discriminative(bank); score = score[score.index.isin(tr.index)]
    if score.empty:
        return np.nan, (np.mean(effK) if effK else np.nan)
    pick = score.sort_values(ascending=False).index[0]
    return tr.max() - tr[pick], (np.mean(effK) if effK else np.nan)

def main():
    print("[train] fetching DAMI ceiling datasets..."); sys.stdout.flush()
    dss = {nm: fetch_dami(nm) for nm in CEILING}
    print("[train] fetched", {k: v.X.shape for k, v in dss.items()}); sys.stdout.flush()
    rows = []
    for nm, ds in dss.items():
        for cfg, fn in CONFIGS.items():
            reg = []; ek = []
            for s in SEEDS:
                t = time.time()
                try:
                    r, k = regret_cfg(ds, s, fn)
                except Exception as e:
                    r, k = np.nan, np.nan; print(f"  {nm}/{cfg}/s{s} FAIL {e}", flush=True)
                if not np.isnan(r): reg.append(r)
                if not np.isnan(k): ek.append(k)
                rr = "nan" if np.isnan(r) else round(r, 3)
                kk = "nan" if np.isnan(k) else round(k, 1)
                print(f"  {nm}/{cfg}/s{s}: regret={rr} effK={kk} ({time.time()-t:.0f}s)", flush=True)
            rows.append({"dataset": nm, "config": cfg,
                         "regret": np.mean(reg) if reg else np.nan,
                         "effK": np.mean(ek) if ek else np.nan})
            os.makedirs("results", exist_ok=True)
            pd.DataFrame(rows).to_csv("results/vade_tuned.csv", index=False)
    df = pd.DataFrame(rows)
    print("\n[train] === regret by config (kmeans=baseline; lower better) ===")
    print(df.pivot_table(index="dataset", columns="config", values="regret").round(3).to_string())
    print("[train] === DONE ==="); sys.stdout.flush()

if __name__ == "__main__":
    main()
