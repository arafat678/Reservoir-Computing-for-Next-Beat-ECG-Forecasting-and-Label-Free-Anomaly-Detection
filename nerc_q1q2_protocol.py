#!/usr/bin/env python3
"""
nerc_q1q2_protocol.py  --  one unified protocol for the GA-tuned reservoir ECG paper.

Implements the revision steps:
  (2) 22-test-patient split (split A), echo-state / Lyapunov test, one-at-a-time sensitivity,
      parameter count and per-beat latency
  (3) beats from annotation positions (default) or a standard detector (wfdb xqrs); results with and
      without the noisy records (default 219 220 231; only those that are test records are dropped)
  (4) mean-of-last-k-beats baselines (k chosen on validation), persistence, linear ridge (1 and 3 beats)
  (5) record-level bootstrap CIs (RMSE, AUC, AP and paired differences), global vs per-patient threshold,
      false alarms per hour.  The beat classifier is dropped.
  GA: widened bounds, gamma/beta in the genome (dendritic model only), same budget for ESN and dendritic.

Usage
  pip install wfdb scikit-learn scipy numpy
  OMP_NUM_THREADS=1 python nerc_q1q2_protocol.py --db mitdb --split A --pop 30 --gens 40 --jobs 5 --out out_mitdb
  python nerc_q1q2_protocol.py --db synthetic --pop 6 --gens 3 --seeds 42 123 --boot 50 --out out_test   # smoke test
Everything printed to <out>/ is produced by this script from the data; nothing is hard-coded.
"""
import argparse, json, os, time, sys, warnings
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
W_OUT = 10  # washout beats, applied identically to every model in fitting and scoring
D = 100
NORMAL = set("NLRej")
ABNORMAL = set("AaJSVEF")
BEAT_SYMS = NORMAL | ABNORMAL | set("/fQ")

SPLITS = {
    "A": dict(  # 44 records, paced excluded, 22 test patients
        train=[101, 106, 109, 114, 115, 116, 118, 119, 124, 203, 205, 207, 209, 215, 223, 230],
        val=[122, 208, 220], calib=[108, 112, 201],
        test=[100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222,
              228, 231, 232, 233, 234]),
    "B": dict(
        train=[100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 115, 117, 118, 119, 121, 122, 123, 124],
        val=[102, 104, 107, 114, 116, 200], calib=[201, 202, 203, 205, 207, 208],
        test=[209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 228, 230, 231, 232, 233, 234]),
}
BOUNDS = dict(N=(50, 300), rho=(0.1, 1.6), kappa=(0.02, 0.30), s=(0.005, 2.0), alpha=(0.02, 1.0),
              lam=(1e-8, 1.0), gamma=(0.0, 2.0), beta=(0.1, 4.0))
LOGP = {"s", "lam"}
GENES_ESN = ["N", "rho", "kappa", "s", "alpha", "lam"]
GENES_DEN = GENES_ESN + ["gamma", "beta"]


# ----------------------------------------------------------------------------- data
class Rec:
    def __init__(self, sig, fs, samp, sym):
        self.sig, self.fs, self.samp, self.sym = sig, fs, np.asarray(samp), np.asarray(sym)
        self.lab = np.array([0 if s in NORMAL else 1 if s in ABNORMAL else -1 for s in self.sym])


def load_mitdb(rid, beats):
    import wfdb
    rec = wfdb.rdrecord(str(rid), pn_dir="mitdb")
    ann = wfdb.rdann(str(rid), "atr", pn_dir="mitdb")
    names = [n.upper() for n in rec.sig_name]
    sig = rec.p_signal[:, names.index("MLII") if "MLII" in names else 0]
    keep = [i for i, s in enumerate(ann.symbol) if s in BEAT_SYMS]
    samp, sym = np.asarray(ann.sample)[keep], np.asarray(ann.symbol)[keep]
    if beats == "xqrs":  # standard detector; labels from nearest annotation within 120 ms, else unlabelled
        from wfdb import processing
        det = np.asarray(processing.xqrs_detect(sig, rec.fs, verbose=False))
        j = np.clip(np.searchsorted(samp, det), 1, len(samp) - 1)
        near = np.where(np.abs(samp[j - 1] - det) <= np.abs(samp[j] - det), j - 1, j)
        ok = np.abs(samp[near] - det) <= 0.12 * rec.fs
        sym = np.where(ok, sym[near], "?")
        samp = det
    return Rec(sig, rec.fs, samp, sym)


def synth_record(rid, rng, n=420, fs=360):
    """Smoke-test data only. Not a substitute for MIT-BIH."""
    amp = rng.uniform(0.7, 1.4)
    rr = rng.normal(0.8, 0.05, n).clip(0.45, 1.2)
    abn = rng.random(n) < 0.12
    rr = np.where(abn, rr * 0.7, rr)
    rr[1:] = np.where(abn[:-1], rr[1:] * 1.3, rr[1:])
    samp = (np.cumsum(rr) * fs).astype(int) + fs
    sig_len = samp[-1] + fs
    sig = rng.normal(0, 0.03, sig_len) + 0.1 * np.sin(np.arange(sig_len) / fs * 0.3)
    
    # Vectorized fast Gaussian pulse calculation
    for p, a in zip(samp, abn):
        w, h = (0.03, 1.5) if a else (0.011, 1.0)
        # Pulse 1: QRS
        rad1 = int(np.ceil(5 * w * fs))
        i0, i1 = max(0, p - rad1), min(sig_len, p + rad1 + 1)
        idx1 = np.arange(i0, i1)
        sig[i0:i1] += amp * h * np.exp(-0.5 * ((idx1 - p) / (w * fs)) ** 2)
        
        # Pulse 2: T wave
        pt = p + 0.27 * fs
        wt = 0.05
        rad2 = int(np.ceil(5 * wt * fs))
        j0, j1 = max(0, int(pt - rad2)), min(sig_len, int(pt + rad2 + 1))
        idx2 = np.arange(j0, j1)
        sig[j0:j1] += amp * (-0.3 if a else 0.25) * np.exp(-0.5 * ((idx2 - pt) / (wt * fs)) ** 2)
        
    return Rec(sig, fs, samp, np.where(abn, "V", "N"))


# Precompute interpolation matrices for fast vectorized feature extraction
_H_360 = int(round(0.150 * 360))
_L_360 = 2 * _H_360
_M1 = np.array([np.interp(np.linspace(0, _L_360 - 1, 50), np.arange(_L_360), col) for col in np.eye(_L_360)])
_L2_360 = _L_360 // 2
_M2 = np.array([np.interp(np.linspace(0, _L2_360 - 1, 50), np.arange(_L2_360), col) for col in np.eye(_L2_360)])

def beat_features(rec):
    sos = butter(4, [0.5, 40.0], btype="band", fs=rec.fs, output="sos")
    s = sosfiltfilt(sos, rec.sig)  # zero-phase (offline); note: non-causal
    h = int(round(0.150 * rec.fs))
    
    valid_idx = []
    valid_samp = []
    for i, p in enumerate(rec.samp):
        if 0 <= p - h and p + h <= len(s):
            valid_idx.append(i)
            valid_samp.append(p)
            
    if not valid_samp:
        return np.empty((0, D)), np.empty(0), np.empty(0)
        
    windows = np.array([s[p - h:p + h] for p in valid_samp])
    windows -= windows.mean(axis=1, keepdims=True)
    
    time_feats = windows @ _M1
    freq_feats = np.abs(np.fft.rfft(windows, axis=1)[:, 1:]) @ _M2
    feats = np.hstack([time_feats, freq_feats])
    
    keep = np.array(valid_idx)
    return feats, rec.lab[keep], rec.samp[keep]


class Data:
    def __init__(self, split, db, beats, seed=0):
        self.split, self.fs = split, 360
        rng = np.random.default_rng(seed)
        raw = {}
        all_rids = sum(split.values(), [])
        print(f"Loading {len(all_rids)} records (db={db}, beats={beats})...", flush=True)
        for idx, rid in enumerate(all_rids):
            r = synth_record(rid, rng) if db == "synthetic" else load_mitdb(rid, beats)
            raw[rid] = beat_features(r)
            self.fs = r.fs
            if (idx + 1) % 10 == 0 or idx + 1 == len(all_rids):
                print(f"  [{idx + 1}/{len(all_rids)}] records processed", flush=True)
        tr = np.vstack([raw[r][0] for r in split["train"]])
        a, b = tr.min(0), tr.max(0)
        self.U = {r: np.clip((raw[r][0] - a) / np.where(b - a > 1e-12, b - a, 1.0), 0, 1) for r in raw}
        self.lab = {r: raw[r][1] for r in raw}
        self.samp = {r: raw[r][2] for r in raw}


# ----------------------------------------------------------------------------- reservoir
def build(g, seed, dend):
    rng = np.random.default_rng(seed)
    N = int(g["N"])
    W0 = rng.standard_normal((N, N)) * (rng.random((N, N)) < g["kappa"])
    ev = np.abs(np.linalg.eigvals(W0)).max()
    W = W0 * (g["rho"] / ev) if ev > 1e-12 else W0
    Win = rng.uniform(-g["s"], g["s"], (N, D))
    Wd, c = rng.normal(0, 0.25, (N, D)), rng.normal(0, 0.1, N)  # always drawn: ESN and dendritic share matrices
    return dict(N=N, W=W, Win=Win, Wd=Wd, c=c, a=g["alpha"], gam=g.get("gamma", 0.5), beta=g.get("beta", 1.0),
                dend=dend)


def drive(R, U):
    A = U @ R["Win"].T
    Dd = np.tanh(U @ R["Wd"].T + R["c"]) if R["dend"] else None
    return A, Dd


def step(R, x, A, Dd, t):
    r = R["W"] @ x + A[t]
    if R["dend"]:
        cand = np.tanh(r + R["gam"] * Dd[t]) * 0.5 * (1 + np.tanh(0.5 * R["beta"] * r))  # sigma(beta r)
    else:
        cand = np.tanh(r)
    return (1 - R["a"]) * x + R["a"] * cand


def states(R, U, x0=None):
    A, Dd = drive(R, U)
    x = np.zeros(R["N"]) if x0 is None else x0.copy()
    X = np.empty((len(U), R["N"]))
    for t in range(len(U)):
        x = step(R, x, A, Dd, t)
        X[t] = x
    return X


def fit_readout(X, Y, lam):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    L = lam * np.eye(Xa.shape[1])
    L[-1, -1] = 0.0  # intercept unpenalised
    try:
        return np.linalg.solve(Xa.T @ Xa + L, Xa.T @ Y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Xa.T @ Xa + L, Xa.T @ Y, rcond=None)[0]


aug = lambda X: np.hstack([X, np.ones((len(X), 1))])


def fit_reservoir(R, data, lam):
    Xs, Ys = [], []
    for rid in data.split["train"]:
        U = data.U[rid]
        Xs.append(states(R, U)[W_OUT:-1]); Ys.append(U[W_OUT + 1:])
    return fit_readout(np.vstack(Xs), np.vstack(Ys), lam)


# ----------------------------------------------------------------------------- GA
def rand_gene(k, rng):
    lo, hi = BOUNDS[k]
    if k == "N": return int(rng.integers(lo, hi + 1))
    if k in LOGP: return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
    return float(rng.uniform(lo, hi))


def mutate(g, genes, rng):
    g = dict(g)
    for k in genes:
        lo, hi = BOUNDS[k]
        if k == "N":
            if rng.random() < 0.25: g[k] = rand_gene(k, rng)
        elif rng.random() < 0.5:
            if k in LOGP:
                v = np.log10(g[k]) + rng.normal(0, 0.12 * (np.log10(hi) - np.log10(lo)))
                g[k] = float(10 ** np.clip(v, np.log10(lo), np.log10(hi)))
            else:
                g[k] = float(np.clip(g[k] + rng.normal(0, 0.12 * (hi - lo)), lo, hi))
    return g


def fitness(g, dend, seed, data):
    R = build(g, seed, dend)
    try:
        Wo = fit_reservoir(R, data, g["lam"])
        mse = []
        for rid in data.split["val"]:
            U = data.U[rid]
            P = aug(states(R, U)[W_OUT:-1]) @ Wo
            mse.append(np.mean((P - U[W_OUT + 1:]) ** 2))
        f = float(np.mean(mse))
        return f if np.isfinite(f) else 1e3
    except Exception:
        return 1e3


def run_ga(dend, seed, data, pop, gens):
    genes = GENES_DEN if dend else GENES_ESN
    rng = np.random.default_rng(seed + 1000)
    cache = {}

    def fit(g):
        key = tuple(round(float(g[k]), 8) for k in genes)
        if key not in cache: cache[key] = (fitness(g, dend, seed, data), dict(g))
        return cache[key][0]

    popn = [{k: rand_gene(k, rng) for k in genes} for _ in range(pop)]
    hist = []
    for gen in range(gens):
        f = np.array([fit(g) for g in popn])
        hist.append(float(f.min()))
        if gen == gens - 1: break
        el = [popn[i] for i in np.argsort(f)[:max(2, pop // 4)]]
        new = [dict(e) for e in el]
        while len(new) < pop:
            a, b = rng.choice(len(el), 2, replace=False)
            if rng.random() < 0.7:
                child = {k: (el[a][k] if rng.random() < 0.5 else el[b][k]) for k in genes}
            else:
                child = dict(el[a])
            new.append(mutate(child, genes, rng))
        popn = new
    bf, bg = min(cache.values(), key=lambda v: v[0])
    return bg, bf, hist


def bound_hits(g, genes, tol=0.02):
    out = []
    for k in genes:
        lo, hi = BOUNDS[k]
        v, a, b = (np.log10(g[k]), np.log10(lo), np.log10(hi)) if k in LOGP else (g[k], lo, hi)
        if v - a < tol * (b - a): out.append(k + ":lower")
        if b - v < tol * (b - a): out.append(k + ":upper")
    return out


# ----------------------------------------------------------------------------- scoring
def part(pred, U, lab, samp, fs):
    tgt = U[W_OUT + 1:]
    e2 = (pred - tgt) ** 2
    sc = np.sqrt(e2.mean(1))
    lb, tt = lab[W_OUT + 1:], samp[W_OUT + 1:]
    m = lb >= 0
    return dict(sse=float(e2.sum()), n=int(e2.size), sc=sc[m], lb=lb[m], tt=tt[m], sc_all=sc, fs=fs)


def meank_pred(U, k):
    cs = np.vstack([np.zeros((1, U.shape[1])), np.cumsum(U, 0)])
    t = np.arange(W_OUT, len(U) - 1)
    return (cs[t + 1] - cs[t + 1 - k]) / k


def ridge_feats(U, k):
    t = np.arange(W_OUT, len(U) - 1)
    return np.hstack([U[t - j] for j in range(k)] + [np.ones((len(t), 1))])


def parts_for(fn, data, rids):
    return {r: part(fn(r), data.U[r], data.lab[r], data.samp[r], data.fs) for r in rids}


def pooled(parts):
    sse = sum(p["sse"] for p in parts); n = sum(p["n"] for p in parts)
    sc = np.concatenate([p["sc"] for p in parts]); lb = np.concatenate([p["lb"] for p in parts])
    try:
        return dict(rmse=float(np.sqrt(sse / n)), auc=float(roc_auc_score(lb, sc)),
                    ap=float(average_precision_score(lb, sc)))
    except ValueError:
        return dict(rmse=float(np.sqrt(sse / n)), auc=np.nan, ap=np.nan)


def thr_metrics(parts, mode, q, calib_sc=None, M=300):
    TP = FP = TN = FN = 0; hours = 0.0
    tau_g = np.quantile(calib_sc, q) if mode == "global" else None
    for p in parts:
        if mode == "global":
            sc, lb, tt, tau = p["sc"], p["lb"], p["tt"], tau_g
        else:  # per patient: threshold from the first M beats of the same record, scored on the rest; no labels used
            if len(p["sc"]) <= M + 10: continue
            tau = np.quantile(p["sc_all"][:M], q)
            sc, lb, tt = p["sc"][M:], p["lb"][M:], p["tt"][M:]
        f = sc > tau
        TP += int((f & (lb == 1)).sum()); FP += int((f & (lb == 0)).sum())
        FN += int((~f & (lb == 1)).sum()); TN += int((~f & (lb == 0)).sum())
        hours += (tt[-1] - tt[0]) / p["fs"] / 3600
    pr = TP / max(TP + FP, 1); rc = TP / max(TP + FN, 1)
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-12), normal_flagged=FP / max(FP + TN, 1),
                fa_per_hour=FP / max(hours, 1e-9))


# ----------------------------------------------------------------------------- one (model, seed) task
def task(args):
    name, dend, seed, pop, gens = args
    data = TASK_DATA
    t0 = time.time()
    g, fbest, hist = run_ga(dend, seed, data, pop, gens)
    R = build(g, seed, dend)
    Wo = fit_reservoir(R, data, g["lam"])
    fn = lambda r: aug(states(R, data.U[r])[W_OUT:-1]) @ Wo
    sp = data.split
    out = dict(name=name, seed=seed, genome=g, val_mse=fbest, hist=hist, hits=bound_hits(g, GENES_DEN if dend else GENES_ESN),
               parts={s: parts_for(fn, data, sp[s]) for s in ("val", "calib", "test")}, secs=time.time() - t0)
    return out, R, Wo


TASK_DATA = None


def task_mp(a):
    out, R, Wo = task(a)
    out["R"], out["Wo"] = R, Wo
    return out


# ----------------------------------------------------------------------------- dynamics, sensitivity, cost
def dynamics(R, data, rids, rng, steps=1500):
    lam, curves = [], []
    for rid in rids:
        U = data.U[rid][:steps]
        A, Dd = drive(R, U)
        x = np.zeros(R["N"]); v = rng.standard_normal(R["N"]); v /= np.linalg.norm(v)
        eps = 1e-6; xp = x + eps * v; s = 0.0; k = 0
        for t in range(len(U)):
            x, xp = step(R, x, A, Dd, t), step(R, xp, A, Dd, t)
            d = np.linalg.norm(xp - x)
            if d < 1e-300: break
            if t >= 50: s += np.log(d / eps); k += 1
            xp = x + (xp - x) * eps / d
        lam.append(s / max(k, 1))
    U = data.U[rids[0]][:400]
    A, Dd = drive(R, U)
    xa, xb = rng.normal(0, 0.5, R["N"]), rng.normal(0, 0.5, R["N"])
    dist = []
    for t in range(len(U)):
        xa, xb = step(R, xa, A, Dd, t), step(R, xb, A, Dd, t)
        dist.append(float(np.linalg.norm(xa - xb)))
    pick = {str(t): dist[t - 1] for t in (1, 10, 50, 100, 300) if t <= len(dist)}
    return dict(lyap_mean=float(np.mean(lam)), lyap_sd=float(np.std(lam)), lyap_max=float(np.max(lam)),
                share_negative=float(np.mean(np.array(lam) < 0)), two_copy_distance=pick)


def sensitivity(g, dend, seed, data, pts=9):
    rows = []
    for k in (GENES_DEN if dend else GENES_ESN):
        lo, hi = BOUNDS[k]
        grid = (np.linspace(lo, hi, pts).astype(int) if k == "N" else
                np.logspace(np.log10(lo), np.log10(hi), pts) if k in LOGP else np.linspace(lo, hi, pts))
        for v in grid:
            gg = dict(g); gg[k] = int(v) if k == "N" else float(v)
            rows.append((k, float(v), fitness(gg, dend, seed, data)))
    return rows


def cost(R, Wo, data, rids, n=1500):
    U = np.vstack([data.U[r] for r in rids])[:n]
    x = np.zeros(R["N"]); ts = []
    for t in range(len(U)):
        t0 = time.perf_counter()
        a = R["Win"] @ U[t]
        r = R["W"] @ x + a
        if R["dend"]:
            d = np.tanh(R["Wd"] @ U[t] + R["c"])
            cand = np.tanh(r + R["gam"] * d) * 0.5 * (1 + np.tanh(0.5 * R["beta"] * r))
        else:
            cand = np.tanh(r)
        x = (1 - R["a"]) * x + R["a"] * cand
        _ = np.append(x, 1.0) @ Wo
        ts.append((time.perf_counter() - t0) * 1e3)
    nnz = int((R["W"] != 0).sum())
    stored = Wo.size + nnz + R["Win"].size + ((R["Wd"].size + R["c"].size) if R["dend"] else 0)
    return dict(N=R["N"], readout_params=int(Wo.size), recurrent_nonzeros=nnz, stored_params=int(stored),
                latency_ms_mean=float(np.mean(ts)), latency_ms_p95=float(np.percentile(ts, 95)))


# ----------------------------------------------------------------------------- main
def ci(x):
    x = np.asarray(x, float)
    return np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5)


def main():
    global TASK_DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mitdb", choices=["mitdb", "synthetic"])
    ap.add_argument("--beats", default="annotation", choices=["annotation", "xqrs"])
    ap.add_argument("--split", default="A", choices=["A", "B"])
    ap.add_argument("--pop", type=int, default=30); ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2025, 3407, 2026])
    ap.add_argument("--boot", type=int, default=500); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--exclude", type=int, nargs="*", default=[219, 220, 231])
    ap.add_argument("--q", type=float, nargs="+", default=[0.95, 0.99])
    ap.add_argument("--calib-beats", type=int, default=300)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)

    data = Data(SPLITS[a.split], a.db, a.beats)
    TASK_DATA = data
    sp = data.split
    test = sp["test"]
    print(f"split {a.split}: train {len(sp['train'])} val {len(sp['val'])} calib {len(sp['calib'])} test {len(test)}; "
          f"beats={a.beats}; db={a.db}", flush=True)

    # ---- deterministic baselines: persistence, mean-of-last-k, ridge on 1 and 3 beats
    res = {}  # model -> list over seeds of {split: {rid: part}}
    def det(fn):
        return [{s: parts_for(fn, data, sp[s]) for s in ("val", "calib", "test")}]
    res["persistence"] = det(lambda r: data.U[r][W_OUT:-1])
    for k in (2, 3, 5, 10):
        res[f"mean_{k}"] = det(lambda r, k=k: meank_pred(data.U[r], k))
    for k in (1, 3):
        Xtr = np.vstack([ridge_feats(data.U[r], k) for r in sp["train"]])
        Ytr = np.vstack([data.U[r][W_OUT + 1:] for r in sp["train"]])
        best = None
        for lam in 10.0 ** np.arange(-8, 1):
            L = lam * np.eye(Xtr.shape[1]); L[-1, -1] = 0
            Wr = np.linalg.solve(Xtr.T @ Xtr + L, Xtr.T @ Ytr)
            v = np.mean([np.mean((ridge_feats(data.U[r], k) @ Wr - data.U[r][W_OUT + 1:]) ** 2) for r in sp["val"]])
            if best is None or v < best[0]: best = (v, Wr)
        res[f"ridge_{k}"] = det(lambda r, k=k, Wr=best[1]: ridge_feats(data.U[r], k) @ Wr)
    kbest = min((2, 3, 5, 10), key=lambda k: pooled(list(res[f"mean_{k}"][0]["val"].values()))["rmse"])
    res["mean_k_best"] = res[f"mean_{kbest}"]
    print(f"mean-of-last-k: k={kbest} selected on validation RMSE", flush=True)

    # ---- GA-tuned models (same budget); one GA per (model, seed)
    jobs = [(nm, dn, s, a.pop, a.gens) for nm, dn in (("esn_ga", False), ("dendritic_ga", True)) for s in a.seeds]
    print(f"Starting GA optimization for {len(jobs)} jobs...", flush=True)
    if a.jobs > 1:
        if os.name == "nt":
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=a.jobs) as ex:
                outs = list(ex.map(task_mp, jobs))
        else:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(a.jobs) as pool:
                outs = pool.map(task_mp, jobs)
    else:
        outs = []
        for j in jobs:
            o = task_mp(j); outs.append(o)
            print(f"  {o['name']} seed {o['seed']}: val_mse {o['val_mse']:.5f} hits {o['hits']} ({o['secs']:.0f}s)", flush=True)
    genomes = {}
    for nm in ("esn_ga", "dendritic_ga"):
        res[nm] = [o["parts"] for o in outs if o["name"] == nm]
        genomes[nm] = [dict(seed=o["seed"], genome=o["genome"], val_mse=o["val_mse"], bound_hits=o["hits"],
                            hist=o["hist"]) for o in outs if o["name"] == nm]
    json.dump(genomes, open(f"{a.out}/genomes.json", "w"), indent=1)

    # ---- evaluation tables, with and without noisy records
    print("Generating evaluation tables...", flush=True)
    def tables(rids, tag):
        idx = [rng.integers(0, len(rids), len(rids)) for _ in range(a.boot)]
        models = list(res)
        def pv(sd, rr):
            q = pooled([sd["test"][r] for r in rr]); return [q["rmse"], q["auc"], q["ap"]]
        full = {m: np.mean([pv(sd, rids) for sd in res[m]], 0) for m in models}
        bs = {m: np.array([np.mean([pv(sd, [rids[i] for i in ix]) for sd in res[m]], 0) for ix in idx]) for m in models}
        L = [f"### Test records ({tag}): n = {len(rids)}; mean over seeds; 95% CI = record-level bootstrap ({a.boot})\n",
             "| Model | RMSE | AUC | AP | dRMSE vs persistence | dRMSE vs mean-k | dAUC vs persistence |", "|---|---|---|---|---|---|---|"]
        for m in models:
            f, b = full[m], bs[m]
            d1 = b[:, 0] - bs["persistence"][:, 0]; d2 = b[:, 0] - bs["mean_k_best"][:, 0]; d3 = b[:, 1] - bs["persistence"][:, 1]
            c = lambda v, i: f"{v[i]:.4f} [{ci(b[:, i])[0]:.4f}, {ci(b[:, i])[1]:.4f}]"
            L.append(f"| {m} | {c(f,0)} | {f[1]:.3f} [{ci(b[:,1])[0]:.3f}, {ci(b[:,1])[1]:.3f}] | "
                     f"{f[2]:.3f} [{ci(b[:,2])[0]:.3f}, {ci(b[:,2])[1]:.3f}] | "
                     f"{f[0]-full['persistence'][0]:+.4f} [{ci(d1)[0]:+.4f}, {ci(d1)[1]:+.4f}] | "
                     f"{f[0]-full['mean_k_best'][0]:+.4f} [{ci(d2)[0]:+.4f}, {ci(d2)[1]:+.4f}] | "
                     f"{f[1]-full['persistence'][1]:+.3f} [{ci(d3)[0]:+.3f}, {ci(d3)[1]:+.3f}] |")
        # per-record RMSE of the GA models vs persistence (how many records does it win?)
        L += ["", "Records where model RMSE < persistence RMSE (seed mean):"]
        for m in ("mean_k_best", "esn_ga", "dendritic_ga"):
            w = 0
            for r in rids:
                pm = np.mean([np.sqrt(sd["test"][r]["sse"] / sd["test"][r]["n"]) for sd in res[m]])
                pp = np.sqrt(res["persistence"][0]["test"][r]["sse"] / res["persistence"][0]["test"][r]["n"])
                w += pm < pp
            L.append(f"- {m}: {w}/{len(rids)}")
        # thresholds
        L += ["", f"### Alarm behaviour ({tag}); threshold quantile q; per-patient uses first {a.calib_beats} beats of each record, "
                  "scored on the rest\n", "| Model | rule | q | precision | recall | F1 | normal beats flagged | false alarms / h |",
              "|---|---|---|---|---|---|---|---|"]
        for m in ("persistence", "mean_k_best", "esn_ga", "dendritic_ga"):
            for mode in ("global", "per_patient"):
                for q in a.q:
                    vals = []
                    for sd in res[m]:
                        cal = np.concatenate([p["sc_all"] for p in sd["calib"].values()])
                        vals.append(thr_metrics([sd["test"][r] for r in rids], mode, q, cal, a.calib_beats))
                    mm = {k: np.mean([v[k] for v in vals]) for k in vals[0]}
                    L.append(f"| {m} | {mode} | {q} | {mm['precision']:.3f} | {mm['recall']:.3f} | {mm['f1']:.3f} | "
                             f"{mm['normal_flagged']:.3f} | {mm['fa_per_hour']:.1f} |")
        return "\n".join(L) + "\n"

    md = tables(test, "all test records")
    excl = [r for r in a.exclude if r in test]
    skipped = [r for r in a.exclude if r not in test]
    if excl:
        md += "\n" + tables([r for r in test if r not in excl], f"without {excl}")
    if skipped:
        md += f"\nNote: {skipped} are not test records in split {a.split} and were not removed.\n"
    open(f"{a.out}/results_models_and_alarms.md", "w").write(md)
    print(md, flush=True)

    # ---- per-record RMSE csv
    with open(f"{a.out}/per_record_rmse.csv", "w") as f:
        ms = list(res); f.write("record," + ",".join(ms) + "\n")
        for r in test:
            f.write(f"{r}," + ",".join(f"{np.mean([np.sqrt(sd['test'][r]['sse']/sd['test'][r]['n']) for sd in res[m]]):.5f}"
                                       for m in ms) + "\n")

    # ---- dynamics / sensitivity / cost on seed-0 models
    print("Computing dynamics, sensitivity, and cost...", flush=True)
    extra = {}
    for nm, dn in (("esn_ga", False), ("dendritic_ga", True)):
        o = next(o for o in outs if o["name"] == nm)
        extra[nm] = dict(dynamics=dynamics(o["R"], data, test, rng), cost=cost(o["R"], o["Wo"], data, test))
        rows = sensitivity(o["genome"], dn, o["seed"], data)
        with open(f"{a.out}/sensitivity_{nm}.csv", "w") as f:
            f.write("param,value,val_mse\n"); [f.write(f"{k},{v},{m}\n") for k, v, m in rows]
    json.dump(extra, open(f"{a.out}/dynamics_cost.json", "w"), indent=1)
    print(json.dumps(extra, indent=1), flush=True)
    print("done ->", a.out, flush=True)


if __name__ == "__main__":
    main()