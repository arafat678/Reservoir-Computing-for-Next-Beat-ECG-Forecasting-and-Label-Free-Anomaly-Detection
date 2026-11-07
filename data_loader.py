"""
data_loader.py
==============
Data loading, preprocessing, feature extraction, and Split A partitioning for the MIT-BIH Arrhythmia Database.

Matches Section 4 & 5.1 of:
"Genetically Tuned Reservoir Computing for Next-Beat ECG Forecasting and Label-Free Anomaly Detection:
A Negative-Findings Benchmark Against Simple Sequence Baselines"
(Arafat Ali Mondal and Ankush Kumar)
"""

import os
import warnings
import numpy as np
from scipy.signal import butter, sosfiltfilt
import wfdb

warnings.filterwarnings("ignore")

D = 100  # 50 time-domain + 50 frequency-domain features
NORMAL_SYMBOLS = set("NLRej")
ABNORMAL_SYMBOLS = set("AaJSVEF")
BEAT_SYMBOLS = NORMAL_SYMBOLS | ABNORMAL_SYMBOLS | set("/fQ")

# Pre-registered Split A (44 unpaced records: 16 train, 3 val, 3 calib, 22 test)
SPLITS = {
    "A": {
        "train": [101, 106, 109, 114, 115, 116, 118, 119, 124, 203, 205, 207, 209, 215, 223, 230],
        "val":   [122, 208, 220],
        "calib": [108, 112, 201],
        "test":  [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222,
                  228, 231, 232, 233, 234]
    }
}

# Pre-computed interpolation matrices for fast vectorized feature extraction (360 Hz)
_H_360 = int(round(0.150 * 360))  # 54 samples (150 ms)
_L_360 = 2 * _H_360               # 108 samples (300 ms window)
_M1 = np.array([np.interp(np.linspace(0, _L_360 - 1, 50), np.arange(_L_360), col) for col in np.eye(_L_360)])
_L2_360 = _L_360 // 2
_M2 = np.array([np.interp(np.linspace(0, _L2_360 - 1, 50), np.arange(_L2_360), col) for col in np.eye(_L2_360)])


def get_local_mitdb_path():
    """Finds the local MIT-BIH directory if available."""
    candidates = [
        r"F:\chatWithAi\mitdb",
        r"C:\Users\Administrator\mitdb_data",
        os.path.join(os.path.dirname(__file__), "mitdb"),
        os.path.join(os.path.dirname(__file__), "..", "mitdb")
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "100.dat")):
            return p
    return None


class Record:
    """Container for raw signal and annotated beats."""
    def __init__(self, sig, fs, samp, sym):
        self.sig = sig
        self.fs = fs
        self.samp = np.asarray(samp, dtype=int)
        self.sym = np.asarray(sym)
        # Normal = 0, Abnormal = 1, Excluded/Unlabelled = -1
        self.lab = np.array([0 if s in NORMAL_SYMBOLS else 1 if s in ABNORMAL_SYMBOLS else -1 for s in self.sym])


def load_mitdb_record(rid, beats="annotation", local_dir=None):
    """
    Loads an MIT-BIH record from local disk or downloads via WFDB.
    Extracts the MLII lead (or first channel if MLII is absent).
    """
    if local_dir is None:
        local_dir = get_local_mitdb_path()

    record_str = str(rid)
    if local_dir and os.path.exists(os.path.join(local_dir, f"{record_str}.dat")):
        record_path = os.path.join(local_dir, record_str)
        rec = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, "atr")
    else:
        rec = wfdb.rdrecord(record_str, pn_dir="mitdb")
        ann = wfdb.rdann(record_str, "atr", pn_dir="mitdb")

    names = [n.upper() for n in rec.sig_name]
    sig_idx = names.index("MLII") if "MLII" in names else 0
    sig = rec.p_signal[:, sig_idx]

    keep = [i for i, s in enumerate(ann.symbol) if s in BEAT_SYMBOLS]
    samp = np.asarray(ann.sample)[keep]
    sym = np.asarray(ann.symbol)[keep]

    if beats == "xqrs":
        # Standard QRS detector; label is nearest annotation within 120 ms
        from wfdb import processing
        det = np.asarray(processing.xqrs_detect(sig, rec.fs, verbose=False))
        if len(samp) > 0 and len(det) > 0:
            j = np.clip(np.searchsorted(samp, det), 1, len(samp) - 1)
            near = np.where(np.abs(samp[j - 1] - det) <= np.abs(samp[j] - det), j - 1, j)
            ok = np.abs(samp[near] - det) <= 0.12 * rec.fs
            sym = np.where(ok, sym[near], "?")
            samp = det
        else:
            samp = det
            sym = np.array(["?"] * len(det))

    return Record(sig, rec.fs, samp, sym)


def extract_beat_features(rec):
    """
    Preprocesses raw ECG and extracts 100-D feature vectors:
      1. Zero-phase 4th-order Butterworth bandpass (0.5 - 40 Hz).
      2. Beat windowing +/- 150 ms (54 samples) centered at R-peak.
      3. Mean removal across window.
      4. 50-point waveform resampling (time-domain).
      5. 50-point spectral magnitude DFT (frequency-domain).
    """
    sos = butter(4, [0.5, 40.0], btype="band", fs=rec.fs, output="sos")
    s = sosfiltfilt(sos, rec.sig)
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


class Dataset:
    """
    Loads and normalizes the full dataset across partitions.
    Min-max scaling is fitted strictly on training records and clipped to [0, 1].
    """
    def __init__(self, split_name="A", beats="annotation", local_dir=None, verbose=True):
        self.split = SPLITS[split_name]
        self.fs = 360
        self.local_dir = local_dir if local_dir else get_local_mitdb_path()

        raw = {}
        all_rids = sum(self.split.values(), [])
        if verbose:
            print(f"Loading {len(all_rids)} records from MIT-BIH (Split {split_name}, beats={beats})...", flush=True)

        for idx, rid in enumerate(all_rids):
            rec = load_mitdb_record(rid, beats=beats, local_dir=self.local_dir)
            raw[rid] = extract_beat_features(rec)
            self.fs = rec.fs
            if verbose and ((idx + 1) % 11 == 0 or idx + 1 == len(all_rids)):
                print(f"  [{idx + 1}/{len(all_rids)}] records processed", flush=True)

        # Fit min-max scaler strictly on training records
        tr_feats = np.vstack([raw[r][0] for r in self.split["train"]])
        a = tr_feats.min(axis=0)
        b = tr_feats.max(axis=0)
        denom = np.where(b - a > 1e-12, b - a, 1.0)

        # Scale and clip features to [0, 1]
        self.U = {r: np.clip((raw[r][0] - a) / denom, 0.0, 1.0) for r in raw}
        self.lab = {r: raw[r][1] for r in raw}
        self.samp = {r: raw[r][2] for r in raw}

