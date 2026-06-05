"""
evaluation_metrics.py
=====================
Computational Reproducibility Evaluation — Python Version
Nonintrusive Vital Signs Monitoring for Sleep Apnea Patients

IDENTICAL pipeline to evaluation_metrics.mojo — used for direct comparison.

Metrics:
  1. MAE   2. RMSE   3. MAPE   4. Pearson r   5. P-value
  6. Regression Plot   7. Bland-Altman Plot

Run:
    pixi run python evaluation_metrics.py
"""

import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time, os, warnings
warnings.filterwarnings("ignore")

DATA_PATH   = "/mnt/c/Users/Nourhan/Downloads/mojo/data/sample_data.csv"
RESULTS_DIR = "/mnt/c/Users/Nourhan/Downloads/mojo/results"
FS, WIN_SIZE = 50.0, 500
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Signal processing (identical to Mojo pipeline) ────────────────────────────
def band_pass_filtering(data, fs, filter_type):
    if filter_type == "bcg":
        b,a = sig.cheby1(2, 0.5, 2.5/(fs/2), btype="high")
        x   = sig.filtfilt(b, a, data)
        b,a = sig.cheby1(4, 0.5, 5.0/(fs/2), btype="low")
        return sig.filtfilt(b, a, x)
    elif filter_type == "breath":
        b,a = sig.cheby1(2, 0.5, 0.01/(fs/2), btype="high")
        x   = sig.filtfilt(b, a, data)
        b,a = sig.cheby1(4, 0.5, 0.4/(fs/2),  btype="low")
        return sig.filtfilt(b, a, x)
    return data

def remove_nonLinear_trend(x, order=3):
    idx = np.arange(len(x))
    return x - np.polyval(np.polyfit(idx, x, order), idx)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(est, ref):
    mae  = np.mean(np.abs(est - ref))
    rmse = np.sqrt(np.mean((est - ref)**2))
    mape = np.mean(np.abs((est - ref) / ref)) * 100
    r, p = stats.pearsonr(est, ref)
    return dict(mae=mae, rmse=rmse, mape=mape, pearson=r, pvalue=p)

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_regression(est, ref, m, path):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(ref, est, alpha=0.65, color="#2E86AB", edgecolors="white", linewidths=0.4, s=65,
               label=f"Windows (n={len(est)})")
    slope, intercept, *_ = stats.linregress(ref, est)
    xl = np.linspace(ref.min(), ref.max(), 200)
    ax.plot(xl, slope*xl+intercept, color="#E84855", lw=2,
            label=f"Fit: y={slope:.3f}x+{intercept:.3f}")
    lims = [min(ref.min(),est.min())-2, max(ref.max(),est.max())+2]
    ax.plot(lims, lims, "k--", lw=1.2, alpha=0.5, label="Identity (y=x)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Reference HR — PulseOximeter (BPM)", fontsize=11)
    ax.set_ylabel("Estimated HR — Python Pipeline (BPM)", fontsize=11)
    ax.set_title(f"Regression Plot  |  r={m['pearson']:.4f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[PLOT] Saved → {path}")

def plot_bland_altman(est, ref, path):
    mean_ba = (est+ref)/2; diff_ba = est-ref
    md = np.mean(diff_ba); sd = np.std(diff_ba, ddof=1)
    loa_hi = md+1.96*sd;   loa_lo = md-1.96*sd
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(mean_ba, diff_ba, alpha=0.65, color="#2E86AB", edgecolors="white", linewidths=0.4, s=65)
    ax.axhline(md,     color="#E84855", lw=2,   ls="-",  label=f"Mean diff: {md:.2f} BPM")
    ax.axhline(loa_hi, color="#F4A261", lw=1.5, ls="--", label=f"+1.96 SD: {loa_hi:.2f}")
    ax.axhline(loa_lo, color="#F4A261", lw=1.5, ls="--", label=f"−1.96 SD: {loa_lo:.2f}")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Mean of Estimated & Reference HR (BPM)", fontsize=11)
    ax.set_ylabel("Difference: Estimated − Reference (BPM)", fontsize=11)
    ax.set_title(f"Bland–Altman Plot  |  Mean diff={md:.2f}  LoA=[{loa_lo:.2f}, {loa_hi:.2f}]", fontsize=11)
    ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[PLOT] Saved → {path}")
    return md, loa_lo, loa_hi

def plot_window_comparison(est, ref, m, path):
    win_idx = np.arange(len(est))
    abs_err = np.abs(est-ref); pct_err = np.abs((est-ref)/ref)*100
    fig, axes = plt.subplots(3,1,figsize=(14,10),sharex=True)
    fig.suptitle("Per-Window Comparison: Estimated vs Reference HR (Python)", fontsize=13, fontweight="bold")
    axes[0].plot(win_idx, ref, "o-", color="#2E86AB", lw=1.5, ms=4, label="Reference (PulseOximeter)")
    axes[0].plot(win_idx, est, "s--", color="#E84855", lw=1.5, ms=4, label="Estimated (Python)")
    axes[0].set_ylabel("HR (BPM)", fontsize=10); axes[0].legend(fontsize=9); axes[0].grid(True,alpha=0.3)
    axes[1].bar(win_idx, abs_err, color="#A8DADC", edgecolor="steelblue", lw=0.5)
    axes[1].axhline(m["mae"], color="#E84855", lw=1.5, ls="--", label=f"MAE={m['mae']:.2f} BPM")
    axes[1].set_ylabel("|Error| (BPM)", fontsize=10); axes[1].legend(fontsize=9); axes[1].grid(True,alpha=0.3)
    axes[2].bar(win_idx, pct_err, color="#FFB347", edgecolor="darkorange", lw=0.5)
    axes[2].axhline(m["mape"], color="#E84855", lw=1.5, ls="--", label=f"MAPE={m['mape']:.2f}%")
    axes[2].set_xlabel("Window Index", fontsize=10); axes[2].set_ylabel("% Error", fontsize=10)
    axes[2].legend(fontsize=9); axes[2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[PLOT] Saved → {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  EVALUATION METRICS — PYTHON VERSION")
    print("  Computational Reproducibility Assessment")
    print("="*60)
    t_start = time.time()

    df        = pd.read_csv(DATA_PATH)
    data_raw  = df["raw_data_sleepMat"].values.astype(float)
    ref_hr_ts = df["PulseOximeter_results"].values.astype(float)
    n_windows = len(data_raw) // WIN_SIZE
    print(f"\n[DATA] Samples: {len(data_raw):,}  |  Windows: {n_windows}  |  FS: {FS} Hz")

    print("\n[PIPELINE] Processing windows...")
    t_pipe = time.time()
    est_hr, ref_hr_w, est_rr = [], [], []

    for i in range(n_windows):
        seg     = data_raw[i*WIN_SIZE:(i+1)*WIN_SIZE]
        ref_seg = ref_hr_ts[i*WIN_SIZE:(i+1)*WIN_SIZE]
        # HR
        bcg = band_pass_filtering(seg, FS, "bcg")
        pk, _ = sig.find_peaks(bcg, distance=int(FS*0.4))
        est_hr.append(round(60.0/np.mean(np.diff(pk)/FS), 2) if len(pk)>1 else np.nan)
        # RR
        br = sig.savgol_filter(remove_nonLinear_trend(band_pass_filtering(seg,FS,"breath"),3),11,3)
        rp, _ = sig.find_peaks(br, distance=int(FS*1.5))
        est_rr.append(round(60.0/np.mean(np.diff(rp)/FS), 2) if len(rp)>1 else np.nan)
        ref_hr_w.append(np.mean(ref_seg))

    t_pipe_end = time.time(); pipe_time = t_pipe_end - t_pipe
    print(f"[PIPELINE] Done in {pipe_time:.3f} s")

    est_hr = np.array(est_hr); ref_hr_w = np.array(ref_hr_w); est_rr = np.array(est_rr)
    valid  = np.isfinite(est_hr)
    est    = est_hr[valid]; ref = ref_hr_w[valid]; rr = est_rr[np.isfinite(est_rr)]

    print(f"[PIPELINE] Valid HR: {valid.sum()}/{n_windows}  |  Valid RR: {np.isfinite(est_rr).sum()}/{n_windows}")

    m = compute_metrics(est, ref)
    diff_ba = est-ref; md=np.mean(diff_ba); sd=np.std(diff_ba,ddof=1)
    loa_hi=md+1.96*sd; loa_lo=md-1.96*sd

    print("\n"+"="*60)
    print("  HEART RATE METRICS vs PulseOximeter Reference")
    print("="*60)
    print(f"  Windows evaluated   : {valid.sum()}")
    print(f"  MAE   (BPM)         : {m['mae']:.4f}")
    print(f"  RMSE  (BPM)         : {m['rmse']:.4f}")
    print(f"  MAPE  (%)           : {m['mape']:.4f}")
    print(f"  Pearson r           : {m['pearson']:.4f}")
    print(f"  P-value             : {m['pvalue']:.6f}")
    print(f"  Correlation         : {'Significant (p<0.05) ✓' if m['pvalue']<0.05 else 'Not significant'}")
    print(f"  Bland-Altman bias   : {md:.4f}")
    print(f"  95% LoA             : [{loa_lo:.2f}, {loa_hi:.2f}]")
    print("\n  RESPIRATORY RATE (descriptive)")
    print(f"  Mean={np.nanmean(rr):.2f}  Std={np.nanstd(rr):.2f}  Min={np.nanmin(rr):.2f}  Max={np.nanmax(rr):.2f}")

    # CSV
    pd.DataFrame({
        "Metric": ["MAE (BPM)","RMSE (BPM)","MAPE (%)","Pearson r","P-value",
                   "BA Bias (BPM)","LoA Lower","LoA Upper","RR Mean","RR Std"],
        "Value":  [f"{m['mae']:.4f}",f"{m['rmse']:.4f}",f"{m['mape']:.4f}",
                   f"{m['pearson']:.4f}",f"{m['pvalue']:.6f}",f"{md:.4f}",
                   f"{loa_lo:.2f}",f"{loa_hi:.2f}",
                   f"{np.nanmean(rr):.2f}",f"{np.nanstd(rr):.2f}"]
    }).to_csv(f"{RESULTS_DIR}/metrics_summary_python.csv", index=False)
    print(f"\n[CSV]  Saved → {RESULTS_DIR}/metrics_summary_python.csv")

    # Plots
    plot_regression(est, ref, m, f"{RESULTS_DIR}/regression_plot_python.png")
    plot_bland_altman(est, ref,   f"{RESULTS_DIR}/bland_altman_python.png")
    plot_window_comparison(est, ref, m, f"{RESULTS_DIR}/window_comparison_python.png")

    t_total = time.time()-t_start
    print("\n"+"="*60)
    print("  COMPUTATIONAL EFFICIENCY")
    print("="*60)
    print(f"  Language              : Python 3")
    print(f"  Total time            : {t_total:.3f} s")
    print(f"  Pipeline time         : {pipe_time:.3f} s")
    print(f"  Per-window time       : {pipe_time/n_windows*1000:.2f} ms")
    print("="*60)
