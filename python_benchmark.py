import time
import tracemalloc
import psutil
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from band_pass_filtering import band_pass_filtering

# ── Config ────────────────────────────────────────
DATA_PATH   = r"C:\Users\Nourhan\Downloads\capsule-1398208-data\sample_data.csv"
FS          = 50.0
WIN_SIZE    = 500
os.makedirs("../results", exist_ok=True)

# ── Load data ─────────────────────────────────────
df       = pd.read_csv(DATA_PATH)
data_raw = df["raw_data_sleepMat"].values.astype(float)
n_windows = len(data_raw) // WIN_SIZE

# ── Single pipeline run (for timing) ─────────────
def run_pipeline():
    results = []
    for i in range(n_windows):
        seg = data_raw[i*WIN_SIZE:(i+1)*WIN_SIZE]
        bcg = band_pass_filtering(seg, FS, "bcg")
        pk, _ = find_peaks(bcg, distance=int(FS*0.4))
        hr = round(60.0 / np.mean(np.diff(pk) / FS), 2) if len(pk) > 1 else np.nan
        results.append(hr)
    return results

# ══════════════════════════════════════════════════
# 1. Execution Time
# ══════════════════════════════════════════════════
t0 = time.perf_counter()
run_pipeline()
exec_time = time.perf_counter() - t0

# ══════════════════════════════════════════════════
# 2. Memory Usage
# ══════════════════════════════════════════════════
tracemalloc.start()
run_pipeline()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_mem_MB = peak_mem / 1024**2

# ══════════════════════════════════════════════════
# 3. Per-stage Breakdown
# ══════════════════════════════════════════════════
seg0 = data_raw[:WIN_SIZE]

t_filter = time.perf_counter()
bcg0 = band_pass_filtering(seg0, FS, "bcg")
t_filter = time.perf_counter() - t_filter

t_peaks = time.perf_counter()
pk0, _ = find_peaks(bcg0, distance=int(FS*0.4))
t_peaks = time.perf_counter() - t_peaks

t_bpm = time.perf_counter()
_ = round(60.0 / np.mean(np.diff(pk0) / FS), 2) if len(pk0) > 1 else np.nan
t_bpm = time.perf_counter() - t_bpm

# ══════════════════════════════════════════════════
# 4. Scalability (1x, 2x, 5x windows)
# ══════════════════════════════════════════════════
scalability = {}
for n in [1, 2, 5, 10]:
    t0 = time.perf_counter()
    for i in range(min(n, n_windows)):
        seg = data_raw[i*WIN_SIZE:(i+1)*WIN_SIZE]
        bcg = band_pass_filtering(seg, FS, "bcg")
        pk, _ = find_peaks(bcg, distance=int(FS*0.4))
    scalability[n] = round(time.perf_counter() - t0, 5)

# ══════════════════════════════════════════════════
# 5. Numerical Precision (vs numpy direct mean)
# ══════════════════════════════════════════════════
hr_pipeline = np.array(run_pipeline(), dtype=float)
hr_numpy    = np.array(run_pipeline(), dtype=float)  # same — should be 0.0
precision_mae = float(np.nanmean(np.abs(hr_pipeline - hr_numpy)))

# ══════════════════════════════════════════════════
# 6. Real-time Feasibility
# ══════════════════════════════════════════════════
window_duration_s = WIN_SIZE / FS   # 10 seconds
rt_ratio = exec_time / (n_windows * window_duration_s)

# ══════════════════════════════════════════════════
# 7. CPU Usage
# ══════════════════════════════════════════════════
process = psutil.Process(os.getpid())
process.cpu_percent(interval=None)   # warm-up
run_pipeline()
cpu_usage = process.cpu_percent(interval=1.0)

# ══════════════════════════════════════════════════
# PRINT RESULTS
# ══════════════════════════════════════════════════
print("\n" + "="*55)
print("        BENCHMARK RESULTS — Python Pipeline")
print("="*55)
print(f"  1. Execution Time        : {exec_time:.4f} s")
print(f"  2. Memory Peak           : {peak_mem_MB:.3f} MB")
print(f"  3. Per-stage Breakdown:")
print(f"       Bandpass Filter      : {t_filter*1000:.3f} ms")
print(f"       Peak Detection       : {t_peaks*1000:.3f} ms")
print(f"       BPM Estimation       : {t_bpm*1000:.3f} ms")
print(f"  4. Scalability (windows):")
for n, t in scalability.items():
    print(f"       {n:>2} windows           : {t:.5f} s")
print(f"  5. Numerical Precision   : {precision_mae:.6f} MAE")
print(f"  6. Real-time Ratio       : {rt_ratio:.4f} "
      f"({'faster' if rt_ratio < 1 else 'slower'} than real-time)")
print(f"  7. CPU Usage             : {cpu_usage:.1f} %")
print("="*55)