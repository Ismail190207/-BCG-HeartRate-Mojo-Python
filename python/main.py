# Import required libraries
import math
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_apnea_events import apnea_events
from detect_body_movements import detect_patterns
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot import data_subplot
# ======================================================================================================================
import time
import tracemalloc
import psutil
# Create results folder if it doesn't exist
os.makedirs('../results', exist_ok=True)

# ── Benchmark Start ──────────────────────────────
tracemalloc.start()
_t_start = time.perf_counter()
_process = psutil.Process(os.getpid())
_cpu_before = _process.cpu_percent(interval=None)
# Main program starts here

file = r"C:\Users\Nourhan\Downloads\capsule-1398208-data\sample_data.csv"

if file.endswith(".csv"):
    fileName = os.path.join(file)
    if os.stat(fileName).st_size != 0:
        rawData = pd.read_csv(fileName, sep=",", header=None, skiprows=1).values
        utc_time = rawData[:, 0]
        data_stream = rawData[:, 1]

        start_point, end_point, window_shift, fs = 0, 500, 500, 50
        # ==========================================================================================================
        data_stream, utc_time = detect_patterns(start_point, end_point, window_shift, data_stream, utc_time, plot=1)
        # ==========================================================================================================
        # BCG signal extraction
        movement = band_pass_filtering(data_stream, fs, "bcg")
        # ==========================================================================================================
        # Respiratory signal extraction
        breathing = band_pass_filtering(data_stream, fs, "breath")
        breathing = remove_nonLinear_trend(breathing, 3)
        breathing = savgol_filter(breathing, 11, 3)
        # ==========================================================================================================
        w = modwt(movement, 'bior3.9', 4)
        dc = modwtmra(w, 'bior3.9')
        wavelet_cycle = dc[4]
        # ==========================================================================================================
        # Vital Signs estimation - (10 seconds window is an optimal size for vital signs measurement)
        t1, t2, window_length, window_shift = 0, 500, 500, 500
        hop_size = math.floor((window_length - 1) / 2)
        limit = int(math.floor(breathing.size / window_shift))
        # ==========================================================================================================
        # Heart Rate
        beats = vitals(t1, t2, window_shift, limit, wavelet_cycle, utc_time, mpd=1, plot=0)
        print('\nHeart Rate Information')
        print('Minimum pulse : ', np.around(np.min(beats)))
        print('Maximum pulse : ', np.around(np.max(beats)))
        print('Average pulse : ', np.around(np.mean(beats)))
        # Breathing Rate
        beats = vitals(t1, t2, window_shift, limit, breathing, utc_time, mpd=1, plot=0)
        print('\nRespiratory Rate Information')
        print('Minimum breathing : ', np.around(np.min(beats)))
        print('Maximum breathing : ', np.around(np.max(beats)))
        print('Average breathing : ', np.around(np.mean(beats)))
        # ==============================================================================================================
        thresh = 0.3
        events = apnea_events(breathing, utc_time, thresh=thresh)
        # ==============================================================================================================
        # Plot Vitals Example
        t1, t2 = 2500, 2500 * 2
        data_subplot(data_stream, movement, breathing, wavelet_cycle, t1, t2)
        # ==============================================================================================================
    print('\nEnd processing ...')
    # ==================================================================================================================
    # ── Benchmark Results ─────────────────────────────
    _t_end = time.perf_counter()
    _current_mem, _peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _cpu_after = _process.cpu_percent(interval=None)

    _exec_time = _t_end - _t_start
    _window_dur = 10.0  # each window = 10 seconds
    _rt_ratio = _exec_time / _window_dur

    print("\n" + "=" * 45)
    print("       BENCHMARK RESULTS — Python")
    print("=" * 45)
    print(f"  Execution Time   : {_exec_time:.4f} s")
    print(f"  Memory Peak      : {_peak_mem / 1024 ** 2:.3f} MB")
    print(f"  Real-time Ratio  : {_rt_ratio:.4f}  ({'✓ faster' if _rt_ratio < 1 else '✗ slower'} than real-time)")
    print(f"  CPU Usage        : {_cpu_after:.1f} %")
    print("=" * 45)
