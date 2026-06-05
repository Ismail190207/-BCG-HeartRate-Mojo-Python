# BCG Heart Rate Detection — Mojo vs. Python
### Nonintrusive Vital Signs Monitoring | Advanced Biostatistics — Dr. Ibrahim Sadek

> Cross-language comparison of BCG-based heart rate estimation pipelines implemented in **Mojo** and **Python**.

---

## Overview

This repository implements and compares two identical BCG (Ballistocardiography) signal processing pipelines for non-contact heart rate (HR) and respiratory rate (RR) estimation from a mattress-embedded fiber optic sensor.

| Language | MAE (BPM) | RMSE (BPM) | Pearson r | Exec. Time (s) | Memory (MB) |
|----------|-----------|------------|-----------|----------------|-------------|
| **Python** | 2.75 | 3.58 | 0.6843 | 0.0863 | 0.038 |
| **Mojo** | 2.75 | 3.58 | 0.6843 | 0.000123 | 0.000145 |

✅ **Identical accuracy** — Mojo is **~700× faster** and uses **262× less memory**

---

## Repository Structure

```
BCG-HeartRate-Mojo-Python/
│
├── python/                          # Python implementation
│   ├── main.py                      # Entry point
│   ├── band_pass_filtering.py       # Chebyshev I bandpass filter  [MANUAL]
│   ├── beat_to_beat.py              # Peak-to-peak BPM calculator  [MANUAL]
│   ├── compute_vitals.py            # Sliding-window vital estimator
│   ├── data_subplot.py              # Signal visualization
│   ├── detect_apnea_events.py       # Apnea event detector         [MANUAL]
│   ├── detect_body_movements.py     # Movement classifier          [MANUAL]
│   ├── detect_peaks.py              # Peak detector (Marcos Duarte)[MANUAL]
│   ├── modwt_matlab_fft.py          # MODWT (≡ MATLAB modwt)       [MANUAL]
│   ├── modwt_mra_matlab_fft.py      # MODWT-MRA                    [MANUAL]
│   ├── remove_nonLinear_trend.py    # Polynomial detrending        [MANUAL]
│   └── python_benchmark.py          # Performance benchmark
│
├── mojo/                            # Mojo implementation
│   ├── main.mojo                    # Entry point
│   ├── band_pass_filtering.mojo     # Chebyshev I bandpass filter  [MANUAL]
│   ├── beat_to_beat.mojo            # Peak-to-peak BPM calculator  [MANUAL]
│   ├── compute_vitals.mojo          # Sliding-window vital estimator
│   ├── data_subplot.mojo            # Signal visualization
│   ├── detect_apnea_events.mojo     # Apnea event detector         [MANUAL]
│   ├── detect_body_movements.mojo   # Movement classifier          [MANUAL]
│   ├── detect_peaks.mojo            # Peak detector (Marcos Duarte)[MANUAL]
│   ├── modwt_matlab_fft.mojo        # MODWT (≡ MATLAB modwt)       [MANUAL]
│   ├── modwt_mra_matlab_fft.mojo    # MODWT-MRA                    [MANUAL]
│   ├── remove_nonLinear_trend.mojo  # Polynomial detrending        [MANUAL]
│   ├── mojo_benchpress.mojo         # Performance benchmark
│   └── pixi.toml                    # Mojo environment config
│
└── README.md
```

> `[MANUAL]` = core algorithm implemented from scratch without black-box functions

---

## Signal Processing Pipeline

```
Raw CSV (timestamp_utc_ms, sensor_amplitude_mV)
        │
        ▼
1. detect_body_movements     → remove bed-empty & movement windows  [MANUAL: SD + MAD]
        │
        ▼
2. band_pass_filtering       → Chebyshev I cascade                 [MANUAL: filtfilt]
   ├── BCG:    HP @ 2.5 Hz + LP @ 5.0 Hz
   └── Breath: HP @ 0.01 Hz + LP @ 0.4 Hz
        │
        ▼
3. modwt + modwtmra          → FFT-based MODWT (bior3.9, J=4)      [MANUAL: ≡ MATLAB]
   └── Level-4 smooth = cardiac wavelet_cycle
        │
        ▼
4. detect_peaks              → Marcos Duarte algorithm             [MANUAL: no scipy]
        │
        ▼
5. beat_to_beat → compute_vitals → HR & RR per window             [MANUAL: mean IPI]
        │
        ▼
6. detect_apnea_events       → SD-threshold apnea detection        [MANUAL]
```

---

## Where Manual Implementations Replace Black-Box Functions

| Module | Replaced | How |
|--------|----------|-----|
| `detect_peaks` | `scipy.signal.find_peaks` | Full Marcos Duarte algorithm |
| `modwt_matlab_fft` | `pywt.swt` | FFT-based MODWT (≡ MATLAB `modwt`) |
| `band_pass_filtering` | Direct `scipy.signal` pipeline | Manual Chebyshev cascade |
| `beat_to_beat` | Any HR library | Manual mean inter-peak interval |
| `detect_body_movements` | `scipy.stats.median_abs_deviation` | Manual MAD |
| `detect_apnea_events` | Any apnea library | Manual SD thresholding |
| Evaluation metrics | `sklearn.metrics` | Manual MAE, RMSE, MAPE, Pearson r |
| Bland-Altman | `pingouin` | Manual bias ± 1.96×SD |

---

## Dependencies

### Python

```bash
pip install numpy pandas scipy PyWavelets pyfftw matplotlib pytz psutil
```

### Mojo

```bash
# Install Mojo via pixi
curl -fsSL https://pixi.sh/install.sh | bash
cd mojo/
pixi install
```

Install Python packages for Mojo interop (WSL/Linux):

```bash
pip3 install numpy pandas scipy PyWavelets pyfftw matplotlib pytz psutil --break-system-packages
export MOJO_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so.1
```

---

## How to Run

### Python

```bash
cd python/
python main.py
```

### Mojo (WSL/Linux)

```bash
cd mojo/
export MOJO_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so.1
pixi run mojo main.mojo
```

### Benchmark

```bash
# Python benchmark
cd python/
python python_benchmark.py

# Mojo benchmark
cd mojo/
pixi run mojo mojo_benchpress.mojo
```

**CSV format:**
```
timestamp_utc_ms,sensor_amplitude
1234567890000,10234.5
1234567890020,10241.2
```

---

## Results

### Accuracy Metrics (n = 111 windows)

| Metric | Python | Mojo |
|--------|--------|------|
| MAE (BPM) | 2.75 | 2.75 |
| RMSE (BPM) | 3.58 | 3.58 |
| MAPE (%) | 2.71 | 2.71 |
| Pearson r | 0.6843 | 0.6843 |
| P-value | < 0.001 | < 0.001 |
| BA Bias (BPM) | 0.80 | 0.80 |
| LoA (BPM) | [−6.08, 7.67] | [−6.08, 7.67] |

### Computational Benchmark

| Metric | Python | Mojo | Speedup |
|--------|--------|------|---------|
| Execution Time (s) | 0.0863 | 0.000123 | **~700×** |
| Memory Peak (MB) | 0.038 | 0.000145 | **262×** |
| Bandpass Filter (ms) | 0.678 | 0.0345 | **20×** |
| Peak Detection (ms) | 0.036 | 0.00154 | **23×** |
| BPM Estimation (ms) | 0.029 | 0.00113 | **26×** |
| Real-time Ratio | 0.0001 | 1.11×10⁻⁷ | ✅ Both |

---

## Python vs. Mojo — Key Differences

| Aspect | Python | Mojo |
|--------|--------|------|
| Type system | Dynamic | Static (`Float64`, `Int`, `PythonObject`) |
| Variables | Implicit | `var` keyword required |
| Slicing | `arr[a:b]` | `Python.evaluate("slice(a,b)")` |
| None | `None` | `Python.evaluate("None")` |
| List building | `[]` | `builtins.list()` via interop |
| Numerical output | **Identical** | **Identical** |

---

## Reference

- Dataset & pipeline: [doi:10.1038/s41597-024-03950-5](https://doi.org/10.1038/s41597-024-03950-5)
- Original CodeOcean: [codeocean.com/capsule/1398208](https://codeocean.com/capsule/1398208/tree)
- Mojo docs: [docs.modular.com/mojo](https://docs.modular.com/mojo/)

---

## Authors

**Author A** — Mojo Implementation  
**Author B** — Python Implementation  

*PhD Coursework · Advanced Biostatistics · Dr. Ibrahim Sadek*
