# Nonintrusive Vital Signs Monitoring for Sleep Apnea Patients
## Mojo Conversion

This is a complete conversion of the original Python project to **Mojo**,
preserving every function, all logic, and identical output.

---

## Objective

Heart and respiratory rate measurement for signals acquired from a
**microbend fiber optic sensor** placed under a subject's mattress
(approximately below chest and stomach).  
The sensor captures mechanical cardiac activity and chest/stomach movement.

---

## Project Structure

```
sleep_apnea_mojo/
├── code/
│   ├── main.mojo                   ← Entry point (run this)
│   ├── band_pass_filtering.mojo    ← Chebyshev I bandpass filter (BCG & breath)
│   ├── beat_to_beat.mojo           ← Peak-to-peak BPM calculator
│   ├── compute_vitals.mojo         ← Sliding-window vital-sign estimator
│   ├── data_subplot.mojo           ← Signal visualisation → vitals.png
│   ├── detect_apnea_events.mojo    ← Apnea event detector (SD thresholding)
│   ├── detect_body_movements.mojo  ← Body-movement / bed-empty classifier
│   ├── detect_peaks.mojo           ← General peak detector (Marcos Duarte)
│   ├── modwt_matlab_fft.mojo       ← MODWT (≡ Matlab modwt)
│   ├── modwt_mra_matlab_fft.mojo   ← MODWT-MRA (≡ Matlab modwtmra)
│   └── remove_nonLinear_trend.mojo ← Polynomial detrending
├── data/
│   └── sample_data.csv             ← Place your CSV data file here
└── results/
    ├── rawData.png                 ← Auto-generated: movement classification
    └── vitals.png                  ← Auto-generated: signal panels
```

---

## How to Run

```bash
cd code
mojo main.mojo
```

Place `sample_data.csv` in the `../data/` directory before running.

**CSV format** (comma-separated, one header row):
```
timestamp_utc_ms, sensor_amplitude
1234567890000, 0.452
...
```

---

## Dependencies

Mojo interoperates with Python, so the same Python packages are required:

| Package      | Purpose                                  |
|--------------|------------------------------------------|
| `numpy`      | Array maths                              |
| `pandas`     | CSV loading, timestamp conversion        |
| `scipy`      | Chebyshev filter, Savitzky–Golay filter  |
| `pywt`       | Wavelet filter coefficients              |
| `pyfftw`     | Fast FFT for MODWT                       |
| `matplotlib` | Signal plots                             |
| `pytz`       | Timezone conversion (Asia/Singapore)     |

Install with:
```bash
pip install numpy pandas scipy PyWavelets pyfftw matplotlib pytz
```

---

## Pipeline

```
CSV data
   │
   ▼
detect_body_movements   →  remove movement/empty windows
   │
   ├──► band_pass_filtering("bcg")     →  BCG (heart) signal
   │        └─► modwt + modwtmra       →  wavelet_cycle (level-4 smooth)
   │
   └──► band_pass_filtering("breath")  →  respiratory signal
            └─► remove_nonLinear_trend + savgol_filter
   │
   ├──► compute_vitals (heart)   →  BPM per window  →  print stats
   ├──► compute_vitals (breath)  →  BPM per window  →  print stats
   ├──► detect_apnea_events      →  start/stop times of apnea events
   └──► data_subplot             →  save vitals.png
```

---

## Python vs Mojo — Key Differences

| Aspect              | Python original          | Mojo conversion                          |
|---------------------|--------------------------|------------------------------------------|
| Type system         | Dynamic                  | Static (`Int`, `Float64`, `String`, …)   |
| Variables           | Implicit                 | `var` keyword required                   |
| Python interop      | Native                   | `from python import Python` + `PythonObject` |
| Closures            | `def` inside `def`       | Nested `fn` inside `fn`                  |
| Slicing             | `arr[a:b]`               | `arr[Python.evaluate("slice(a,b)")]`     |
| None                | `None`                   | `Python.evaluate("None")`                |
| List building       | `[]`                     | `builtins.list()` via Python interop     |
| Numerical output    | **Identical** — all heavy computation delegated to numpy/scipy/pyfftw |

---

## Output

```
start processing ...

Heart Rate Information
Minimum pulse :  55.0
Maximum pulse :  82.0
Average pulse :  68.0

Respiratory Rate Information
Minimum breathing :  12.0
Maximum breathing :  20.0
Average breathing :  16.0

Apnea Information
start time : ['02.14.35']  stop time : ['02.14.45']

End processing ...
```

Plots saved to `../results/rawData.png` and `../results/vitals.png`.
