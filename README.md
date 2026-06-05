# BCG Heart Rate Detection — Mojo vs. Python
**Advanced Biostatistics PhD Coursework — Dr. Ibrahim Sadek**

This project implements and compares two identical BCG (Ballistocardiography) 
heart rate detection pipelines in Mojo and Python. Both follow the same 
five-stage workflow: movement removal, Chebyshev bandpass filtering, 
MODWT decomposition, peak detection, and BPM estimation.

Both pipelines produce identical accuracy results (MAE = 2.75 BPM, 
r = 0.6843), while Mojo achieves ~700× faster execution and 262× 
lower memory usage than Python.

## How to Run

**Python:**
```bash
cd python/
python main.py
```

**Mojo:**
```bash
cd mojo/
export MOJO_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so.1
pixi run mojo main.mojo
```

## Authors
Ismail E. Mohamed & Ibrahim Sadek — Helwan University
