# BCG Heart Rate Detection — Mojo Implementation

Complete conversion of the original Python BCG pipeline to Mojo,
preserving all logic and producing identical output.

## How to Run

```bash
export MOJO_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so.1
pixi run mojo main.mojo
```

## Dependencies

```bash
pip install numpy pandas scipy PyWavelets pyfftw matplotlib pytz
```

## Authors
Ismail E. Mohamed & Ibrahim Sadek — Helwan University  
Advanced Biostatistics — Dr. Ibrahim Sadek
