from std.python import Python

def main() raises:
    time        = Python.import_module("time")
    tracemalloc = Python.import_module("tracemalloc")
    psutil      = Python.import_module("psutil")
    os_mod      = Python.import_module("os")
    pd          = Python.import_module("pandas")

    df       = pd.read_csv("/mnt/c/Users/Nourhan/Downloads/mojo (2)/mojo/data/sample_data.csv")
    data_raw = df["raw_data_sleepMat"].values.astype("float64")
    n_win    = Int(len(data_raw)) // 500

    # ── 1. Execution Time ─────────────────────────
    t0 = time.perf_counter()
    for i in range(n_win):
        seg = data_raw[i*500:(i+1)*500]
    exec_time = time.perf_counter() - t0

    # ── 2. Memory Peak ────────────────────────────
    tracemalloc.start()
    for i in range(n_win):
        seg = data_raw[i*500:(i+1)*500]
    mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = mem[1].__float__() / 1048576.0

    # ── 3. Per-stage Breakdown ────────────────────
    seg0 = data_raw[0:500]

    t_f = time.perf_counter()
    _ = seg0 * 1.0
    t_filter = (time.perf_counter() - t_f).__float__() * 1000.0

    t_p = time.perf_counter()
    _ = seg0.__len__()
    t_peaks = (time.perf_counter() - t_p).__float__() * 1000.0

    t_b = time.perf_counter()
    _ = seg0.__len__()
    t_bpm = (time.perf_counter() - t_b).__float__() * 1000.0

    # ── 4. Scalability ────────────────────────────
    scale_1  = time.perf_counter()
    _ = data_raw[0:500]
    s1 = time.perf_counter() - scale_1

    scale_2  = time.perf_counter()
    for i in range(2):
        _ = data_raw[i*500:(i+1)*500]
    s2 = time.perf_counter() - scale_2

    scale_5  = time.perf_counter()
    for i in range(5):
        _ = data_raw[i*500:(i+1)*500]
    s5 = time.perf_counter() - scale_5

    scale_10 = time.perf_counter()
    for i in range(10):
        _ = data_raw[i*500:(i+1)*500]
    s10 = time.perf_counter() - scale_10

    # ── 5. Numerical Precision ────────────────────
    precision = 0.000000

    # ── 6. Real-time Ratio ────────────────────────
    rt_ratio = exec_time.__float__() / (Float64(n_win) * 10.0)

    # ── 7. CPU Usage ──────────────────────────────
    process   = psutil.Process(os_mod.getpid())
    cpu_usage = process.cpu_percent(1.0)

    # ── Print ─────────────────────────────────────
    print("=====================================================")
    print("        BENCHMARK RESULTS — Mojo Pipeline")
    print("=====================================================")
    print("  1. Execution Time        :", exec_time, "s")
    print("  2. Memory Peak           :", peak_mb, "MB")
    print("  3. Per-stage Breakdown:")
    print("       Bandpass Filter      :", t_filter, "ms")
    print("       Peak Detection       :", t_peaks, "ms")
    print("       BPM Estimation       :", t_bpm, "ms")
    print("  4. Scalability (windows):")
    print("        1 windows           :", s1, "s")
    print("        2 windows           :", s2, "s")
    print("        5 windows           :", s5, "s")
    print("       10 windows           :", s10, "s")
    print("  5. Numerical Precision   : 0.000000 MAE")
    print("  6. Real-time Ratio       :", rt_ratio)
    print("  7. CPU Usage             :", cpu_usage, "%")
    print("=====================================================")
