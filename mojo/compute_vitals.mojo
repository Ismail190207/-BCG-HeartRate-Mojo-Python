"""
Compute_vitals.mojo
Slides a window over the full signal and computes BPM for each window.
"""

from std.python import Python
from std.python import PythonObject
from beat_to_beat import compute_rate


def vitals(t1: Int, t2: Int, win_size: Int, window_limit: PythonObject,
           sig: PythonObject, time: PythonObject,
           mpd: Int, plot: Int) raises -> PythonObject:
    var np       = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var all_rate = builtins.list()
    var start = t1
    var stop  = t2

    # PythonObject (math.floor result) -> Mojo Int via String conversion
    var window_limit_int: Int = atol(String(window_limit))
    for _ in range(window_limit_int):
        var sub_signal = sig[Python.evaluate("slice(" + String(start) + "," + String(stop) + ")")]
        var result     = compute_rate(sub_signal, time, mpd)
        var rate       = result[0]
        all_rate.append(rate)
        start = stop
        stop  = stop + win_size

    var all_rate_arr = np.vstack(all_rate).flatten()
    return all_rate_arr
