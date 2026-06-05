"""
Beat_to_beat.mojo
Computes heart / respiratory rate (BPM) from a detected-peak sequence.
"""

from std.python import Python
from std.python import PythonObject
from detect_peaks import detect_peaks


def compute_rate(beats: PythonObject, time: PythonObject, mpd: Int) raises -> PythonObject:
    var np       = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var indices = detect_peaks(
        beats,
        mph       = Python.evaluate("None"),
        mpd       = mpd,
        threshold = 0.0,
        edge      = "rising",
        kpsh      = False,
        valley    = False,
        show      = False
    )

    if len(indices) > 1:
        var peak_to_peak = np.array(builtins.list(), dtype="float64")

        # Convert PythonObject indices.size -> Mojo Int via String
        var indices_size: Int = atol(String(indices.size))
        for i in range(indices_size - 1):
            var interval = time[indices[i + 1]] - time[indices[i]]
            peak_to_peak = np.append(peak_to_peak, interval)

        var mean_interval = np.average(peak_to_peak, axis=0)
        var bpm_avg = 1000.0 * (60.0 / mean_interval)
        bpm_avg = np.round(bpm_avg, decimals=2)

        # Build Python tuple via builtins.list() to avoid Mojo List[PythonObject] type error
        var ret1 = builtins.list()
        ret1.append(bpm_avg)
        ret1.append(indices)
        return builtins.tuple(ret1)
    else:
        # Build Python tuple via builtins.list() to avoid Mojo List[Float64] type error
        var ret2 = builtins.list()
        ret2.append(0.0)
        ret2.append(0.0)
        return builtins.tuple(ret2)
