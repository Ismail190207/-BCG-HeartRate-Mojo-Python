"""
band_pass_filtering.mojo
Chebyshev Type I bandpass filter for BCG (heart) and breathing signals.
"""

from std.python import Python
from std.python import PythonObject


def band_pass_filtering(data: PythonObject, fs: Float64, filter_type: String) raises -> PythonObject:
    var scipy_signal = Python.import_module("scipy.signal")

    var filtered_data: PythonObject

    if filter_type == "bcg":
        var hp = scipy_signal.cheby1(2, 0.5, 2.5 / (fs / 2.0), btype="high", analog=False)
        var b_high = hp[0]
        var a_high = hp[1]
        var bcg_ = scipy_signal.filtfilt(b_high, a_high, data)

        var lp = scipy_signal.cheby1(4, 0.5, 5.0 / (fs / 2.0), btype="low", analog=False)
        var b_low = lp[0]
        var a_low = lp[1]
        filtered_data = scipy_signal.filtfilt(b_low, a_low, bcg_)

    elif filter_type == "breath":
        var hp = scipy_signal.cheby1(2, 0.5, 0.01 / (fs / 2.0), btype="high", analog=False)
        var b_high = hp[0]
        var a_high = hp[1]
        var bcg_ = scipy_signal.filtfilt(b_high, a_high, data)

        var lp = scipy_signal.cheby1(4, 0.5, 0.4 / (fs / 2.0), btype="low", analog=False)
        var b_low = lp[0]
        var a_low = lp[1]
        filtered_data = scipy_signal.filtfilt(b_low, a_low, bcg_)

    else:
        filtered_data = data

    return filtered_data
