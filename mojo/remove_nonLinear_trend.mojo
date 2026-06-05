"""
remove_nonLinear_trend.mojo
Remove nonlinear trend from a signal using polynomial detrending.
"""

from std.python import Python
from std.python import PythonObject


def remove_nonLinear_trend(input_signal: PythonObject, order: Int) raises -> PythonObject:
    var np = Python.import_module("numpy")

    var indices   = np.arange(0, input_signal.size)
    var model     = np.polyfit(indices, input_signal, order)
    var predicted = np.polyval(model, indices)
    var filteredSignal = input_signal - predicted

    return filteredSignal
