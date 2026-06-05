"""
Data_subplot.mojo
Normalises and plots four signals in a 3-panel figure, saved to ../results/vitals.png.
"""

from std.python import Python
from std.python import PythonObject


def normalize(sig: PythonObject) raises -> PythonObject:
    var np    = Python.import_module("numpy")
    var denom = np.sum(np.abs(sig) ** 2, axis=-1) ** (1.0 / 2.0)
    return np.divide(sig, denom)


def data_subplot(raw_data: PythonObject, movement: PythonObject,
                 breathing: PythonObject, dc: PythonObject,
                 t1: Int, t2: Int) raises:
    var matplotlib = Python.import_module("matplotlib")
    matplotlib.use("agg")
    var plt      = Python.import_module("matplotlib.pyplot")
    var np       = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var raw_n = normalize(raw_data)
    var mov_n = normalize(movement)
    var bre_n = normalize(breathing)
    var dc_n  = normalize(dc)

    var steps = np.arange(t1, t2) / 50.0
    var sl = Python.evaluate("slice(" + String(t1) + "," + String(t2) + ")")

    var fig = plt.figure()
    var ax1 = fig.add_subplot(3, 1, 1)
    var ax2 = fig.add_subplot(3, 1, 2)
    var ax3 = fig.add_subplot(3, 1, 3)

    # bbox_to_anchor=[1, 1.3] must be built as a Python list, not a Mojo list literal
    var bbox = builtins.list()
    bbox.append(1)
    bbox.append(1.3)

    ax1.plot(steps, raw_n[sl], lw=2, color="k", label="Raw Signal")
    ax1.set_xlabel("Time [Seconds]")
    ax1.set_ylabel("Amplitude")
    ax1.legend(bbox_to_anchor=bbox, loc="center right")
    plt.subplots_adjust(hspace=0.8)

    ax2.plot(steps, mov_n[sl], lw=2, color="k", label="BCG Signal")
    ax2.plot(steps, dc_n[sl],  lw=2, color="r", ls="-.", label="Level 4 Smooth")
    ax2.set_xlabel("Time [Seconds]")
    ax2.set_ylabel("Amplitude")
    ax2.legend(bbox_to_anchor=bbox, loc="center right")
    plt.subplots_adjust(hspace=0.8)

    ax3.plot(steps, bre_n[sl], lw=2, color="k", label="Respiratory Signal")
    ax3.set_xlabel("Time [Seconds]")
    ax3.set_ylabel("Amplitude")
    ax3.legend(bbox_to_anchor=bbox, loc="center right")
    plt.subplots_adjust(hspace=0.8)

    plt.savefig("/mnt/c/Users/Nourhan/Downloads/mojo/results/vitals.png")
    plt.close()
