"""
Detect_peaks.mojo
Original author: Marcos Duarte. Detects peaks in a 1-D signal.
"""

from std.python import Python
from std.python import PythonObject


def detect_peaks(x_in: PythonObject, mph: PythonObject, mpd: Int,
                 threshold: Float64, edge: String,
                 kpsh: Bool, valley: Bool, show: Bool) raises -> PythonObject:
    var np       = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    # Function parameters are immutable in Mojo — use a local mutable copy
    var x = np.atleast_1d(x_in).astype("float64")

    if x.size < 3:
        return np.array(builtins.list(), dtype="int")

    if valley:
        x = -x

    var dx = x[Python.evaluate("slice(1,None)")] - x[Python.evaluate("slice(None,-1)")]

    var indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan]                      = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    var empty = np.array(builtins.list(), dtype="int")
    var ine   = empty
    var ire   = empty
    var ife   = empty

    if edge == "":
        # np.hstack([dx, 0]) — build via builtins.list()
        var h1a = builtins.list()
        h1a.append(dx)
        h1a.append(0)
        var h1b = builtins.list()
        h1b.append(0)
        h1b.append(dx)
        ine = np.where(
            (np.hstack(h1a) < 0) & (np.hstack(h1b) > 0)
        )[0]
    else:
        var edge_lower = edge.lower()
        if edge_lower == "rising" or edge_lower == "both":
            var h2a = builtins.list()
            h2a.append(dx)
            h2a.append(0)
            var h2b = builtins.list()
            h2b.append(0)
            h2b.append(dx)
            ire = np.where(
                (np.hstack(h2a) <= 0) & (np.hstack(h2b) > 0)
            )[0]
        if edge_lower == "falling" or edge_lower == "both":
            var h3a = builtins.list()
            h3a.append(dx)
            h3a.append(0)
            var h3b = builtins.list()
            h3b.append(0)
            h3b.append(dx)
            ife = np.where(
                (np.hstack(h3a) < 0) & (np.hstack(h3b) >= 0)
            )[0]

    # np.hstack([ine, ire, ife]) — build via builtins.list()
    var hstack_ind = builtins.list()
    hstack_ind.append(ine)
    hstack_ind.append(ire)
    hstack_ind.append(ife)
    var ind = np.unique(np.hstack(hstack_ind))

    if ind.size > 0 and indnan.size > 0:
        # np.hstack([indnan, indnan-1, indnan+1]) — build via builtins.list()
        var hstack_nan = builtins.list()
        hstack_nan.append(indnan)
        hstack_nan.append(indnan - 1)
        hstack_nan.append(indnan + 1)
        var nan_neighbors = np.unique(np.hstack(hstack_nan))
        ind = ind[np.in1d(ind, nan_neighbors, invert=True)]

    if ind.size > 0 and ind[0] == 0:
        ind = ind[Python.evaluate("slice(1,None)")]
    if ind.size > 0 and ind[-1] == x.size - 1:
        ind = ind[Python.evaluate("slice(None,-1)")]

    var is_none = (mph == Python.evaluate("None"))
    if ind.size > 0 and not is_none:
        ind = ind[x[ind] >= mph]

    if ind.size > 0 and threshold > 0:
        # np.vstack([a, b]) — build via builtins.list()
        var vstack_dx = builtins.list()
        vstack_dx.append(x[ind] - x[ind - 1])
        vstack_dx.append(x[ind] - x[ind + 1])
        var dx2 = np.min(np.vstack(vstack_dx), axis=0)
        ind = np.delete(ind, np.where(dx2 < threshold)[0])

    if ind.size > 0 and mpd > 1:
        ind = ind[np.argsort(x[ind])]
        ind = ind[Python.evaluate("slice(None,None,-1)")]

        var idel = np.zeros(ind.size, dtype="bool")

        # Convert PythonObject ind.size -> Mojo Int via String
        var ind_size: Int = atol(String(ind.size))
        for i in range(ind_size):
            if not idel[i]:
                var mask: PythonObject
                if kpsh:
                    mask = (
                        (ind >= ind[i] - mpd) &
                        (ind <= ind[i] + mpd) &
                        (x[ind[i]] > x[ind])
                    )
                else:
                    mask = (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd)
                idel = idel | mask
                idel[i] = False

        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            _ = -x   # restore sign for display only; value not used further

    return ind


def _plot_peaks(x_in: PythonObject, mph: PythonObject, mpd: Int,
                threshold: Float64, edge: String,
                valley: Bool, ind: PythonObject) raises:
    var np  = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")
    var builtins = Python.import_module("builtins")

    # Local mutable copy of immutable parameter
    var x = x_in

    # figsize=[8,4] — build via builtins.list()
    var figsize = builtins.list()
    figsize.append(8)
    figsize.append(4)
    var fig_ax = plt.subplots(1, 1, figsize=figsize)
    var ax     = fig_ax[1]

    ax.plot(x, "b", lw=1)

    if ind.size > 0:
        var label: String
        if valley:
            label = "valley"
        else:
            label = "peak"
        # Convert PythonObject ind.size -> Mojo Int via String
        var ind_size: Int = atol(String(ind.size))
        if ind_size > 1:
            label = label + "s"

        ax.plot(ind, x[ind], "+", mfc=Python.evaluate("None"),
                mec="r", mew=2, ms=8,
                label=String(ind_size) + " " + label)
        ax.legend(loc="best", framealpha=0.5, numpoints=1)

    ax.set_xlim(-0.02 * x.size, x.size * 1.02 - 1)

    var finite_x = x[np.isfinite(x)]
    var ymin = finite_x.min()
    var ymax = finite_x.max()
    var yrange = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    ax.set_xlabel("Data #", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)

    var mode = "Valley detection" if valley else "Peak detection"
    ax.set_title(mode)

    plt.show()
