"""
Modwt_mra_matlab_fft.mojo
Multi-Resolution Analysis from MODWT coefficients — equivalent to Matlab modwtmra().
"""

from std.python import Python
from std.python import PythonObject


def modwtmra(w_in: PythonObject, wname: String) raises -> PythonObject:
    var np       = Python.import_module("numpy")
    var pyfftw   = Python.import_module("pyfftw")
    var pywt     = Python.import_module("pywt")
    var sys      = Python.import_module("sys")
    var builtins = Python.import_module("builtins")

    var w = w_in

    var cfslength = w.shape[1]
    var J0        = w.shape[0] - 1
    var N         = cfslength
    var nullinput = np.zeros(cfslength)

    var wavelet = pywt.Wavelet(wname)
    var Lo = np.array(wavelet.rec_lo).flatten() / np.sqrt(2)
    var Hi = np.array(wavelet.rec_hi).flatten() / np.sqrt(2)

    if cfslength < len(Lo):
        var tile_shape = builtins.list()
        tile_shape.append(1)
        tile_shape.append(len(Lo) - cfslength)
        var wp = np.tile(w, tile_shape)
        w         = np.append(w, wp, axis=1)
        cfslength = w.shape[1]
        nullinput = np.zeros(cfslength)

    var fft_iface = pyfftw.interfaces.numpy_fft
    var G = fft_iface.fft(Lo, cfslength, planner_effort="FFTW_ESTIMATE", threads=1).T
    var H = fft_iface.fft(Hi, cfslength, planner_effort="FFTW_ESTIMATE", threads=1).T

    # PythonObject (w.shape[0] - 1) -> Mojo Int via String conversion
    var mra_list = builtins.list()
    var J0_int: Int = atol(String(J0))

    for level in range(J0_int, 0, -1):
        var wcfs = w[level - 1]
        var v    = nullinput
        var ww   = wcfs
        for jj in range(level, 0, -1):
            var Nv       = v.size
            var Vhat     = fft_iface.fft(v,  planner_effort="FFTW_ESTIMATE", threads=1).T
            var What     = fft_iface.fft(ww, planner_effort="FFTW_ESTIMATE", threads=1).T
            var upfactor = 2 ** (jj - 1)
            var idx      = np.mod(upfactor * np.arange(0, Nv), Nv)
            var Gup      = np.conj(G[idx])
            var Hup      = np.conj(H[idx])
            v  = fft_iface.ifft(np.multiply(Gup, Vhat) + np.multiply(Hup, What), planner_effort="FFTW_ESTIMATE", threads=1).real
            ww = nullinput
        mra_list.append(v[Python.evaluate("slice(None," + String(N) + ")")])

    # Smooth component
    var scalingcoefs = w[Python.evaluate("slice(" + String(J0_int) + ",None)")].flatten()
    var v = scalingcoefs
    for level in range(J0_int, 0, -1):
        var Nv       = v.size
        var Vhat     = fft_iface.fft(v,         planner_effort="FFTW_ESTIMATE", threads=1).T
        var What     = fft_iface.fft(nullinput,  planner_effort="FFTW_ESTIMATE", threads=1).T
        var upfactor = 2 ** (level - 1)
        var idx      = np.mod(upfactor * np.arange(0, Nv), Nv)
        var Gup      = np.conj(G[idx])
        var Hup      = np.conj(H[idx])
        v = fft_iface.ifft(np.multiply(Gup, Vhat) + np.multiply(Hup, What), planner_effort="FFTW_ESTIMATE", threads=1).real

    mra_list = builtins.list(builtins.reversed(mra_list))
    mra_list.append(v[Python.evaluate("slice(None," + String(N) + ")")])

    return np.vstack(mra_list)
