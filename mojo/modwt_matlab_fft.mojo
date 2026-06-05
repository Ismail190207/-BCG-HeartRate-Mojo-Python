"""
Modwt_matlab_fft.mojo
Maximal Overlap Discrete Wavelet Transform — equivalent to Matlab modwt().
"""

from std.python import Python
from std.python import PythonObject


def modwt(x_in: PythonObject, wname: String, J: Int) raises -> PythonObject:
    var np       = Python.import_module("numpy")
    var math     = Python.import_module("math")
    var pyfftw   = Python.import_module("pyfftw")
    var pywt     = Python.import_module("pywt")
    var builtins = Python.import_module("builtins")

    var x = x_in.flatten()
    var siglen = x.size
    var Nrep   = siglen

    var Jmax = np.floor(math.log(siglen, 2))
    if J <= 0:
        print("Wavelet:modwt:MRALevel — J must be > 0")
        var sys = Python.import_module("sys")
        sys.exit()

    var wavelet = pywt.Wavelet(wname)
    var Lo = np.array(wavelet.rec_lo).flatten() / np.sqrt(2)
    var Hi = np.array(wavelet.rec_hi).flatten() / np.sqrt(2)

    if siglen < len(Lo):
        var tile_dims = builtins.list()
        tile_dims.append(1)
        tile_dims.append(len(Lo) - siglen)
        var xp = np.tile(x, tile_dims)
        x    = np.append(x, xp)
        Nrep = x.size

    var fft_iface = pyfftw.interfaces.numpy_fft
    var G    = fft_iface.fft(Lo, Nrep, planner_effort="FFTW_ESTIMATE", threads=1).T
    var H    = fft_iface.fft(Hi, Nrep, planner_effort="FFTW_ESTIMATE", threads=1).T
    var Vhat = fft_iface.fft(x,  planner_effort="FFTW_ESTIMATE").T

    var w_list = builtins.list()

    for jj in range(J):
        var N        = Vhat.size
        var upfactor = 2 ** jj
        var idx      = np.mod(upfactor * np.arange(0, N), N)
        var Gup      = G[idx]
        var Hup      = H[idx]
        var new_Vhat = np.multiply(Gup, Vhat)
        var What     = np.multiply(Hup, Vhat)
        Vhat = new_Vhat
        w_list.append(fft_iface.ifft(What, planner_effort="FFTW_ESTIMATE", threads=1).real)

    w_list.append(fft_iface.ifft(Vhat, planner_effort="FFTW_ESTIMATE", threads=1).real)

    var w = np.vstack(w_list)
    w = w[Python.evaluate("slice(None)"), Python.evaluate("slice(None," + String(siglen) + ")")]
    return w
