"""
Main.mojo
=========
Nonintrusive Vital Signs Monitoring for Sleep Apnea Patients
Converted from Python to Mojo — identical pipeline, identical output.

Run:
    mojo main.mojo

Data:
    Place sample_data.csv in the ../data/ directory.
    CSV format: first column = UTC timestamp (ms), second = sensor amplitude.
"""

from std.python import Python
from std.python import PythonObject

# ── Import all converted modules ──────────────────────────────────────────────
from band_pass_filtering    import band_pass_filtering
from compute_vitals         import vitals
from detect_apnea_events    import apnea_events
from detect_body_movements  import detect_patterns
from modwt_matlab_fft       import modwt
from modwt_mra_matlab_fft   import modwtmra
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot           import data_subplot


def main() raises:
    # ── Python stdlib & third-party imports needed in main ────────────────
    var np       = Python.import_module("numpy")
    var pd       = Python.import_module("pandas")
    var math     = Python.import_module("math")
    var os       = Python.import_module("os")
    var builtins = Python.import_module("builtins")
    var scipy_signal = Python.import_module("scipy.signal")

    print("\nstart processing ...")

    var file = "/mnt/c/Users/Nourhan/Downloads/mojo/data/sample_data.csv"

    # ── Load CSV ──────────────────────────────────────────────────────────
    if file.endswith(".csv"):
        var fileName = os.path.join(file)

        if os.stat(fileName).st_size != 0:
            var rawData = pd.read_csv(
                fileName,
                sep=",",
                header=Python.evaluate("None"),
                skiprows=1
            ).values

            var utc_time    = rawData[Python.evaluate("slice(None)"), 0]
            var data_stream = rawData[Python.evaluate("slice(None)"), 1]

            # ── Window parameters ─────────────────────────────────────────
            var start_point  = 0
            var end_point    = 500
            var window_shift = 500
            var fs: Float64  = 50.0

            # ── 1. Body-movement / bed-empty detection ────────────────────
            var result = detect_patterns(
                start_point, end_point, window_shift,
                data_stream, utc_time, plot=1
            )
            data_stream = result[0]
            utc_time    = result[1]

            # ── 2. BCG signal extraction (heartbeat band) ─────────────────
            var movement = band_pass_filtering(data_stream, fs, "bcg")

            # ── 3. Respiratory signal extraction ──────────────────────────
            var breathing = band_pass_filtering(data_stream, fs, "breath")
            breathing = remove_nonLinear_trend(breathing, 3)
            breathing = scipy_signal.savgol_filter(breathing, 11, 3)

            # ── 4. MODWT decomposition for heart-cycle isolation ──────────
            var w            = modwt(movement, "bior3.9", 4)
            var dc           = modwtmra(w, "bior3.9")
            var wavelet_cycle = dc[4]

            # ── 5. Vital-signs windows ────────────────────────────────────
            var t1: Int = 0
            var t2: Int = 500
            var ws            = window_shift
            var limit = math.floor(breathing.size / ws)

            # ── 6. Heart Rate ─────────────────────────────────────────────
            var beats_hr = vitals(t1, t2, ws, limit, wavelet_cycle, utc_time,
                               mpd=1, plot=0)
            print("\nHeart Rate Information")
            print("Minimum pulse : ", np.around(np.min(beats_hr)))
            print("Maximum pulse : ", np.around(np.max(beats_hr)))
            print("Average pulse : ", np.around(np.mean(beats_hr)))

            # ── 7. Respiratory Rate ───────────────────────────────────────
            var beats_rr = vitals(t1, t2, ws, limit, breathing, utc_time,
                           mpd=1, plot=0)
            print("\nRespiratory Rate Information")
            print("Minimum breathing : ", np.around(np.min(beats_rr)))
            print("Maximum breathing : ", np.around(np.max(beats_rr)))
            print("Average breathing : ", np.around(np.mean(beats_rr)))

            # ── 8. Apnea event detection ──────────────────────────────────
            var thresh = 0.3
            var events = apnea_events(breathing, utc_time, thresh=thresh)

            # ── 9. Save signal plots ──────────────────────────────────────
            var plot_t1: Int = 2500
            var plot_t2: Int = 2500 * 2
            data_subplot(data_stream, movement, breathing, wavelet_cycle,
                         plot_t1, plot_t2)

            # ── 10. Export per-window results CSV for evaluation ──────────
            var results_df = pd.DataFrame(
                Python.evaluate("{'estimated_hr': [], 'estimated_rr': []}")
            )
            var hr_list = builtins.list()
            var rr_list = builtins.list()
            var n_wins: Int = atol(String(beats_hr.size))
            for i in range(n_wins):
                hr_list.append(beats_hr[i])
                rr_list.append(beats_rr[i])
            var out_dict = builtins.dict()
            out_dict["estimated_hr"] = hr_list
            out_dict["estimated_rr"] = rr_list
            var out_df = pd.DataFrame(out_dict)
            out_df.to_csv(
                "/mnt/c/Users/Nourhan/Downloads/mojo/results/mojo_output.csv",
                index=False
            )
            print("\n[OUTPUT] Per-window results saved to results/mojo_output.csv")

    print("\nEnd processing ...")
