"""
Detect_apnea_events.mojo
Detects apnea events by finding sub-windows whose respiratory standard
deviation falls below a fraction of the window mean SD.
"""

from std.python import Python
from std.python import PythonObject


def unix_time_converter(unix_time: PythonObject) raises -> PythonObject:
    var pd = Python.import_module("pandas")
    var tm = pd.to_datetime(unix_time, unit="ms")
    var readable_time = (
        tm.tz_localize("UTC")
          .tz_convert("Asia/Singapore")
          .strftime("%H.%M.%S")
    )
    return readable_time


def apnea_events(data: PythonObject, utc_time: PythonObject,
                 thresh: Float64) raises -> PythonObject:
    var np       = Python.import_module("numpy")
    var math     = Python.import_module("math")
    var builtins = Python.import_module("builtins")

    var win_size  = 1500
    var hop_size  = 500
    var win_shift = win_size
    var pt1       = 0
    var pt2       = win_size

    # PythonObject (math.floor result) -> Mojo Int via String conversion
    var limit_py  = math.floor(data.size / win_size)
    var limit_int: Int = atol(String(limit_py))
    var counter = 0

    var start_time_list = builtins.list()
    var stop_time_list  = builtins.list()

    for _ in range(limit_int):
        var sub_data     = data[Python.evaluate("slice(" + String(pt1) + "," + String(pt2) + ")")]
        var sub_utc_time = utc_time[Python.evaluate("slice(" + String(pt1) + "," + String(pt2) + ")")]

        var StDs = builtins.list()
        var sub_sub_utc_time = builtins.list()

        var so = 0
        while so < win_shift:
            var ndx       = np.arange(so, so + hop_size)
            sub_sub_utc_time.append(sub_utc_time[ndx])
            var fiber_data = sub_data[ndx]
            StDs.append(np.std(fiber_data, ddof=1))
            so = so + hop_size

        var StDs_arr = np.array(StDs)
        var T        = np.mean(StDs_arr)

        # Convert StDs_arr.size (PythonObject) to Mojo Int via String
        var stds_size: Int = atol(String(StDs_arr.size))
        var ind = builtins.list()
        for idx in range(stds_size):
            if StDs_arr[idx] <= thresh * T:
                ind.append(idx)

        if len(ind) > 0:
            for j in ind:
                counter += 1
                var current_time = sub_sub_utc_time[j]
                start_time_list.append(unix_time_converter(current_time[0]))
                stop_time_list.append(unix_time_converter(current_time[-1]))

                print("\nApnea Information")
                print("start time : ", start_time_list, " stop time : ", stop_time_list)

        pt1 = pt2
        pt2 = pt2 + win_size

    var events = builtins.dict()
    events[0] = start_time_list
    events[1] = stop_time_list
    return events
