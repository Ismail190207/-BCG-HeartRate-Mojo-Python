"""
Detect_body_movements.mojo
Classifies each time window of the raw sensor signal into:
    1 = Sleeping, 2 = Body movement, 3 = No subject on mat
"""

from std.python import Python
from std.python import PythonObject


def detect_patterns(pt1: Int, pt2: Int, win_size: Int,
                    data: PythonObject, time: PythonObject,
                    plot: Int) raises -> PythonObject:
    var np       = Python.import_module("numpy")
    var math     = Python.import_module("math")
    var builtins = Python.import_module("builtins")

    var pt1_orig = pt1
    var pt2_orig = pt2

    # PythonObject (math.floor result) -> Mojo Int via String conversion
    var limit_py = math.floor(data.size / win_size)
    var limit: Int = atol(String(limit_py))

    # Use Python list to build shape to avoid Mojo List[Int] issue
    var shape = builtins.list()
    shape.append(limit)
    shape.append(1)
    var event_flags = np.zeros(shape)

    var segments_sd = builtins.list()
    var p1 = pt1
    var p2 = pt2

    for _ in range(limit):
        var sub_data = data[Python.evaluate("slice(" + String(p1) + "," + String(p2) + ")")]
        segments_sd.append(np.std(sub_data, ddof=1))
        p1 = p2
        p2 = p2 + win_size

    var segments_sd_arr = np.array(segments_sd)
    var n = builtins.float(len(segments_sd))
    var mad = np.sum(np.abs(segments_sd_arr - np.mean(segments_sd_arr))) / n

    var thresh1: Float64 = 15.0
    var thresh2 = 2.0 * mad

    for j in range(limit):
        var std_val = np.around(segments_sd_arr[j])
        if std_val < thresh1:
            event_flags[j] = 3
        elif std_val > thresh2:
            event_flags[j] = 2
        else:
            event_flags[j] = 1

    if plot == 1:
        var matplotlib = Python.import_module("matplotlib")
        matplotlib.use("agg")
        var plt     = Python.import_module("matplotlib.pyplot")
        var patches = Python.import_module("matplotlib.patches")

        var data_min = np.min(data)
        var data_max = np.max(data)
        var width: PythonObject = data_min
        var height: PythonObject

        if data_min < 0:
            height = data_max + np.abs(data_min)
        else:
            height = data_max

        var current_axis = plt.gca()
        plt.plot(np.arange(0, data.size), data, "-k", linewidth=1)
        plt.xlabel("Time [Samples]")
        plt.ylabel("Amplitude [mV]")
        plt.gcf().autofmt_xdate()

        var r1 = pt1_orig
        var r2 = pt2_orig

        for j in range(limit):
            var sub_data = data[Python.evaluate("slice(" + String(r1) + "," + String(r2) + ")")]
            var sub_time = np.arange(r1, r2) / 50.0
            var rect_xy = builtins.list()
            rect_xy.append(r1)
            rect_xy.append(width)

            if event_flags[j] == 3:
                plt.plot(sub_time, sub_data, "-k", linewidth=1)
                current_axis.add_patch(patches.Rectangle(rect_xy, win_size, height, facecolor="#FAF0BE", alpha=0.2))
            elif event_flags[j] == 2:
                plt.plot(sub_time, sub_data, "-k", linewidth=1)
                current_axis.add_patch(patches.Rectangle(rect_xy, win_size, height, facecolor="#FF004F", alpha=1.0))
            else:
                plt.plot(sub_time, sub_data, "-k", linewidth=1)
                current_axis.add_patch(patches.Rectangle(rect_xy, win_size, height, facecolor="#00FFFF", alpha=0.2))
            r1 = r2
            r2 = r2 + win_size

        plt.savefig("/mnt/c/Users/Nourhan/Downloads/mojo/results/rawData.png")
        plt.close()

    var ind_move   = np.where(event_flags == 2)[0]
    var ind_empty  = np.where(event_flags == 3)[0]
    var ind2remove = np.sort(np.append(ind_empty, ind_move))

    var mask = np.ones(data.size, dtype="bool")
    mask[ind2remove] = False

    var filtered_data = data[mask]
    var filtered_time = time[mask]

    # Build Python tuple via builtins.list() to avoid Mojo List[PythonObject] type error
    var result_list = builtins.list()
    result_list.append(filtered_data)
    result_list.append(filtered_time)
    return builtins.tuple(result_list)
