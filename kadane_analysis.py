import numpy as np
from collections import Counter


def kadane(arr):
    max_sum = arr[0]
    current_sum = arr[0]
    start = 0
    end = 0
    temp_start = 0

    for i in range(1, len(arr)):
        if arr[i] > current_sum + arr[i]:
            current_sum = arr[i]
            temp_start = i
        else:
            current_sum += arr[i]

        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i

    return max_sum, start, end


def sensor_max_deviation(signal):
    diffs = np.abs(np.diff(signal))
    centered = diffs - np.mean(diffs)

    max_sum, start, end = kadane(centered)

    return {
        "max_sum": float(max_sum),
        "start": int(start),
        "end": int(end + 1),  # shift to align roughly with original signal
        "processed_series": centered
    }


def analyze_all_sensors(df, sensor_cols):
    results = []

    for sensor in sensor_cols:
        signal = df[sensor].astype(float).to_numpy()

        if len(signal) < 2:
            continue

        out = sensor_max_deviation(signal)

        interval_labels = df.iloc[out["start"]:out["end"] + 1]["rul_class"].tolist()
        counts = Counter(interval_labels)

        dominant_class = None
        if counts:
            dominant_class = counts.most_common(1)[0][0]

        results.append({
            "sensor": sensor,
            "max_sum": out["max_sum"],
            "start": out["start"],
            "end": out["end"],
            "dominant_rul_class": dominant_class,
            "rul_class_counts": dict(counts)
        })

    return results
