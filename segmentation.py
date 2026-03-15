import numpy as np


def segment_signal(signal, threshold, min_len=32, start=0, end=None):
    if end is None:
        end = len(signal)

    segment = signal[start:end]

    if len(segment) <= min_len:
        return [(start, end)]

    seg_var = np.var(segment)

    if seg_var > threshold:
        mid = (start + end) // 2
        left_segments = segment_signal(signal, threshold, min_len, start, mid)
        right_segments = segment_signal(signal, threshold, min_len, mid, end)
        return left_segments + right_segments
    else:
        return [(start, end)]


def segmentation_complexity(segments):
    return len(segments)


def segment_statistics(df, sensor_list, threshold_factor=0.5, min_len=32):
    results = []

    for sensor in sensor_list:
        signal = df[sensor].astype(float).to_numpy()

        full_variance = np.var(signal)
        threshold = threshold_factor * full_variance

        segments = segment_signal(signal, threshold=threshold, min_len=min_len)
        complexity = segmentation_complexity(segments)

        segment_info = []
        for start, end in segments:
            majority_class = df.iloc[start:end]["rul_class"].mode()[0]
            segment_info.append({
                "start": start,
                "end": end,
                "length": end - start,
                "majority_rul_class": majority_class
            })

        results.append({
            "sensor": sensor,
            "threshold": threshold,
            "num_segments": complexity,
            "segments": segments,
            "segment_info": segment_info
        })

    return results
