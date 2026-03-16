import matplotlib.pyplot as plt
import os


def plot_segments(df, sensor, segments, out_path):
    x = range(len(df))
    y = df[sensor].astype(float).to_numpy()

    plt.figure(figsize=(12, 4))
    plt.plot(x, y)

    for start, end in segments:
        plt.axvline(start, linestyle="--")
        plt.axvline(end, linestyle="--")

    plt.title(f"Segmentation for {sensor}")
    plt.xlabel("Time Index")
    plt.ylabel(sensor)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
