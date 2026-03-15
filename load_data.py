import kagglehub
import pandas as pd
import os


def load_dataset(n_rows=10000):

    # Download dataset
    path = kagglehub.dataset_download(
        "anseldsouza/water-pump-rul-predictive-maintenance"
    )

    print("Dataset downloaded to:", path)

    # Find CSV file
    csv_file = None
    for f in os.listdir(path):
        if f.endswith(".csv"):
            csv_file = os.path.join(path, f)
            break

    if csv_file is None:
        raise Exception("CSV file not found.")

    print("Using dataset file:", csv_file)

    df = pd.read_csv(csv_file)

    print("\nColumns in dataset:")
    print(df.columns)

    # Automatically detect sensor columns
    sensor_cols = [c for c in df.columns if "sensor" in c]

    print("\nDetected sensors:", len(sensor_cols))

    # Keep required columns
    needed_columns = ["timestamp", "rul"] + sensor_cols

    df = df[needed_columns].iloc[:n_rows].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    print("\nDataset shape:", df.shape)

    return df, sensor_cols
