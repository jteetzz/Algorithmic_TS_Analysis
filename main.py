import os
import numpy as np
import pandas as pd

from load_data import load_dataset
from preprocess import make_rul_categories, select_10_sensors
from segmentation import segment_statistics
from clustering import top_down_clustering, summarize_clusters
from kadane_analysis import analyze_all_sensors
from visualization import plot_segments


def main():
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # Load data
    df, sensor_cols = load_dataset(n_rows=10000)

    # Create RUL categories
    df, quantiles = make_rul_categories(df)

    print("\nRUL Quantiles:")
    print(quantiles)

    # Task 1: Segmentation
    selected_sensors = select_10_sensors(sensor_cols)
    print("\nSelected 10 sensors for Task 1:")
    print(selected_sensors)

    seg_results = segment_statistics(
        df,
        selected_sensors,
        threshold_factor=0.5,
        min_len=32
    )

    segmentation_rows = []

    for result in seg_results:
        sensor = result["sensor"]
        num_segments = result["num_segments"]

        print(f"\n{sensor}: {num_segments} segments")

        segmentation_rows.append({
            "sensor": sensor,
            "num_segments": num_segments,
            "threshold": result["threshold"]
        })

        plot_segments(
            df,
            sensor,
            result["segments"],
            f"results/figures/{sensor}_segments.png"
        )

    pd.DataFrame(segmentation_rows).to_csv(
        "results/tables/task1_segmentation_summary.csv",
        index=False
    )

    # save detailed segment info
    detailed_segment_rows = []
    for result in seg_results:
        for info in result["segment_info"]:
            detailed_segment_rows.append({
                "sensor": result["sensor"],
                "start": info["start"],
                "end": info["end"],
                "length": info["length"],
                "majority_rul_class": info["majority_rul_class"]
            })

    pd.DataFrame(detailed_segment_rows).to_csv(
        "results/tables/task1_segmentation_details.csv",
        index=False
    )

    # Task 2: Clustering
    X = df[sensor_cols].astype(float).to_numpy()

    # z-score normalization without ML libraries
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    X_norm = (X - means) / stds

    clusters = top_down_clustering(X_norm, k=4)
    cluster_summary = summarize_clusters(clusters, df["rul_class"].tolist())

    print("\nCluster Summary:")
    for row in cluster_summary:
        print(row)

    pd.DataFrame(cluster_summary).to_csv(
        "results/tables/task2_cluster_summary.csv",
        index=False
    )

    # Save row-to-cluster assignments
    assignments = []
    for cluster_id, indices in enumerate(clusters):
        for idx in indices:
            assignments.append({
                "row_index": idx,
                "cluster_id": cluster_id,
                "true_rul_class": df.iloc[idx]["rul_class"]
            })

    pd.DataFrame(assignments).sort_values("row_index").to_csv(
        "results/tables/task2_cluster_assignments.csv",
        index=False
    )

    # Task 3: Kadane Analysis
    kadane_results = analyze_all_sensors(df, sensor_cols)

    print("\nKadane Results:")
    for row in kadane_results[:5]:
        print(row)

    pd.DataFrame(kadane_results).to_csv(
        "results/tables/task3_kadane_summary.csv",
        index=False
    )

    print("\nDone. Results saved in results/ folder.")


if __name__ == "__main__":
    main()
