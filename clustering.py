import numpy as np
from collections import Counter


def squared_distance(a, b):
    diff = a - b
    return float(np.dot(diff, diff))


def compute_centroid(X, indices):
    return np.mean(X[indices], axis=0)


def compute_sse(X, indices):
    if len(indices) == 0:
        return 0.0
    centroid = compute_centroid(X, indices)
    return sum(squared_distance(X[i], centroid) for i in indices)


def farthest_pair_indices(X, indices):
    best_pair = None
    best_dist = -1

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            d = squared_distance(X[indices[i]], X[indices[j]])
            if d > best_dist:
                best_dist = d
                best_pair = (indices[i], indices[j])

    return best_pair


def split_cluster(X, indices):
    if len(indices) < 2:
        return indices, []

    seed_a, seed_b = farthest_pair_indices(X, indices)

    cluster_a = []
    cluster_b = []

    for idx in indices:
        da = squared_distance(X[idx], X[seed_a])
        db = squared_distance(X[idx], X[seed_b])

        if da <= db:
            cluster_a.append(idx)
        else:
            cluster_b.append(idx)

    if len(cluster_a) == 0 or len(cluster_b) == 0:
        mid = len(indices) // 2
        cluster_a = indices[:mid]
        cluster_b = indices[mid:]

    return cluster_a, cluster_b


def top_down_clustering(X, k=4):
    clusters = [list(range(len(X)))]

    while len(clusters) < k:
        sse_values = [compute_sse(X, cluster) for cluster in clusters]
        split_index = int(np.argmax(sse_values))

        cluster_to_split = clusters.pop(split_index)

        left, right = split_cluster(X, cluster_to_split)

        if len(left) == 0 or len(right) == 0:
            clusters.append(cluster_to_split)
            break

        clusters.append(left)
        clusters.append(right)

    return clusters


def summarize_clusters(clusters, true_labels):
    summary = []

    for cluster_id, indices in enumerate(clusters):
        labels = [true_labels[i] for i in indices]
        counts = Counter(labels)

        majority_label = None
        majority_count = 0
        if counts:
            majority_label, majority_count = counts.most_common(1)[0]

        summary.append({
            "cluster_id": cluster_id,
            "size": len(indices),
            "majority_label": majority_label,
            "majority_count": majority_count,
            "label_counts": dict(counts)
        })

    return summary
