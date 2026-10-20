import numpy as np
from scipy.stats import skew, kurtosis


def get_pairwise_distances(
        coords: np.ndarray) -> np.ndarray:

    n_atoms = len(coords)

    distances = []

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            diff = coords[i] - coords[j]
            
            distance = np.linalg.norm(diff)
            distances.append(distance)

    return np.array(distances)

def distance_statistics(
        distances: np.ndarray) -> dict[str, float]:

    statistics = {
        "mean_distance": np.mean(distances),
        "std_distance": np.std(distances),

        "min_distance": min(distances),
        "max_distance": max(distances),
        "range_distance": max(distances) - min(distances),

        "10th_distance": np.percentile(distances, 10),
        "25th_distance": np.percentile(distances, 25),
        "median_distance": np.percentile(distances, 50),
        "75th_distance": np.percentile(distances, 75),
        "90th_distance": np.percentile(distances, 90),

        "iqr_distance": np.percentile(distances, 75) - np.percentile(distances, 25),
        
        "rms_distance": np.sqrt(np.mean(distances**2)),
        "cv_distance": np.std(distances) / np.mean(distances),

        "skew_distance": skew(distances),
        "kurtosis_distance": kurtosis(distances),
    }

    return statistics

def distance_histogram(
        distances: np.ndarray,
        bins: np.ndarray) -> np.ndarray:

    counts, _ = np.histogram(distances, bins=bins)

    normalized_counts = counts / len(distances)

    return normalized_counts

