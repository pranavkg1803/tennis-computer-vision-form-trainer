import numpy as np
import dtaidistance
#import h5py
#import hypothesis
# Two curves (as lists or numpy arrays)
curve1 = [1, 2, 3, 4, 5]
curve2 = [2, 3, 4, 5, 6]

# Compute the distance and the warping path
distance = dtaidistance.dtw.distance(curve1, curve2)
path = dtaidistance.dtw.warping_path(curve1, curve2)

print("Distance:", distance)
print("Warping Path:", path)

def distance_with_warping_path(curve1, curve2, warping_path):
    total_distance = 0
    for (i, j) in warping_path:
        total_distance += (curve1[i] - curve2[j]) ** 2
    return np.sqrt(total_distance)

def distance_with_warping_path_fast(curve1, curve2, warping_path):
    # This saves time by instead of looping through pairs and calculating, we use vectorization to do it all simultaneously.
    indices_curve1, indices_curve2 = zip(*warping_path)  # Separate the (i, j) pairs
    indices_curve1 = np.array(indices_curve1)  # Convert to NumPy arrays for vectorized operations
    indices_curve2 = np.array(indices_curve2)

    aligned_values_curve1 = curve1[indices_curve1]  # Values from curve1 along the path
    aligned_values_curve2 = curve2[indices_curve2]  # Values from curve2 along the path

    squared_differences = (aligned_values_curve1 - aligned_values_curve2) ** 2  # We take the square of the differences
    total_distance = np.sqrt(np.sum(squared_differences))
    return total_distance