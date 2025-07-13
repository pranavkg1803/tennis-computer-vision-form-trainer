from fastDTW import fastdtw
from scipy.spatial.distance import euclidean
import mediapipe

# Two curves (as lists or numpy arrays)
curve1 = [1, 2, 3, 4, 5]
curve2 = [2, 3, 4, 5, 6]

# Compute the distance and the warping path
distance, path = fastdtw(curve1, curve2, dist=euclidean)

print("Distance:", distance)
print("Warping Path:", path)