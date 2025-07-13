import numpy as np

def distance_with_warping_path_fast(curve1, curve2, warping_path):
    if not curve1 or not curve2 or not warping_path:
        return None
    curve1 = np.array(curve1, dtype=float)
    curve2 = np.array(curve2, dtype=float)
    
    if not curve1.size or not curve2.size: # Check if array is empty after conversion
        return None
    
    indices_curve1, indices_curve2 = zip(*warping_path)
    indices_curve1 = np.array(indices_curve1)
    indices_curve2 = np.array(indices_curve2)
    
    aligned_values_curve1 = curve1[indices_curve1]
    aligned_values_curve2 = curve2[indices_curve2]
    
    squared_differences = (aligned_values_curve1 - aligned_values_curve2) ** 2
    rmse = np.sqrt(np.mean(squared_differences))
    
    return float(rmse)

import numpy as np

def calculateSimilarityWeighted(pro_angles, user_angles, path):
    
    weights = {   
        'rShoulderAngles': 1.5,    # Critical for serve and overhead motions
        'rElbowAngles': 1.5,       # Essential for power generation
        'rWristAngles': 1.5,       # Important for racquet control
        'rHipAngles': 0.2,         # Key for rotation and power transfer
        'rKneeAngles': 0.5,        # Important for loading and extension
        'rAnkleAngles': 0.1,       # Base stability
        # Left side (non-dominant for most players)
        'lShoulderAngles': 1,    # Balance and coordination
        'lElbowAngles': 1,       # Support for two-handed backhand
        'lWristAngles': 1,       # Support for two-handed backhand
        'lHipAngles': 0.1,         # Rotation and balance
        'lKneeAngles': 0.25,        # Support and stability
        'lAnkleAngles': 0.05 
    }
    
    weighted_distances = []
    none_angles = []
    
    for angle in pro_angles.angles.keys():
        if pro_angles.angles[angle] and user_angles.angles[angle]: 
            distance = distance_with_warping_path_fast(pro_angles.angles[angle], user_angles.angles[angle], path)
            if distance is not None:
                weight = weights.get(angle, 1.0)  # Default weight of 1.0 if not specified
                weighted_distances.append(distance * weight)
            else:
                none_angles.append(angle)
    
    if none_angles:
        print("The following angles could not be compared:")
        for angle in none_angles:
            print(f"- {angle}")
    
    return np.mean(weighted_distances) if weighted_distances else None
 

def calculateSimilarity1(pro, user, path):
    #the first algorithm will just use path to find difference in each joint. Here we print which angles could not be compred.
    distances = []
    none_angles = []

    for angle in pro.angles.keys():
        if pro.angles[angle] and user.angles[angle]:
            distance = distance_with_warping_path_fast(pro.angles[angle], user.angles[angle], path)
            if distance is not None:
                distances.append(distance)
            else:
                none_angles.append(angle)

    # Print angles that returned None
    if none_angles:
        print("The following angles could not be compared:")
        for angle in none_angles:
            print(f"- {angle}")

    # Calculate mean of valid distances
    mean_distance = np.mean(distances) if distances else None
    return mean_distance

#For Strucutral Similairty we need to iterate through the frames, getting similairty of each skeleton frame.
def calculateSimilarityStructural(pro, user, path):
    
    if not pro.angles["rElbowAngles"]:
        num_frames = len(pro.angles["rElbowAngles"])
    else:
        num_frames = next(len(angles) for angles in pro.angles.values() if angles)
    similarities = [] #data points both user and pro have in commen
    
    for i in range(len(path)):
        (proIndex,userIndex) = path[i]
        
    
    return None

def vector_similarity(v1, v2):
    """
    Computes similarity between two vectors, scaled from 0 to 1.
    
    Args:
        v1 (np.array): First vector.
        v2 (np.array): Second vector.
    
    Returns:
        float: Similarity score between 0 (opposite) and 1 (identical).
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Compute cosine similarity
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # If one of the vectors is zero, similarity is 0
    
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    
    # Scale from [-1, 1] to [0, 1]
    return (cosine_similarity + 1) / 2
    