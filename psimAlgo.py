import numpy as np

# Define body parts and their joints
BODY_PARTS = {
    'torso': ['head', 'neck', 'spine', 'hip_center'],
    'left_arm': ['left_shoulder', 'left_elbow', 'left_wrist'],
    'right_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
    'left_leg': ['left_hip', 'left_knee', 'left_ankle'],
    'right_leg': ['right_hip', 'right_knee', 'right_ankle']
}

def calculate_2d_angle(x, y):
    """Calculate angle from positive x-axis in 2D."""
    return np.arctan2(y, x)

def get_joint_angles(pose):
    """Convert all joint positions to angles."""
    angles = {}
    for joint_name, position in pose.items():
        angle = calculate_2d_angle(position[0], position[1])
        angles[joint_name] = angle
    return angles

def normalize_angles_for_body_part(angles, body_part_joints):
    """Apply MSCN normalization to angles within a body part."""
    # Collect angles for this body part
    part_angles = [angles[joint] for joint in body_part_joints]
    angle_array = np.array(part_angles)
    
    # Calculate statistics
    mu = np.mean(angle_array)
    sigma = np.std(angle_array) + 1e-6  # avoid division by zero
    
    # Normalize
    normalized = (angle_array - mu) / sigma
    
    return normalized

def calculate_similarity(v1, v2, C=1e-6):
    """Calculate similarity between two vectors."""
    numerator = 2 * np.sum(v1 * v2) + C
    denominator = (np.sum(v1**2) + np.sum(v2**2)) + C
    return numerator / denominator

def local_similarity(ref_angles, target_angles):
    """Calculate local structural similarity."""
    similarities = []
    
    for part_name, joints in BODY_PARTS.items():
        # Get normalized angle vectors for this body part
        QR = normalize_angles_for_body_part(ref_angles, joints)
        QT = normalize_angles_for_body_part(target_angles, joints)
        
        # Calculate similarity for this body part
        sim = calculate_similarity(QR, QT)
        similarities.append(sim)
    
    return np.mean(similarities)

def global_similarity(ref_angles, target_angles):
    """Calculate global structural similarity."""
    # Get all joints in order
    all_joints = []
    for joints in BODY_PARTS.values():
        all_joints.extend(joints)
    
    # Get normalized angles for all joints
    QR = normalize_angles_for_body_part(ref_angles, all_joints)
    QT = normalize_angles_for_body_part(target_angles, all_joints)
    
    return calculate_similarity(QR, QT)

def positional_similarity(ref_pose, target_pose):
    """Calculate positional similarity."""
    # Convert poses to vectors (using only x, y coordinates)
    XR = np.concatenate([pos[:2] for pos in ref_pose.values()])
    XT = np.concatenate([pos[:2] for pos in target_pose.values()])
    
    return calculate_similarity(XR, XT)

def get_bone_lengths(pose, skeleton_edges):
    """Calculate bone lengths for a pose."""
    bones = []
    for joint1, joint2 in skeleton_edges:
        # Use only x, y coordinates for 2D distance
        length = np.linalg.norm(pose[joint2][:2] - pose[joint1][:2])
        bones.append(length)
    return np.array(bones)

def intrinsic_similarity(ref_pose, target_pose, skeleton_edges):
    """Calculate intrinsic similarity based on bone lengths."""
    BR = get_bone_lengths(ref_pose, skeleton_edges)
    BT = get_bone_lengths(target_pose, skeleton_edges)
    
    return calculate_similarity(BR, BT)

def compute_psim(ref_pose, target_pose, skeleton_edges):
    """Compute the full PSIM metric between two poses."""
    # Convert poses to angles
    ref_angles = get_joint_angles(ref_pose)
    target_angles = get_joint_angles(target_pose)
    
    # Calculate all similarities
    local_sim = local_similarity(ref_angles, target_angles)
    global_sim = global_similarity(ref_angles, target_angles)
    pos_sim = positional_similarity(ref_pose, target_pose)
    intrinsic_sim = intrinsic_similarity(ref_pose, target_pose, skeleton_edges)
    
    # Weighted combination
    weights = {
        'local': 0.22,
        'global': 0.32,
        'positional': 0.28,
        'intrinsic': 0.18
    }
    
    psim = (weights['local'] * local_sim +
            weights['global'] * global_sim +
            weights['positional'] * pos_sim +
            weights['intrinsic'] * intrinsic_sim)
    
    return psim

def compute_temporal_psim(ref_sequence, target_sequence, skeleton_edges, window_size=3):
    """Compute PSIM for a sequence of poses with temporal weighting."""
    sequence_length = len(ref_sequence)
    scores = []
    weights = []
    
    for t in range(sequence_length):
        # Get window of frames
        start_idx = max(0, t - window_size + 1)
        window_R = ref_sequence[start_idx:t+1]
        window_T = target_sequence[start_idx:t+1]
        
        # Calculate base PSIM score for current frame
        score = compute_psim(ref_sequence[t], target_sequence[t], skeleton_edges)
        
        # Calculate temporal weight based on changes in window
        if len(window_R) > 1:
            # Calculate mean-subtracted poses in window (using only x, y coordinates)
            mean_R = np.mean([list(pose.values())[:2] for pose in window_R], axis=0)
            mean_T = np.mean([list(pose.values())[:2] for pose in window_T], axis=0)
            
            diff_R = np.array(list(window_R[-1].values()))[:2] - mean_R
            diff_T = np.array(list(window_T[-1].values()))[:2] - mean_T
            
            weight = np.mean(np.abs(diff_R - diff_T))
        else:
            weight = 1.0
        
        scores.append(score)
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Calculate weighted average
    return np.sum(np.array(scores) * weights)

# Example usage:
if __name__ == "__main__":
    # Example skeleton edges (joints that are connected)
    SKELETON_EDGES = [
        ('neck', 'left_shoulder'),
        ('neck', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
    ]
    
    # Example pose (dictionary of 2D joint positions)
   # example_pose = {
    #    'neck': np.array([0, 0]),
     #   'left_shoulder': np.array([-1, 0]),
     #   'left_elbow': np.array([-2, 0]),
     #   'left_wrist': np.array([-3, 0]),
    # }