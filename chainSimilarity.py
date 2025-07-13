import numpy as np
from collections import defaultdict

# Existing code from chainSimilarity.py
# Define chains matching our actual stored vectors and angles
KINEMATIC_CHAINS = {
    "right_arm": {
        "base_weight": 1.5,
        "joints": [
            {
                "vector": "rUpperArmVector",  # Starting vector from torso
                "angle": "rShoulderAngles",
                "threshold": 15
            },
            {
                "angle": "rElbowAngles",
                "threshold": 15
            }
        ]
    },
    "left_arm": {
        "base_weight": 1.5,
        "joints": [
            {
                "vector": "lUpperArmVector",  # Starting vector from torso
                "angle": "lShoulderAngles",
                "threshold": 15
            },
            {
                "angle": "lElbowAngles",
                "threshold": 15
            }
        ]
    },
    "right_leg": {
        "base_weight": 1.0,
        "joints": [
            {
                "vector": "rUpperLegVector",  # Starting vector from torso
                "angle": "rHipAngles",
                "threshold": 15
            },
            {
                "angle": "rKneeAngles",
                "threshold": 15
            },
            {
                "angle": "rAnkleAngles",
                "threshold": 15
            }
        ]
    },
    "left_leg": {
        "base_weight": 1.0,
        "joints": [
            {
                "vector": "lUpperLegVector",  # Starting vector from torso
                "angle": "lHipAngles",
                "threshold": 15
            },
            {
                "angle": "lKneeAngles",
                "threshold": 15
            },
            {
                "angle": "lAnkleAngles",
                "threshold": 15
            }
        ]
    }
}

# Add joint descriptions for user-friendly feedback
JOINT_DESCRIPTIONS = {
    "rShoulderAngles": {
        "name": "Right Shoulder",
        "action_increase": "raise your right arm more",
        "action_decrease": "lower your right arm more"
    },
    "rElbowAngles": {
        "name": "Right Elbow",
        "action_increase": "straighten your right arm more",
        "action_decrease": "bend your right elbow more"
    },
    "rHipAngles": {
        "name": "Right Hip",
        "action_increase": "increase your right hip angle",
        "action_decrease": "decrease your right hip angle"
    },
    "rKneeAngles": {
        "name": "Right Knee",
        "action_increase": "straighten your right leg more",
        "action_decrease": "bend your right knee more"
    },
    "rAnkleAngles": {
        "name": "Right Ankle",
        "action_increase": "point your right foot downward more",
        "action_decrease": "point your right foot upward more"
    },
    "lShoulderAngles": {
        "name": "Left Shoulder",
        "action_increase": "raise your left arm more",
        "action_decrease": "lower your left arm more"
    },
    "lElbowAngles": {
        "name": "Left Elbow",
        "action_increase": "straighten your left arm more",
        "action_decrease": "bend your left elbow more"
    },
    "lHipAngles": {
        "name": "Left Hip",
        "action_increase": "increase your left hip angle",
        "action_decrease": "decrease your left hip angle"
    },
    "lKneeAngles": {
        "name": "Left Knee",
        "action_increase": "straighten your left leg more",
        "action_decrease": "bend your left knee more"
    },
    "lAnkleAngles": {
        "name": "Left Ankle",
        "action_increase": "point your left foot downward more",
        "action_decrease": "point your left foot upward more"
    },
    "rUpperArmVector": {
        "name": "Right Upper Arm Position",
        "action_increase": "adjust your right arm position",
        "action_decrease": "adjust your right arm position"
    },
    "lUpperArmVector": {
        "name": "Left Upper Arm Position",
        "action_increase": "adjust your left arm position",
        "action_decrease": "adjust your left arm position"
    },
    "rUpperLegVector": {
        "name": "Right Upper Leg Position",
        "action_increase": "adjust your right leg position",
        "action_decrease": "adjust your right leg position"
    },
    "lUpperLegVector": {
        "name": "Left Upper Leg Position",
        "action_increase": "adjust your left leg position",
        "action_decrease": "adjust your left leg position"
    }
}

def calculate_chain_based_similarity(pro_angles, user_angles, dtw_path):
    """
    Calculate similarity using frame-by-frame analysis and chain-based approach.
    
    Args:
        pro_angles (PersonAngles): Professional's measurements
        user_angles (PersonAngles): User's measurements
        dtw_path (list): DTW warping path
    
    Returns:
        tuple: (temporal_scores, chain_weights, joint_scores)
    """
    print("Starting chain-based similarity calculation")
    
    # Input validation
    print(f"DTW Path length: {len(dtw_path)}")
    
    print("Pro angles keys:", list(pro_angles.angles.keys()))
    print("User angles keys:", list(user_angles.angles.keys()))
    
    # Validate input
    if not dtw_path:
        print("Warning: Empty DTW path")
        return {}, {}, {}
    
    # Initialize data structures
    temporal_scores = {}
    chain_weights = {}
    joint_scores = {}
    
    print("Initialized data structures")
    
    # Initialize joint scores for each angle type
    for angle_name in pro_angles.angles.keys():
        joint_scores[angle_name] = []
    
    print("Preparing to process DTW path")
    
    # Process each aligned frame pair from DTW path
    for path_index, (pro_idx, user_idx) in enumerate(dtw_path):
        if path_index % 100 == 0:  # Print progress for long paths
            print(f"Processing path pair {path_index}/{len(dtw_path)}")
        
        frame_score = 0
        valid_angles = 0
        
        # Calculate differences for each joint angle
        for angle_name in pro_angles.angles.keys():
            try:
                # Skip processing if not a valid angle/vector
                if not angle_name in joint_scores:
                    continue
                
                # Check if indices are valid
                if pro_idx >= len(pro_angles.angles[angle_name]):
                    print(f"Warning: Pro index {pro_idx} out of range for {angle_name}")
                    continue
                
                if user_idx >= len(user_angles.angles[angle_name]):
                    print(f"Warning: User index {user_idx} out of range for {angle_name}")
                    continue
                
                pro_val = pro_angles.angles[angle_name][pro_idx]
                user_val = user_angles.angles[angle_name][user_idx]
                
                if pro_val is not None and user_val is not None:
                    # Determine if we're dealing with a vector or scalar
                    if (isinstance(pro_val, (list, tuple, np.ndarray)) and 
                        isinstance(user_val, (list, tuple, np.ndarray))):
                        # For vector types (like Upper Arm Vector), use vector similarity
                        similarity = vector_similarity(pro_val, user_val)
                    else:
                        # For scalar types (like angles), calculate direct difference
                        diff = abs(pro_val - user_val)
                        similarity = max(0, 100 - diff)  # Convert to 0-100 scale
                    
                    # Add to joint scores
                    joint_scores[angle_name].append(similarity)
                    
                    # Add to frame score
                    frame_score += similarity
                    valid_angles += 1
                else:
                    joint_scores[angle_name].append(0)  # Add zero for missing data
                    
            except Exception as e:
                print(f"Error processing {angle_name}: {e}")
                import traceback
                traceback.print_exc()
                joint_scores[angle_name].append(0)  # Add zero for errors
        
        # Calculate average frame score
        if valid_angles > 0:
            temporal_scores[user_idx] = frame_score / valid_angles
        else:
            temporal_scores[user_idx] = 0
    
    print("Completed chain-based similarity calculation")
    print(f"Temporal scores length: {len(temporal_scores)}")
    print(f"Joint scores keys: {list(joint_scores.keys())}")
    
    return temporal_scores, chain_weights, joint_scores

def find_worst_segment(temporal_scores, segment_size_percent=25):
    """
    Find the worst-performing segment in the temporal scores.
    
    Args:
        temporal_scores (dict): Dictionary mapping frame indices to similarity scores
        segment_size_percent (float): Size of segment as percentage of total length
    
    Returns:
        tuple: (start_idx, end_idx, avg_score) of the worst segment
    """
    if not temporal_scores:
        return None, None, None
    
    # Convert to sorted list of (frame_idx, score) tuples
    sorted_scores = sorted(temporal_scores.items())
    total_frames = len(sorted_scores)
    
    if total_frames == 0:
        return None, None, None
    
    # Calculate segment size
    segment_size = max(3, int(total_frames * segment_size_percent / 100))
    
    # Find worst segment using sliding window
    worst_segment_start = 0
    worst_segment_score = float('inf')
    
    for i in range(total_frames - segment_size + 1):
        segment = sorted_scores[i:i+segment_size]
        segment_frames, segment_scores = zip(*segment)
        avg_score = sum(segment_scores) / len(segment_scores)
        
        if avg_score < worst_segment_score:
            worst_segment_score = avg_score
            worst_segment_start = i
    
    start_idx = sorted_scores[worst_segment_start][0]
    end_idx = sorted_scores[worst_segment_start + segment_size - 1][0]
    
    print(f"Worst segment: frames {start_idx}-{end_idx}, score: {worst_segment_score:.2f}")
    
    return start_idx, end_idx, worst_segment_score

def identify_problem_joints(joint_scores, start_idx, end_idx, num_joints=3):
    """
    Identify the worst-performing joints in a specific segment.
    
    Args:
        joint_scores (dict): Dictionary mapping joint names to lists of scores
        start_idx (int): Start index of segment
        end_idx (int): End index of segment
        num_joints (int): Number of problematic joints to identify
    
    Returns:
        list: List of tuples (joint_name, avg_score, pro_avg, user_avg, difference)
    """
    if not joint_scores or start_idx is None or end_idx is None:
        return []
    
    joint_segment_scores = {}
    
    for joint_name, scores in joint_scores.items():
        # Skip if joint scores are too short
        if len(scores) <= end_idx:
            continue
            
        # Extract scores for the segment
        segment_scores = scores[start_idx:end_idx+1]
        
        # Calculate average score for this joint in this segment
        if segment_scores:
            avg_score = sum(segment_scores) / len(segment_scores)
            joint_segment_scores[joint_name] = avg_score
    
    # Sort joints by average score (ascending) and get worst ones
    worst_joints = sorted(joint_segment_scores.items(), key=lambda x: x[1])[:num_joints]
    
    return worst_joints

def generate_feedback(pro_angles, user_angles, dtw_path, segment_size_percent=25, num_problem_joints=3):
    """
    Generate actionable feedback by analyzing the worst segment and problem joints.
    
    Args:
        pro_angles (PersonAngles): Professional's measurements
        user_angles (PersonAngles): User's measurements
        dtw_path (list): DTW warping path
        segment_size_percent (float): Size of segment as percentage of total
        num_problem_joints (int): Number of problematic joints to identify
    
    Returns:
        dict: Feedback information including worst segment, problem joints, and recommendations
    """
    # Calculate similarity scores
    temporal_scores, chain_weights, joint_scores = calculate_chain_based_similarity(
        pro_angles, user_angles, dtw_path
    )
    
    # Find worst segment
    start_idx, end_idx, worst_segment_score = find_worst_segment(
        temporal_scores, segment_size_percent
    )
    
    # Map DTW path indices to actual frame indices
    frame_mapping = {}
    for path_idx, (pro_idx, user_idx) in enumerate(dtw_path):
        frame_mapping[path_idx] = user_idx
    
    # Find problem joints in worst segment
    worst_joints = identify_problem_joints(
        joint_scores, start_idx, end_idx, num_problem_joints
    )
    
    # Calculate detailed statistics for each problem joint
    detailed_joint_feedback = []
    for joint_name, avg_score in worst_joints:
        try:
            # Extract joint values for the segment using DTW path mapping
            segment_dtw_indices = [i for i, (_, user_idx) in enumerate(dtw_path) 
                                if user_idx >= start_idx and user_idx <= end_idx]
            
            segment_pro_indices = [dtw_path[i][0] for i in segment_dtw_indices]
            segment_user_indices = [dtw_path[i][1] for i in segment_dtw_indices]
            
            # Get pro and user values for this joint in the segment
            pro_values = [pro_angles.angles[joint_name][i] for i in segment_pro_indices 
                        if i < len(pro_angles.angles[joint_name]) and pro_angles.angles[joint_name][i] is not None]
            
            user_values = [user_angles.angles[joint_name][i] for i in segment_user_indices 
                         if i < len(user_angles.angles[joint_name]) and user_angles.angles[joint_name][i] is not None]
            
            if not pro_values or not user_values:
                continue
                
            # Calculate statistics
            pro_avg = sum(pro_values) / len(pro_values)
            user_avg = sum(user_values) / len(user_values)
            difference = user_avg - pro_avg
            
            # Generate specific feedback based on the joint and difference
            if joint_name in JOINT_DESCRIPTIONS:
                joint_info = JOINT_DESCRIPTIONS[joint_name]
                friendly_name = joint_info["name"]
                
                # Determine action based on difference
                if difference > 0:
                    action = joint_info["action_decrease"]
                    direction = "decrease"
                else:
                    action = joint_info["action_increase"]
                    direction = "increase"
                
                abs_diff = abs(difference)
                
                feedback = {
                    "joint_name": joint_name,
                    "friendly_name": friendly_name,
                    "avg_score": avg_score,
                    "pro_avg": pro_avg,
                    "user_avg": user_avg,
                    "difference": difference,
                    "abs_difference": abs_diff,
                    "action": action,
                    "direction": direction,
                    "feedback": f"{friendly_name}: {action} by approximately {abs_diff:.1f} degrees"
                }
                
                detailed_joint_feedback.append(feedback)
                
        except Exception as e:
            print(f"Error generating feedback for {joint_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create feedback summary
    feedback_summary = {
        "worst_segment": {
            "start_frame": start_idx,
            "end_frame": end_idx,
            "score": worst_segment_score
        },
        "problem_joints": detailed_joint_feedback,
        "overall_feedback": "Focus on improving these specific joints to match the professional's form."
    }
    
    # Format time range from frames
    if start_idx is not None and end_idx is not None:
        feedback_summary["time_range"] = f"Frames {start_idx} to {end_idx}"
    
    return feedback_summary

def vector_similarity(v1, v2):
    """
    Compute similarity between two vectors.
    Returns value between 0 (completely different) and 100 (identical).
    """
    if v1 is None or v2 is None:
        return 0
        
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Compute cosine similarity
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norms == 0:
        return 0
        
    cos_sim = dot_product / norms
    
    # Convert to percentage (0-100)
    return ((cos_sim + 1) / 2) * 100