
def calculate_chain_based_similarity(pro_angles, user_angles, dtw_path):
    """
    Calculate similarity using the chain-based approach where joint
    weighting depends on the similarity of joints higher in the chain.
    
    Args:
        pro_angles (PersonAngles): Professional's measurements
        user_angles (PersonAngles): User's measurements
        dtw_path (list): DTW warping path as [(pro_idx, user_idx), ...]
    
    Returns:
        tuple: (temporal_scores, chain_weights, joint_scores, chain_scores)
    """
    print("Starting chain-based similarity calculation")
    
    # Input validation
    if not dtw_path:
        print("Warning: Empty DTW path")
        return {}, {}, {}, {}
    
    # Initialize data structures
    temporal_scores = {}  # Overall frame-by-frame scores
    chain_weights = {}    # Weights for each chain
    joint_scores = {}     # Raw similarity for each joint
    chain_scores = {}     # Similarity scores for each chain
    
    # Initialize the joint scores dictionary
    for angle_name in pro_angles.angles.keys():
        joint_scores[angle_name] = []
    
    # Initialize chain scores
    for chain_name in KINEMATIC_CHAINS.keys():
        chain_scores[chain_name] = []
    
    # Process each aligned frame pair from DTW path
    for path_index, (pro_idx, user_idx) in enumerate(dtw_path):
        if path_index % 100 == 0:  # Print progress for long paths
            print(f"Processing frame pair {path_index}/{len(dtw_path)}")
        
        # Calculate chain scores for this frame pair
        frame_chain_scores = {}
        
        # Process each chain (right arm, left arm, right leg, left leg)
        for chain_name, chain_info in KINEMATIC_CHAINS.items():
            # Get the base weight for this chain
            base_weight = chain_info["base_weight"]
            
            # Calculate each joint's similarity in the chain
            chain_joint_scores = []
            chain_joint_weights = []
            
            # Track if the previous joint in chain was similar enough
            prev_joint_similar = True  # The core is always a fixed reference
            prev_joint_similarity = 100  # Start with perfect similarity for core
            
            # Process each joint in this chain
            for joint in sorted(chain_info["joints"], key=lambda j: j["position"]):
                angle_name = joint["angle"]
                threshold = joint["threshold"]
                
                # Calculate vector similarity if this is a base joint
                if "vector" in joint:
                    vector_name = joint["vector"]
                    
                    # Get vectors if available
                    if (vector_name in pro_angles.angles and 
                        vector_name in user_angles.angles and
                        pro_idx < len(pro_angles.angles[vector_name]) and
                        user_idx < len(user_angles.angles[vector_name])):
                        
                        pro_vector = pro_angles.angles[vector_name][pro_idx]
                        user_vector = user_angles.angles[vector_name][user_idx]
                        
                        if pro_vector is not None and user_vector is not None:
                            # Calculate vector similarity for base joint position
                            vector_sim = vector_similarity(pro_vector, user_vector)
                            joint_scores.setdefault(vector_name, []).append(vector_sim)
                
                # Skip if angle data is missing
                if (angle_name not in pro_angles.angles or 
                    angle_name not in user_angles.angles or
                    pro_idx >= len(pro_angles.angles[angle_name]) or
                    user_idx >= len(user_angles.angles[angle_name])):
                    continue
                
                pro_angle = pro_angles.angles[angle_name][pro_idx]
                user_angle = user_angles.angles[angle_name][user_idx]
                
                # Skip if angle data is None
                if pro_angle is None or user_angle is None:
                    joint_scores[angle_name].append(0)
                    chain_joint_scores.append(0)
                    chain_joint_weights.append(0)
                    continue
                
                # Calculate angle similarity
                similarity = angle_similarity(pro_angle, user_angle, threshold)
                
                # Store raw similarity score
                joint_scores[angle_name].append(similarity)
                
                # Calculate joint weight based on previous joint's similarity
                # Key innovation: Weight decreases down the chain if previous joints don't match
                # This implements the hierarchical dependency described in the text
                if joint["position"] == 0:
                    # Base joint always has full weight (connected to core)
                    joint_weight = 1.0
                else:
                    # Joint weight depends on similarity of previous joint
                    # If previous joint was very different, this joint's weight is reduced
                    # This implements the "fixed reference point" concept
                    joint_weight = (prev_joint_similarity / 100)
                
                # Determine if this joint is similar enough to be a "fixed point"
                # for the next joint in the chain
                prev_joint_similar = similarity > 70  # 70% similarity threshold
                prev_joint_similarity = similarity  # Store for next joint
                
                # Add to chain scores with weighting
                chain_joint_scores.append(similarity)
                chain_joint_weights.append(joint_weight)
            
            # Calculate overall chain score as weighted average
            if chain_joint_scores and sum(chain_joint_weights) > 0:
                weighted_score = sum(s * w for s, w in zip(chain_joint_scores, chain_joint_weights))
                weighted_score /= sum(chain_joint_weights)
                frame_chain_scores[chain_name] = weighted_score
            else:
                frame_chain_scores[chain_name] = 0
            
            # Store chain score for this frame
            chain_scores[chain_name].append(frame_chain_scores[chain_name])
        
        # Calculate overall frame score (weighted average of chain scores)
        if frame_chain_scores:
            weighted_sum = 0
            total_weight = 0
            
            for chain_name, score in frame_chain_scores.items():
                chain_weight = KINEMATIC_CHAINS[chain_name]["base_weight"]
                weighted_sum += score * chain_weight
                total_weight += chain_weight
            
            if total_weight > 0:
                temporal_scores[user_idx] = weighted_sum / total_weight
            else:
                temporal_scores[user_idx] = 0
        else:
            temporal_scores[user_idx] = 0
    
    print("Completed chain-based similarity calculation")
    print(f"Processed {len(dtw_path)} frame pairs")
    
    return temporal_scores, chain_weights, joint_scores, chain_scores
            
        