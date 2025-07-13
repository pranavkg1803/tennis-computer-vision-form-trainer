import numpy as np
import cv2
import mediapipe as mp

def visualize_pose_feedback(image, current_angles, reference_angles, threshold=15):
    
    return image

def get_direction_vector(angle_diff, direction, origin, length=50):
    """Calculate the endpoint of a direction arrow based on angle difference"""
    angle_rad = np.radians(angle_diff * direction)
    dx = int(length * np.cos(angle_rad))
    dy = int(length * np.sin(angle_rad))
    end_point = (origin[0] + dx, origin[1] + dy)
    return end_point

def visualize_pose_feedback1(image, current_angles, reference_angles, threshold=15):
    """
    Add visual feedback comparing current pose to reference pose.
    
    Args:
        image: OpenCV image to draw on
        current_angles: Dictionary with angles and coordinates
        reference_angles: Dictionary with angles and coordinates
        threshold: Acceptable angle difference threshold in degrees
    
    Returns:
        Image with visual feedback drawn on it
    """
    h, w = image.shape[:2]
    
    # Map joint names to their angle keys
    joint_mapping = {
        "lShoulder": "lShoulderAngle",
        "rShoulder": "rShoulderAngle",
        "lElbow": "lElbowAngle",
        "rElbow": "rElbowAngle",
        "lHip": "lHipAngle",
        "rHip": "rHipAngle",
        "lKnee": "lKneeAngle",
        "rKnee": "rKneeAngle",
        "lAnkle": "lAnkleAngle",
        "rAnkle": "rAnkleAngle"
    }
    
    for joint_name, angle_key in joint_mapping.items():
        # Get coordinates for this joint
        joint_coords = current_angles[joint_name]
        if joint_coords is None:
            continue
            
        # Get pixel coordinates
        pixel_coords = (int(joint_coords[0] * w), int(joint_coords[1] * h))
        
        # Get angles
        current = current_angles[angle_key]
        reference = reference_angles[angle_key]
        
        if current is not None and reference is not None:
            # Calculate angle difference
            diff = (reference - current + 180) % 360 - 180
            abs_diff = abs(diff)
            direction = 1 if diff > 0 else -1
            
            if abs_diff <= threshold:
                # Draw green circle for correct angle
                cv2.circle(image, pixel_coords, 15, (0, 255, 0), -1)
                cv2.circle(image, pixel_coords, 17, (255, 255, 255), 2)  # White outline
            else:
                # Draw red circle and correction arrow
                cv2.circle(image, pixel_coords, 15, (0, 0, 255), -1)
                cv2.circle(image, pixel_coords, 17, (255, 255, 255), 2)  # White outline
                
                # Draw direction arrow
                end_point = get_direction_vector(abs_diff, direction, pixel_coords)
                cv2.arrowedLine(image, pixel_coords, end_point, (255, 255, 255), 5)  # White outline
                cv2.arrowedLine(image, pixel_coords, end_point, (0, 0, 255), 3)      # Red arrow
                
                # Add text showing required adjustment
                text = f"{abs_diff:.1f}°"
                text_pos = (pixel_coords[0] - 20, pixel_coords[1] - 20)
                cv2.putText(image, text, text_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)  # White outline
                cv2.putText(image, text, text_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)      # Red text
    
    return image


def get_target_vector(angle, origin, length=50):
    """
    Calculate vector endpoint based on an absolute angle.
    Angle is in degrees where:
    0/360° = right
    90° = down
    180° = left
    270° = up
    """
    angle_rad = np.radians(angle)
    dx = int(length * np.cos(angle_rad))
    dy = int(length * np.sin(angle_rad))
    end_point = (origin[0] + dx, origin[1] + dy)
    return end_point

def visualize_pose_feedback2(image, current_angles, reference_angles, threshold=15):
    """
    Add visual feedback showing where each body segment should point.
    
    Args:
        image: OpenCV image to draw on
        current_angles: Dictionary with angles and coordinates
        reference_angles: Dictionary with angles and coordinates
        threshold: Acceptable angle difference threshold in degrees
    
    Returns:
        Image with visual feedback drawn on it
    """
    h, w = image.shape[:2]
    
    # Map joints to their relevant points and angles
    joint_mapping = {
        # joint_name: (angle_key, proximal_point, distal_point)
        "lElbow": ("lElbowAngle", "lShoulder", "lWrist"),
        "rElbow": ("rElbowAngle", "rShoulder", "rWrist"),
        "lShoulder": ("lShoulderAngle", "lHip", "lElbow"),
        "rShoulder": ("rShoulderAngle", "rHip", "rElbow"),
        "lHip": ("lHipAngle", "lKnee", "lShoulder"),
        "rHip": ("rHipAngle", "rKnee", "rShoulder"),
        "lKnee": ("lKneeAngle", "lAnkle", "lHip"),
        "rKnee": ("rKneeAngle", "rAnkle", "rHip")
    }
    
    for joint_name, (angle_key, prox_key, dist_key) in joint_mapping.items():
        # Get joint position
        joint_coords = current_angles[joint_name]
        if joint_coords is None:
            continue
            
        # Get pixel coordinates
        pixel_coords = (int(joint_coords[0] * w), int(joint_coords[1] * h))
        
        # Get angles
        current = current_angles[angle_key]
        reference = reference_angles[angle_key]
        
        if current is not None and reference is not None:
            # Calculate angle difference
            diff = (reference - current + 180) % 360 - 180
            abs_diff = abs(diff)
            
            if abs_diff <= threshold:
                # Draw green circle for correct angle
                cv2.circle(image, pixel_coords, 15, (0, 255, 0), -1)
                cv2.circle(image, pixel_coords, 17, (255, 255, 255), 2)  # White outline
            else:
                # Draw red circle
                cv2.circle(image, pixel_coords, 15, (0, 0, 255), -1)
                cv2.circle(image, pixel_coords, 17, (255, 255, 255), 2)  # White outline
                
                # Draw target direction arrow
                target_end = get_target_vector(reference, pixel_coords, length=60)
                
                # Draw arrow with white outline for visibility
                cv2.arrowedLine(image, pixel_coords, target_end, (255, 255, 255), 5)  # White outline
                cv2.arrowedLine(image, pixel_coords, target_end, (0, 0, 255), 3)      # Red arrow
                
                # Add text showing angle difference
                text = f"{abs_diff:.1f}°"
                text_pos = (pixel_coords[0] - 20, pixel_coords[1] - 20)
                cv2.putText(image, text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)  # White outline
                cv2.putText(image, text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)      # Red text
    
    return image