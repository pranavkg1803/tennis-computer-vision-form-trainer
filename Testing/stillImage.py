import cv2
import mediapipe as mp
import numpy as np
import os

class TennisPoseEstimator:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=0.5
        )

    def _get_limb_colors(self, stage):
        """
        Define limb colors based on the stage of tennis shot
        Green: Most important
        Yellow: Medium importance
        Red: Least important
        """
        # Ensure colors are in BGR format for OpenCV
        green = (0, 255, 0)    # Green
        yellow = (0, 255, 255) # Yellow
        red = (0, 0, 255)      # Red
        default_color = yellow  # Default to yellow

        # Color mapping
        color_map = {
            'preparation': {
                # Everything yellow during preparation
                self.mp_pose.PoseLandmark.LEFT_SHOULDER: yellow,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER: yellow,
                self.mp_pose.PoseLandmark.LEFT_ELBOW: yellow,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW: yellow,
                self.mp_pose.PoseLandmark.LEFT_KNEE: yellow,
                self.mp_pose.PoseLandmark.RIGHT_KNEE: yellow,
                self.mp_pose.PoseLandmark.LEFT_HIP: yellow,
                self.mp_pose.PoseLandmark.RIGHT_HIP: yellow
            },
            'backswing': {
                # Strong (right) arm green, weak arm and legs yellow
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER: green,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW: green,
                self.mp_pose.PoseLandmark.RIGHT_WRIST: green,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER: yellow,
                self.mp_pose.PoseLandmark.LEFT_ELBOW: yellow,
                self.mp_pose.PoseLandmark.LEFT_WRIST: yellow,
                self.mp_pose.PoseLandmark.LEFT_KNEE: yellow,
                self.mp_pose.PoseLandmark.RIGHT_KNEE: yellow
            },
            'contact': {
                # Strong arm green, weak arm yellow, legs red
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER: green,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW: green,
                self.mp_pose.PoseLandmark.RIGHT_WRIST: green,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER: yellow,
                self.mp_pose.PoseLandmark.LEFT_ELBOW: yellow,
                self.mp_pose.PoseLandmark.LEFT_WRIST: yellow,
                self.mp_pose.PoseLandmark.LEFT_KNEE: red,
                self.mp_pose.PoseLandmark.RIGHT_KNEE: red
            },
            'follow_through': {
                # Dominant arm red, everything else yellow
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER: red,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW: red,
                self.mp_pose.PoseLandmark.RIGHT_WRIST: red,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER: yellow,
                self.mp_pose.PoseLandmark.LEFT_ELBOW: yellow,
                self.mp_pose.PoseLandmark.LEFT_WRIST: yellow,
                self.mp_pose.PoseLandmark.LEFT_KNEE: yellow,
                self.mp_pose.PoseLandmark.RIGHT_KNEE: yellow
            }
        }

        return color_map.get(stage, {})

    def estimate_pose(self, image_path, stage='preparation'):
        """
        Estimate pose and draw with stage-specific coloring
        """
        # Print detailed debugging information
        print(f"Processing image: {image_path}")
        
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Check if image is loaded successfully
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Print image details for debugging
        print(f"Image shape: {image.shape}")
        
        # Convert image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find poses
        results = self.pose.process(image_rgb)
        
        # Check if any pose was detected
        if not results.pose_landmarks:
            print("No pose detected in the image.")
            return image
        
        # Get stage-specific color mapping
        color_map = self._get_limb_colors(stage)
        
        # Convert image back to BGR for OpenCV drawing
        image_bgr = image.copy()
        
        # Landmarks
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            # Convert landmark to image coordinates
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            # Choose color based on stage and landmark
            color = color_map.get(i, (200, 200, 200))
            
            # Draw the landmark
            cv2.circle(image_bgr, (cx, cy), 5, color, cv2.FILLED)
        
        # Connections
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]
            
            # Convert to image coordinates
            start_point = (
                int(start_landmark.x * image.shape[1]), 
                int(start_landmark.y * image.shape[0])
            )
            end_point = (
                int(end_landmark.x * image.shape[1]), 
                int(end_landmark.y * image.shape[0])
            )
            
            # Get connection color (blend start and end landmark colors)
            start_color = color_map.get(start_idx, (200, 200, 200))
            end_color = color_map.get(end_idx, (200, 200, 200))
            
            # Ensure color is a tuple of integers
            line_color = tuple(map(int, np.mean([start_color, end_color], axis=0)))
            
            # Draw the connection
            cv2.line(image_bgr, start_point, end_point, line_color, 2)
        
        return image_bgr

    def visualize_pose(self, image_path, stage='preparation'):
        """
        Visualize the pose estimation
        """
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct full path
        full_image_path = os.path.join(script_dir, image_path)
        
        # Estimate and draw pose
        result_image = self.estimate_pose(full_image_path, stage)
        
        # Display the image
        cv2.imshow(f'Tennis Pose Estimation - {stage.capitalize()} Stage', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Create the pose estimator
    estimator = TennisPoseEstimator()
    
    # Path to your tennis player image (in the same directory)
    image_path = 'followThrough.png'
    
    # Demonstrate different stages
    stages = ['preparation', 'backswing', 'contact', 'follow_through']
    
    try:
        for stage in stages:
            print(f"Processing stage: {stage}")
            estimator.visualize_pose(image_path, stage)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()