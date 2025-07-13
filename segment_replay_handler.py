import cv2
import numpy as np
import mediapipe as mp

class SegmentReplayHandler:
    """
    Helper class to manage the replaying of problem segments from the reference video.
    This handles the frame extraction, pose estimation, and visualization.
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.start_frame = None
        self.end_frame = None
        self.current_frame = None
        self.replay_active = False
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def start_replay(self, start_frame, end_frame):
        """Start replaying a specific segment of the video"""
        # Close any existing capture
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.current_frame = start_frame
        
        # Set to starting frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.replay_active = True
        
        return True
    
    def get_next_frame(self, highlight_joints=None):
        """
        Get the next frame in the replay sequence
        
        Args:
            highlight_joints (dict): Optional dict mapping joint names to colors for highlighting
        
        Returns:
            tuple: (success, processed_image, frame_index)
        """
        if not self.replay_active or self.cap is None:
            return False, None, None
        
        ret, frame = self.cap.read()
        
        if not ret or self.current_frame >= self.end_frame:
            # Loop back to beginning of segment
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.current_frame = self.start_frame
            ret, frame = self.cap.read()
            
            if not ret:
                self.stop_replay()
                return False, None, None
        
        # Process frame with pose estimation
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Basic pose landmark drawing
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            
            # Additional highlighting for problem joints if provided
            if highlight_joints and results.pose_landmarks:
                self.highlight_problem_joints(image, results.pose_landmarks, highlight_joints)
        
        # Add frame indicator
        cv2.putText(
            image,
            f"PROBLEM SEGMENT - Frame {self.current_frame}/{self.end_frame}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        
        frame_index = self.current_frame
        self.current_frame += 1
        
        return True, image, frame_index
    
    def highlight_problem_joints(self, image, landmarks, highlight_joints):
        """
        Highlight specific problem joints in the pose
        
        Args:
            image: OpenCV image
            landmarks: MediaPipe pose landmarks
            highlight_joints: Dict mapping joint names to colors for highlighting
        """
        # Define mapping from joint names to MediaPipe pose landmark indices
        joint_to_landmark = {
            "rShoulder": self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            "rElbow": self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            "rWrist": self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            "rHip": self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            "rKnee": self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            "rAnkle": self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            "lShoulder": self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            "lElbow": self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            "lWrist": self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            "lHip": self.mp_pose.PoseLandmark.LEFT_HIP.value,
            "lKnee": self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            "lAnkle": self.mp_pose.PoseLandmark.LEFT_ANKLE.value
        }
        
        h, w, c = image.shape
        
        for joint_name, color in highlight_joints.items():
            if joint_name in joint_to_landmark:
                idx = joint_to_landmark[joint_name]
                landmark = landmarks.landmark[idx]
                
                # Draw highlighted circle around joint
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (cx, cy), 15, color, 3)
                
                # Add label
                cv2.putText(
                    image,
                    joint_name,
                    (cx + 15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA
                )
    
    def stop_replay(self):
        """Stop the replay and release resources"""
        self.replay_active = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        return True
    
    def is_active(self):
        """Check if replay is currently active"""
        return self.replay_active
    
    def __del__(self):
        """Cleanup resources on deletion"""
        if self.cap is not None:
            self.cap.release()
        
        # Release the pose processor
        self.pose.close()