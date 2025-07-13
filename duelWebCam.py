import cv2
import mediapipe as mp
import numpy as np
import time
import moviepy as mvp
import dtaidistance
from collections import defaultdict
import similarity_metrics as sm
import formVisualisation as fv
import chainSimilarity as cs

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
BRIGHT_BLUE = "\033[94m"
RESET = "\033[0m"

# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_vector(a, b):
    """
    Computes the 2D vector from point A to point B.
    """
    a = np.array(a)
    b = np.array(b)
    return b - a 

# Function to annotate angle
def annotate_angle(coord_name, angle, image):
    cv2.putText(image, str(angle),
                tuple(np.multiply(coord_name, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

class PersonAngles:
    """
    A class to track and store joint angles during pose estimation.
    """
    def __init__(self):
        self.angles = {
            "lShoulderAngles": [], "lElbowAngles": [], "lHipAngles": [],
            "lKneeAngles": [], "lAnkleAngles": [], "lFootAngles": [],
            "rShoulderAngles": [], "rElbowAngles": [], "rHipAngles": [],
            "rKneeAngles": [], "rAnkleAngles": [], "rFootAngles": [],
            "lUpperArmVector": [], "lUpperLegVector": [], "rUpperArmVector": [],
            "rUpperLegVector": []
        }

    def add_angle(self, joint, angle):
        """
        Add an angle measurement for a specific joint.
        
        Args:
            joint (str): Name of the joint
            angle (float): Angle measurement
        """
        if joint in self.angles:
            self.angles[joint].append(angle)
        else:
            raise ValueError(f"Invalid joint name: {joint}")

    def get_angles(self, joint):
        
        return self.angles.get(joint, [])
    
    def __str__(self):
        
        angle_strs = []
        for joint, angles in self.angles.items():
            angle_str = f"\033[94m{joint}\033[0m: {angles}"
            angle_strs.append(angle_str)
        return "\n".join(angle_strs)

def get_video_duration(video_path):
    """Get duration of video in seconds"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration

def calculate_effective_video_duration(original_duration, original_fps, sampling_interval):
    """
    Calculate the effective video duration when sampling frames at a specific interval.
    
    Returns:
        float: Effective duration of the video when sampled
    """

    newFPS = 1 / sampling_interval
    effective_duration = original_duration * (original_fps / newFPS)
    return effective_duration 

def store_reference_video(video_path):
    """
    Process reference video with pose estimation and draw landmarks.
    
    Args:
        video_path (str): Path to the input video
    
    Returns:
        str: Path to the processed video
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_path = 'processed_reference_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
            
            out.write(image)
    
    cap.release()
    out.release()
    
    return output_path

def analyzeVideo(person_angles, video_path):
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        recording = False
        start_time = None
        time_interval = 0.05
            
                
        next_sample_time = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
            
            try:
                landmarks = results.pose_landmarks.landmark
            except Exception as e:
                landmarks = None
                print(f"Error occurred1: {e}")
            
            if landmarks is not None:
                # Extract joint coordinates for calculating angles
                lShoulderAngle = None
                lElbowAngle = None
                lHipAngle = None
                lKneeAngle = None
                lAnkleAngle = None
                rShoulderAngle = None
                rElbowAngle = None
                rHipAngle = None
                rKneeAngle = None
                rAnkleAngle = None
                
                lShoulder = None
                lElbow = None
                lWrist = None
                lHip = None
                lKnee = None
                lAnkle = None
                lFoot = None
                rShoulder = None
                rElbow = None
                rWrist = None
                rHip = None
                rKnee = None
                rAnkle = None
                rFoot = None
                confidenceVal = 0.0
                lShoulder = ([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > confidenceVal else None)
                lElbow = ([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > confidenceVal else None)
                lWrist = ([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > confidenceVal else None)
                lHip = ([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > confidenceVal else None)
                lKnee = ([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > confidenceVal else None)
                lAnkle = ([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > confidenceVal else None)
                lFoot = ([landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility > confidenceVal else None)
                rShoulder = ([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > confidenceVal else None)
                rElbow = ([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > confidenceVal else None)
                rWrist = ([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > confidenceVal else None)
                rHip = ([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > confidenceVal else None)
                rKnee = ([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > confidenceVal else None)
                rAnkle = ([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > confidenceVal else None)
                rFoot = ([landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility > confidenceVal else None)
                #print("Shoulder coords:", lShoulder)
                # Calculate angles for all joints
                
                lShoulderAngle = round(calculate_angle(lHip, lShoulder, lElbow), 3) if None not in [lHip, lShoulder, lElbow] else None
                lElbowAngle = round(calculate_angle(lShoulder, lElbow, lWrist), 3) if None not in [lShoulder, lElbow, lWrist] else None
                lHipAngle = round(calculate_angle(lKnee, lHip, lShoulder), 3) if None not in [lKnee, lHip, lShoulder] else None
                lKneeAngle = round(calculate_angle(lAnkle, lKnee, lHip), 3) if None not in [lAnkle, lKnee, lHip] else None
                lAnkleAngle = round(calculate_angle(lFoot, lAnkle, lKnee), 3) if None not in [lFoot, lAnkle, lKnee] else None
                rShoulderAngle = round(calculate_angle(rHip, rShoulder, rElbow), 3) if None not in [rHip, rShoulder, rElbow] else None
                rElbowAngle = round(calculate_angle(rShoulder, rElbow, rWrist), 3) if None not in [rShoulder, rElbow, rWrist] else None
                rHipAngle = round(calculate_angle(rKnee, rHip, rShoulder), 3) if None not in [rKnee, rHip, rShoulder] else None
                rKneeAngle = round(calculate_angle(rAnkle, rKnee, rHip), 3) if None not in [rAnkle, rKnee, rHip] else None
                rAnkleAngle = round(calculate_angle(rFoot, rAnkle, rKnee), 3) if None not in [rFoot, rAnkle, rKnee] else None
                
                lUpperArmVector = calculate_vector(lShoulder, lElbow) if None not in [lShoulder, lElbow] else None
                lUpperLegVector = calculate_vector(lElbow, lWrist) if None not in [lElbow, lWrist] else None
                rUpperArmVector = calculate_vector(rShoulder, rElbow) if None not in [rShoulder, rElbow] else None
                rUpperLegVector = calculate_vector(rElbow, rWrist) if None not in [rElbow, rWrist] else None
                
                

                # Annotate all angles on the image
                annotate_angle(lShoulder, lShoulderAngle, image)
                annotate_angle(lElbow, lElbowAngle, image)
                annotate_angle(lHip, lHipAngle, image)
                annotate_angle(lKnee, lKneeAngle, image)
                annotate_angle(lAnkle, lAnkleAngle, image)
                annotate_angle(rShoulder, rShoulderAngle, image)
                annotate_angle(rElbow, rElbowAngle, image)
                annotate_angle(rHip, rHipAngle, image)
                annotate_angle(rKnee, rKneeAngle, image)
                annotate_angle(rAnkle, rAnkleAngle, image)
            
                

            # Draw the pose landmarks and skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            

            # Display the image with annotations
            cv2.imshow('Feed', image)

            
            
            # Handle key presses
            key = cv2.waitKey(50) & 0xFF
            # Video mode: Start recording immediately
            if recording is False:
                recording = True
                start_time = time.time()
                next_sample_time = start_time + time_interval
            if recording:
                current_time = time.time()
                if current_time >= next_sample_time:
                    next_sample_time += time_interval
                    # we are storing the angles while the video is displaying, only every sample interval
                    person_angles.add_angle("lShoulderAngles", lShoulderAngle)
                    person_angles.add_angle("lElbowAngles", lElbowAngle)
                    person_angles.add_angle("lHipAngles", lHipAngle)
                    person_angles.add_angle("lKneeAngles", lKneeAngle)
                    person_angles.add_angle("lAnkleAngles", lAnkleAngle)
                    person_angles.add_angle("rShoulderAngles", rShoulderAngle)
                    person_angles.add_angle("rElbowAngles", rElbowAngle)
                    person_angles.add_angle("rHipAngles", rHipAngle)
                    person_angles.add_angle("rKneeAngles", rKneeAngle)
                    person_angles.add_angle("rAnkleAngles", rAnkleAngle)
                    person_angles.add_angle("lUpperArmVector", lUpperArmVector)
                    person_angles.add_angle("lUpperLegVector", lUpperLegVector)
                    person_angles.add_angle("rUpperArmVector", rUpperArmVector)
                    person_angles.add_angle("rUpperLegVector", rUpperLegVector)
                    
            if key == ord('q'):
                    print('q has been pressed')
                    cap.release()
                    cv2.destroyAllWindows()
                    break  # Press 'q' to exit
        cap.release()
        cv2.destroyAllWindows()


def record_pose_landmarks(input_video_path):
    """
    Record pose landmarks from the reference video.
    
    Args:
        input_video_path (str): Path to the input video file
    
    Returns:
        list: Sequence of pose landmarks for each frame
    """
    cap = cv2.VideoCapture(input_video_path)
    pose_landmarks_sequence = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            pose_landmarks_sequence.append(results.pose_landmarks)
    
    cap.release()
    return pose_landmarks_sequence

def transform_landmarks(reference_landmarks, webcam_landmarks):
    """
    Transform reference video landmarks to match the webcam pose.
    
    Args:
        reference_landmarks: Pose landmarks from the reference video
        webcam_landmarks: Current webcam pose landmarks
    
    Returns:
        tuple: Transformation parameters or None if detection fails
    """
    if not reference_landmarks or not webcam_landmarks:
        return None

    key_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    ref_points = [
        (reference_landmarks.landmark[lm].x, reference_landmarks.landmark[lm].y) 
        for lm in key_landmarks
    ]
    
    webcam_points = [
        (webcam_landmarks.landmark[lm].x, webcam_landmarks.landmark[lm].y) 
        for lm in key_landmarks
    ]

    ref_width = abs(ref_points[0][0] - ref_points[1][0])
    ref_height = abs(ref_points[2][1] - ref_points[0][1])
    ref_center_x = sum(p[0] for p in ref_points) / len(ref_points)
    ref_center_y = sum(p[1] for p in ref_points) / len(ref_points)

    webcam_width = abs(webcam_points[0][0] - webcam_points[1][0])
    webcam_height = abs(webcam_points[2][1] - webcam_points[0][1])
    webcam_center_x = sum(p[0] for p in webcam_points) / len(webcam_points)
    webcam_center_y = sum(p[1] for p in webcam_points) / len(webcam_points)

    width_scale = webcam_width / ref_width if ref_width != 0 else 1
    height_scale = webcam_height / ref_height if ref_height != 0 else 1
    scale_factor = min(width_scale, height_scale)

    x_offset = webcam_center_x - (ref_center_x * scale_factor)
    y_offset = webcam_center_y - (ref_center_y * scale_factor)

    return x_offset, y_offset, scale_factor, ref_center_x, ref_center_y, webcam_center_x, webcam_center_y



def transform_landmarks_basic(reference_landmarks, webcam_landmarks):
    """
    Basic transformation of reference video landmarks to match the webcam pose.
    
    This method provides a simple translation of landmarks without scaling,
    aligning the center of the reference pose with the center of the webcam pose.
    
    Args:
        reference_landmarks: Pose landmarks from the reference video
        webcam_landmarks: Current webcam pose landmarks
    
    Returns:
        tuple: Translation parameters or None if detection fails
    """
    if not reference_landmarks or not webcam_landmarks:
        return None

    key_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    # Calculate reference pose center
    ref_points = [
        (reference_landmarks.landmark[lm].x, reference_landmarks.landmark[lm].y) 
        for lm in key_landmarks
    ]
    ref_center_x = sum(p[0] for p in ref_points) / len(ref_points)
    ref_center_y = sum(p[1] for p in ref_points) / len(ref_points)

    # Calculate webcam pose center
    webcam_points = [
        (webcam_landmarks.landmark[lm].x, webcam_landmarks.landmark[lm].y) 
        for lm in key_landmarks
    ]
    webcam_center_x = sum(p[0] for p in webcam_points) / len(webcam_points)
    webcam_center_y = sum(p[1] for p in webcam_points) / len(webcam_points)

    # Calculate translation offsets
    x_offset = webcam_center_x - ref_center_x
    y_offset = webcam_center_y - ref_center_y

    return x_offset, y_offset, ref_center_x, ref_center_y, webcam_center_x, webcam_center_y

def analyze_video_and_webcam(person_angles, processed_video_path, reference_pose_landmarks,
                            transform_method='scaled', frame_callback=None, recording_callback=None,
                            completion_callback=None, should_run_callback=None, video_duration=10):
    """
    Analyze video with webcam and overlay reference pose skeleton.
    
    Args:
        person_angles (PersonAngles): Object to store angle measurements
        processed_video_path (str): Path to processed reference video
        reference_pose_landmarks (list): Sequence of reference video pose landmarks
        transform_method (str): Transformation method to use. 
                                Options: 'scaled' (default) or 'basic'
    """
    
    
    
    
    cap_webcam = cv2.VideoCapture(0)
    cap_video = cv2.VideoCapture(processed_video_path)
    total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        
        recording = False
        start_time = None
        time_interval = 0.05
        next_sample_time = None
        skip_count = 2
        thickness = 4
        
        while cap_webcam.isOpened() and cap_video.isOpened():
            #new code here. might break
            
            if should_run_callback and not should_run_callback(): #when we press stop, we break out of loop
                break
            
            if not cap_video.get(cv2.CAP_PROP_POS_FRAMES) < cap_video.get(cv2.CAP_PROP_FRAME_COUNT):
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0  #this indicates we are resetting we put frame count back to 0, as we loop back to start of video

            ret_video, frame_video = cap_video.read() #we get the next frame of the video
            ret_webcam, frame_webcam = cap_webcam.read()
            
            if not ret_webcam or not ret_video:
                break

            image_webcam = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
            image_webcam.flags.writeable = False
            results_webcam = pose.process(image_webcam)
            image_webcam.flags.writeable = True
            image_webcam = cv2.cvtColor(image_webcam, cv2.COLOR_RGB2BGR)

            if (frame_count < len(reference_pose_landmarks) and 
                reference_pose_landmarks[frame_count] is not None and 
                results_webcam.pose_landmarks is not None):
                
                # Choose transformation method
                if transform_method == 'scaled':
                    transform = transform_landmarks(
                        reference_pose_landmarks[frame_count], 
                        results_webcam.pose_landmarks
                    )
                    if transform:
                        x_offset, y_offset, scale_factor, ref_center_x, ref_center_y, webcam_center_x, webcam_center_y = transform
                        
                        transformed_landmarks = reference_pose_landmarks[frame_count]
                        
                        for landmark in transformed_landmarks.landmark:
                            landmark.x = (landmark.x - ref_center_x) * scale_factor + webcam_center_x
                            landmark.y = (landmark.y - ref_center_y) * scale_factor + webcam_center_y
                
                else:  # basic transformation
                    transform = transform_landmarks_basic(
                        reference_pose_landmarks[frame_count], 
                        results_webcam.pose_landmarks
                    )
                    if transform:
                        x_offset, y_offset, ref_center_x, ref_center_y, webcam_center_x, webcam_center_y = transform
                        
                        transformed_landmarks = reference_pose_landmarks[frame_count]
                        
                        for landmark in transformed_landmarks.landmark:
                            landmark.x += x_offset
                            landmark.y += y_offset
                
                # Draw transformed landmarks
                mp_drawing.draw_landmarks(
                    image_webcam, 
                    transformed_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=thickness, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=thickness, circle_radius=2)
                )

            # Draw current webcam pose landmarks (skeleton and points)
            if results_webcam.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_webcam, 
                    results_webcam.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=thickness, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=thickness, circle_radius=2)
                )
              
            try:
                landmarks = results_webcam.pose_landmarks.landmark
            except Exception as e:
                landmarks = None
                print(f"Error occurred2: {e}")
            
            if landmarks is not None:
                # Extract joint coordinates for calculating angles
                
                confidenceVal = 0.0
                lShoulder = ([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > confidenceVal else None)
                lElbow = ([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > confidenceVal else None)
                lWrist = ([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > confidenceVal else None)
                lHip = ([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > confidenceVal else None)
                lKnee = ([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > confidenceVal else None)
                lAnkle = ([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > confidenceVal else None)
                lFoot = ([landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                    if landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility > confidenceVal else None)
                rShoulder = ([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > confidenceVal else None)
                rElbow = ([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > confidenceVal else None)
                rWrist = ([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > confidenceVal else None)
                rHip = ([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > confidenceVal else None)
                rKnee = ([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > confidenceVal else None)
                rAnkle = ([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > confidenceVal else None)
                rFoot = ([landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                    if landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility > confidenceVal else None)
                #print("Shoulder coords:", lShoulder)
                # Calculate angles for all joints
                
                lShoulderAngle = round(calculate_angle(lHip, lShoulder, lElbow), 3) if None not in [lHip, lShoulder, lElbow] else None
                lElbowAngle = round(calculate_angle(lShoulder, lElbow, lWrist), 3) if None not in [lShoulder, lElbow, lWrist] else None
                lHipAngle = round(calculate_angle(lKnee, lHip, lShoulder), 3) if None not in [lKnee, lHip, lShoulder] else None
                lKneeAngle = round(calculate_angle(lAnkle, lKnee, lHip), 3) if None not in [lAnkle, lKnee, lHip] else None
                lAnkleAngle = round(calculate_angle(lFoot, lAnkle, lKnee), 3) if None not in [lFoot, lAnkle, lKnee] else None
                rShoulderAngle = round(calculate_angle(rHip, rShoulder, rElbow), 3) if None not in [rHip, rShoulder, rElbow] else None
                rElbowAngle = round(calculate_angle(rShoulder, rElbow, rWrist), 3) if None not in [rShoulder, rElbow, rWrist] else None
                rHipAngle = round(calculate_angle(rKnee, rHip, rShoulder), 3) if None not in [rKnee, rHip, rShoulder] else None
                rKneeAngle = round(calculate_angle(rAnkle, rKnee, rHip), 3) if None not in [rAnkle, rKnee, rHip] else None
                rAnkleAngle = round(calculate_angle(rFoot, rAnkle, rKnee), 3) if None not in [rFoot, rAnkle, rKnee] else None

                lUpperArmVector = calculate_vector(lShoulder, lElbow) if None not in [lShoulder, lElbow] else None
                lUpperLegVector = calculate_vector(lElbow, lWrist) if None not in [lElbow, lWrist] else None
                rUpperArmVector = calculate_vector(rShoulder, rElbow) if None not in [rShoulder, rElbow] else None
                rUpperLegVector = calculate_vector(rElbow, rWrist) if None not in [rElbow, rWrist] else None
                
                

                # Annotate all angles on the image
                annotate_angle(lShoulder, lShoulderAngle, image_webcam)
                annotate_angle(lElbow, lElbowAngle, image_webcam)
                annotate_angle(lHip, lHipAngle, image_webcam)
                annotate_angle(lKnee, lKneeAngle, image_webcam)
                annotate_angle(lAnkle, lAnkleAngle, image_webcam)
                annotate_angle(rShoulder, rShoulderAngle, image_webcam)
                annotate_angle(rElbow, rElbowAngle, image_webcam)
                annotate_angle(rHip, rHipAngle, image_webcam)
                annotate_angle(rKnee, rKneeAngle, image_webcam)
                annotate_angle(rAnkle, rAnkleAngle, image_webcam)
            
                #a default dictionary is used here for saftey
                currFrameAngleVectorMap = defaultdict(lambda: None, {
                    # Angles
                    "lShoulderAngle": lShoulderAngle,
                    "lElbowAngle": lElbowAngle,
                    "lHipAngle": lHipAngle,
                    "lKneeAngle": lKneeAngle,
                    "lAnkleAngle": lAnkleAngle,
                    "rShoulderAngle": rShoulderAngle,
                    "rElbowAngle": rElbowAngle,
                    "rHipAngle": rHipAngle,
                    "rKneeAngle": rKneeAngle,
                    "rAnkleAngle": rAnkleAngle,
                    
                    # Coordinates
                    "lShoulder": lShoulder,
                    "rShoulder": rShoulder,
                    "lElbow": lElbow,
                    "rElbow": rElbow,
                    "lHip": lHip,
                    "rHip": rHip,
                    "lKnee": lKnee,
                    "rKnee": rKnee,
                    "lAnkle": lAnkle,
                    "rAnkle": rAnkle,
                    
                    # Vectors
                    "lUpperArmVector": lUpperArmVector,
                    "lUpperLegVector": lUpperLegVector,
                    "rUpperArmVector": rUpperArmVector,
                    "rUpperLegVector": rUpperLegVector
                })
                
                #extracting data for the pro (again becasue we need to do it for this frame specifically)
                try:
                    landmarksPro = transformed_landmarks.landmark
                except Exception as e:
                    landmarksPro = None
                    print(f"Error occurred with pro landmarks: {e}")

                if landmarksPro is not None:
                    # Extract joint coordinates for calculating angles
                    confidenceVal = 0.0
                    pro_lShoulder = ([landmarksPro[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarksPro[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > confidenceVal else None)
                    pro_lElbow = ([landmarksPro[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarksPro[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > confidenceVal else None)
                    pro_lWrist = ([landmarksPro[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarksPro[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > confidenceVal else None)
                    pro_lHip = ([landmarksPro[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarksPro[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > confidenceVal else None)
                    pro_lKnee = ([landmarksPro[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarksPro[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > confidenceVal else None)
                    pro_lAnkle = ([landmarksPro[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarksPro[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > confidenceVal else None)
                    pro_lFoot = ([landmarksPro[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarksPro[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility > confidenceVal else None)
                    pro_rShoulder = ([landmarksPro[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarksPro[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > confidenceVal else None)
                    pro_rElbow = ([landmarksPro[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarksPro[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > confidenceVal else None)
                    pro_rWrist = ([landmarksPro[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarksPro[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > confidenceVal else None)
                    pro_rHip = ([landmarksPro[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarksPro[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > confidenceVal else None)
                    pro_rKnee = ([landmarksPro[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarksPro[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > confidenceVal else None)
                    pro_rAnkle = ([landmarksPro[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarksPro[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > confidenceVal else None)
                    pro_rFoot = ([landmarksPro[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarksPro[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                        if landmarksPro[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility > confidenceVal else None)

                    # Calculate angles for all joints
                    pro_current_frame = defaultdict(lambda: None, {
                        "lShoulderAngle": round(calculate_angle(pro_lHip, pro_lShoulder, pro_lElbow), 3) if None not in [pro_lHip, pro_lShoulder, pro_lElbow] else None,
                        "lElbowAngle": round(calculate_angle(pro_lShoulder, pro_lElbow, pro_lWrist), 3) if None not in [pro_lShoulder, pro_lElbow, pro_lWrist] else None,
                        "lHipAngle": round(calculate_angle(pro_lKnee, pro_lHip, pro_lShoulder), 3) if None not in [pro_lKnee, pro_lHip, pro_lShoulder] else None,
                        "lKneeAngle": round(calculate_angle(pro_lAnkle, pro_lKnee, pro_lHip), 3) if None not in [pro_lAnkle, pro_lKnee, pro_lHip] else None,
                        "lAnkleAngle": round(calculate_angle(pro_lFoot, pro_lAnkle, pro_lKnee), 3) if None not in [pro_lFoot, pro_lAnkle, pro_lKnee] else None,
                        "rShoulderAngle": round(calculate_angle(pro_rHip, pro_rShoulder, pro_rElbow), 3) if None not in [pro_rHip, pro_rShoulder, pro_rElbow] else None,
                        "rElbowAngle": round(calculate_angle(pro_rShoulder, pro_rElbow, pro_rWrist), 3) if None not in [pro_rShoulder, pro_rElbow, pro_rWrist] else None,
                        "rHipAngle": round(calculate_angle(pro_rKnee, pro_rHip, pro_rShoulder), 3) if None not in [pro_rKnee, pro_rHip, pro_rShoulder] else None,
                        "rKneeAngle": round(calculate_angle(pro_rAnkle, pro_rKnee, pro_rHip), 3) if None not in [pro_rAnkle, pro_rKnee, pro_rHip] else None,
                        "rAnkleAngle": round(calculate_angle(pro_rFoot, pro_rAnkle, pro_rKnee), 3) if None not in [pro_rFoot, pro_rAnkle, pro_rKnee] else None,
                        
                        # Vectors
                        "lUpperArmVector": calculate_vector(pro_lShoulder, pro_lElbow) if None not in [pro_lShoulder, pro_lElbow] else None,
                        "lUpperLegVector": calculate_vector(pro_lElbow, pro_lWrist) if None not in [pro_lElbow, pro_lWrist] else None,
                        "rUpperArmVector": calculate_vector(pro_rShoulder, pro_rElbow) if None not in [pro_rShoulder, pro_rElbow] else None,
                        "rUpperLegVector": calculate_vector(pro_rElbow, pro_rWrist) if None not in [pro_rElbow, pro_rWrist] else None
                    })
                
                #Now we will add the visuilisation feedback graphics
                
                image_webcam = fv.visualize_pose_feedback2(
                    image_webcam,
                    currFrameAngleVectorMap,  # current user's angles
                    pro_current_frame,           # reference pro angles
                    threshold=15,
                    
                )
                
            
                

            #This displays the image
            if frame_callback:
                #when the gui is used, send frames to GUI
                frame_callback(frame_video, image_webcam)
            else: #if not using GUI we visualise as usual
                image_video_resized = cv2.resize(frame_video, (800, 600))
                image_webcam_resized = cv2.resize(image_webcam, (800, 600))
                combined_image = np.hstack((image_video_resized, image_webcam_resized))
                cv2.imshow('Reference | Webcam', combined_image)

            key = cv2.waitKey(50) & 0xFF
            current_time = time.time()
            
            if (key == ord('w') and not recording) or (recording_callback and not recording and recording_callback()): #the callback activates when the the GUI button is pressed
                print('w has been pressed')
                #when recording we simply reset back to frame 0
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                recording = True
                start_time = time.time()
                next_sample_time = start_time + time_interval
                if recording_callback:
                    recording_callback()
                
            if recording:
                print(f"Recording: {current_time - start_time:.2f} / {5} seconds") 
                
                if current_time >= next_sample_time:
                    next_sample_time += time_interval
                    #print(f"Before storing - lShoulderAngle: {lShoulderAngle}")
                    person_angles.add_angle("lShoulderAngles", lShoulderAngle)
                    # Print the actual list after adding
                    #print(f"Actual stored angles: {person_angles.angles['lShoulderAngles']}")
                    person_angles.add_angle("lElbowAngles", lElbowAngle)
                    person_angles.add_angle("lHipAngles", lHipAngle)
                    person_angles.add_angle("lKneeAngles", lKneeAngle)
                    person_angles.add_angle("lAnkleAngles", lAnkleAngle)
                    person_angles.add_angle("rShoulderAngles", rShoulderAngle)
                    person_angles.add_angle("rElbowAngles", rElbowAngle)
                    person_angles.add_angle("rHipAngles", rHipAngle)
                    person_angles.add_angle("rKneeAngles", rKneeAngle)
                    person_angles.add_angle("rAnkleAngles", rAnkleAngle)
                    person_angles.add_angle("lUpperArmVector", lUpperArmVector)
                    person_angles.add_angle("lUpperLegVector", lUpperLegVector)
                    person_angles.add_angle("rUpperArmVector", rUpperArmVector)
                    person_angles.add_angle("rUpperLegVector", rUpperLegVector)
                
            
                # Store all angles
            
            if recording and (current_time - start_time >= 15 or frame_count >= total_frames - 1): #we reach the last frame
                print(f"Recording complete: {current_time - start_time:.2f} seconds") 
                recording = False
                if completion_callback:
                    completion_callback()
            
            if key == ord('q'):
                    print('q has been pressed')
                    cap_webcam.release()
                    cap_video.release()
                    cv2.destroyAllWindows()
                    break  # Press 'q' to exit
            frame_count += 1

        cap_webcam.release()
        cap_video.release()
        cv2.destroyAllWindows()

def main():
    """
    Main execution function for pose estimation and visualization.
    """
    video_path = "federer.mp4"
    print("Analyzing video...")
    pro = PersonAngles()
    analyzeVideo(pro, video_path=video_path)
    
    #now we store the pro video with skeleton, as well as copy of skeleton
    processed_video_path = store_reference_video(video_path) #this video is of origonal length
    reference_pose_landmarks = record_pose_landmarks(video_path) #this is a copy of the skeleton
    
   
   
    
    # Choose transformation method: 'scaled' (default) or 'basic'
    # Process webcam analysis (waits for button press to start recording)
    print("Analyzing webcam...")
    user = PersonAngles()
    analyze_video_and_webcam(user, processed_video_path, reference_pose_landmarks, transform_method='basic') #'scaled' or 'basic'
    
    # Output results for both video and webcam angles
    print(f"{BRIGHT_BLUE}Video Recorded Angles:{RESET}", pro)
    print(f"{BRIGHT_BLUE}Webcam Recorded Angles:{RESET}", user)
    print(pro)
    path = dtaidistance.dtw.warping_path(pro.angles['rElbowAngles'], user.angles['rElbowAngles'])
    print(f"{BRIGHT_BLUE}Warping path for right elbow:{RESET}", path)
    similarity = sm.calculateSimilarity1(pro, user, path)
    
    print(f"{BRIGHT_BLUE}Similarity between pro and user:{RESET}", similarity)
    weightedSimilarity = sm.calculateSimilarityWeighted(pro, user, path)
    print(f"{BRIGHT_BLUE}Weighted Similarity between pro and user:{RESET}", weightedSimilarity)
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    