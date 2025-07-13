import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video capture
cap = cv2.VideoCapture('video.mp4')

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables
old_gray = None
p0 = None

# Function to manually select the racket tip
def select_point(event, x, y, flags, param):
    global p0, old_gray
    if event == cv2.EVENT_LBUTTONDOWN:
        p0 = np.array([[[x, y]]], dtype=np.float32)
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Set up mouse callback to select the racket tip
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', select_point)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pose estimation
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist_point = np.array([wrist.x * frame.shape[1], wrist.y * frame.shape[0]], dtype=np.float32)
        elbow_point = np.array([elbow.x * frame.shape[1], elbow.y * frame.shape[0]], dtype=np.float32)

        if p0 is not None:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            if st[0] == 1:
                racket_tip = p1[0].ravel()
                cv2.circle(frame, (racket_tip[0], racket_tip[1]), 5, (0, 0, 255), -1)

                # Calculate angle
                vector1 = wrist_point - elbow_point
                vector2 = racket_tip - wrist_point
                angle = np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0])
                angle = np.degrees(angle)
                if angle < 0:
                    angle += 360

                cv2.putText(frame, f'Angle: {angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = p1

    cv2.imshow('frame', frame)
    key = cv2.waitKey(10) & 0xFF  
    if key == ord('q'): #we can press q to quit the camera feed
        break

cap.release()
cv2.destroyAllWindows()