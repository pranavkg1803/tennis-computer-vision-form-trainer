import cv2

cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
else: print("Could open video")
exit()
