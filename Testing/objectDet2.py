import cv2
from ultralytics import YOLO
import time
# Use the 2nd smallest possible model
model = YOLO("yolov8n.pt")
# Access webcam
cap = cv2.VideoCapture(0)
# Use even lower resolution for processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# Set display size
display_width = 1200
display_height = 720
# Frame counter
frame_count = 0
last_boxes = []
# Process fewer frames for better performance
process_frequency = 3  # Only process every 5th frame
# Initialize time tracking for FPS counter
prev_time = time.time()
fps = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    # Calculate FPS
    current_time = time.time()
    if current_time - prev_time > 0:
        fps = 1/(current_time - prev_time)
    prev_time = current_time
    
    # Process only every Nth frame
    frame_count += 1
    process_this_frame = (frame_count % process_frequency == 0)
    if process_this_frame:
        # Use half resolution for even faster inference
        small_frame = cv2.resize(frame, (160, 120))
        # Only detect tennis rackets with higher confidence threshold
        results = model(small_frame, conf=0.35, classes=[38], iou=0.45)
        # Update last known boxes (scale coordinates back to original frame size)
        scale_x = frame.shape[1] / small_frame.shape[1]
        scale_y = frame.shape[0] / small_frame.shape[0]
        last_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                # Scale back to original frame size
                last_boxes.append((x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y, conf))
    # Draw bounding boxes
    for box in last_boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Tennis Racket: {conf:.2f}", 
                   (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Add FPS counter to frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Resize the display frame
    display_frame = cv2.resize(frame, (display_width, display_height))
    # Display the larger frame
    cv2.imshow("Tennis Racket Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()