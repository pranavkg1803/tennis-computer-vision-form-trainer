import cv2
from ultralytics import YOLO

# Load a more powerful model
model = YOLO("yolov8m.pt")

# Access webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Run detection with lower confidence threshold
    results = model(frame, conf=0.25)
    
    # Draw ALL detections (not just tennis rackets)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            conf = box.conf[0].item()
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Use different colors for different classes
            color = (0, 255, 0) if cls == 43 else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{cls_name}: {conf:.2f}", 
                       (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display the frame with detections
    cv2.imshow("Object Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()