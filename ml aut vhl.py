import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video or webcam
cap = cv2.VideoCapture(0)  # or use "video.mp4"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    vehicle_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["car", "truck", "bus", "motorcycle"]:
                vehicle_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Headlight Logic
    if vehicle_detected:
        text = "LOW BEAM"
        color = (0,0,255)
    else:
        text = "HIGH BEAM"
        color = (0,255,0)

    cv2.putText(frame, text, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Smart Headlight Control", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()