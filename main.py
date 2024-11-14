import cv2
from ultralytics import YOLO


# Load detection model
model = YOLO("yolo11n.pt")

# get rtsp video
rtsp_url = "http://192.168.18.210:8080/video"
cap = cv2.VideoCapture(rtsp_url)

# Perform object detection on each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    annotated_frame = results.plot()
    frame = cv2.rotate(annotated_frame,cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('image', cv2.resize(frame,(640,720)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Render the detections on the frame
cv2.destroyAllWindows()