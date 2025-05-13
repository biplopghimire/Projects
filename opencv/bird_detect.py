import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]

    if "person" in labels:
        print("Human detected")
    elif any(label in ["bird"] for label in labels):
        print("Bird detected")
    else:
        print("Neither human nor bird detected")

    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()