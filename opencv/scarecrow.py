import cv2
from ultralytics import YOLO
import serial
import time

unwanted_animals = ["bird", "duck", "goose", "eagle", "raccoon", "deer", "cat", "dog"]

arduino = serial.Serial('/dev/cu.usbserial-1140', 9600)
time.sleep(2)

model = YOLO("yolov8x.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]

    if any(label in unwanted_animals for label in labels):
        # print("Animal detected:", labels)
        arduino.write(b'B')
    # elif "person" in labels:
    #     print("Human detected")
    # else:
    #     print("Nothing dangerous detected")

    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()