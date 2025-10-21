from ultralytics import YOLO
import cv2

model = YOLO('yolov11s.pt')

cap = cv2.VideoCapture('video_calle.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # inferencia
    annotated_frame = results[0].plot()  # dibuja bounding boxes

    cv2.imshow("YOLOv11", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()