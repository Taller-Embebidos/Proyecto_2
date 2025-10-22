from ultralytics import YOLO
import cv2
import time

# Cargar modelo liviano
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("video_test.mp4")


start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reducir resolución
    frame = cv2.resize(frame, (640, 360))

    # Inferencia
    results = model(frame, imgsz=640, device="cpu", verbose=False)

    # Dibujar bounding boxes
    annotated_frame = results[0].plot()

    time.sleep(1/15)
    cv2.imshow("YOLO11n", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()

end = time.time()
print(f"Procesamiento completo en {end - start:.2f} s")