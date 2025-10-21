import cv2
from semaforo_fedora40 import detect_on_frame, labels, VIDEO_PATH_VEHICULOS

cap = cv2.VideoCapture(VIDEO_PATH_VEHICULOS)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    dets = detect_on_frame(frame)
    veh_count = sum(1 for d in dets if "car" in labels.get(d["class_id"], "").lower())
    print(f"Vehículos detectados: {veh_count}")
cap.release()