#!/usr/bin/env python3
"""
semaforo_fedora40.py

Sistema de control de semáforo inteligente para Fedora 40.
 - Flujo 1: VIDEO pregrabado (vehículos)
 - Flujo 2: Cámara integrada (peatones y fauna)

Usa OpenCV + TensorFlow Lite para detección con modelos livianos.
"""

import cv2
import time
import threading
import numpy as np
import sys
from collections import deque
import os

# ---------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------
# Ruta al video (modifica según tu archivo)
VIDEO_PATH_VEHICULOS = "video_test.mp4"   # Ejemplo: ./videos/video_calle.mp4
CAM_PEATONES_FAUNA = 0                     # Índice de la cámara integrada (generalmente 0)
MODEL_PATH = "model.tflite"                # Modelo de detección TFLite
LABELS_PATH = "labels.txt"                 # Etiquetas
CONF_THRESHOLD = 0.1                       # Umbral de confianza
PERSISTENCE_FRAMES = 5                     # Frames consecutivos para confirmar detección
AMARILLO_SECONDS = 2.0                     # Duración del amarillo
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------------------------
# INTÉRPRETE TFLITE
# ---------------------------
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter

def load_labels(path):
    labels = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1]
                else:
                    labels[len(labels)] = line
    except FileNotFoundError:
        print(f"Aviso: no se encontró {path}. Las etiquetas se usarán por índice.")
    return labels

labels = load_labels(LABELS_PATH)
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
in_shape = input_details[0]['shape']
_, in_h, in_w, in_c = in_shape if len(in_shape) == 4 else (1, 300, 300, 3)

def preprocess(frame):
    img = cv2.resize(frame, (in_w, in_h))
    if input_details[0]['dtype'] == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)
    return np.expand_dims(img, axis=0)

def detect_on_frame(frame):
    img_in = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], img_in)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    detections = []
    h, w = frame.shape[:2]
    for i in range(len(scores)):
        if scores[i] < CONF_THRESHOLD:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        detections.append({
            "class_id": int(classes[i]),
            "score": float(scores[i]),
            "bbox": (x1, y1, x2, y2)
        })
    return detections

# ---------------------------
# WORKER DE CÁMARA / VIDEO
# ---------------------------
class VideoWorker(threading.Thread):
    """Procesa una fuente de video (archivo o cámara) en hilo separado."""
    def __init__(self, source, name="cam"):
        super().__init__()
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir fuente de video: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.running = True
        self.frame = None
        self.detections = []
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # Si es video, reiniciamos al final
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.1)
                    continue
            with self.lock:
                self.frame = frame.copy()
            try:
                dets = detect_on_frame(frame)
            except Exception as e:
                print(f"[{self.name}] Error detección: {e}")
                dets = []
            with self.lock:
                self.detections = dets
            time.sleep(0.02)

    def stop(self):
        self.running = False
        try:
            self.cap.release()
        except:
            pass

    def get_latest(self):
        with self.lock:
            return self.frame, list(self.detections)

# ---------------------------
# CLASIFICACIÓN POR TIPO
# ---------------------------
VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorbike", "bicycle"}
PEDESTRIAN_CLASS_NAMES = {"person"}
FAUNA_CLASS_NAMES = {"cat", "dog", "bird", "deer", "animal"}

def classify_counts(detections):
    veh = ped = fauna = 0
    for d in detections:
        cid = d.get("class_id", -1)
        name = labels.get(cid, str(cid)).lower()
        if any(v in name for v in VEHICLE_CLASS_NAMES):
            veh += 1
        elif any(p in name for p in PEDESTRIAN_CLASS_NAMES):
            ped += 1
        elif any(f in name for f in FAUNA_CLASS_NAMES):
            fauna += 1
    return veh, ped, fauna

# ---------------------------
# CONTROL DEL SEMÁFORO
# ---------------------------
estado_semaforo = "VERDE"
_last_transition_time = time.time()
veh_hist, ped_hist, fauna_hist = deque(maxlen=PERSISTENCE_FRAMES), deque(maxlen=PERSISTENCE_FRAMES), deque(maxlen=PERSISTENCE_FRAMES)

def decide_semaforo(count_veh, count_ped, count_fauna):
    global estado_semaforo, _last_transition_time
    peatones_present = count_ped > 0
    fauna_present = count_fauna > 0
    if peatones_present or fauna_present:
        if estado_semaforo == "VERDE":
            estado_semaforo = "AMARILLO"
            _last_transition_time = time.time()
        elif estado_semaforo == "AMARILLO" and (time.time() - _last_transition_time) >= AMARILLO_SECONDS:
            estado_semaforo = "ROJO"
    else:
        if estado_semaforo == "ROJO":
            estado_semaforo = "AMARILLO"
            _last_transition_time = time.time()
        elif estado_semaforo == "AMARILLO" and (time.time() - _last_transition_time) >= AMARILLO_SECONDS:
            estado_semaforo = "VERDE"

# ---------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------
def main():
    print("Inicializando detección (solo video)...")
    worker = VideoWorker(VIDEO_PATH_VEHICULOS, name="Vehiculos")
    worker.start()

    try:
        while True:
            frame, dets = worker.get_latest()
            if frame is None:
                time.sleep(0.05)
                continue

            veh_count, ped_count, fauna_count = classify_counts(dets)
            decide_semaforo(veh_count, ped_count, fauna_count)

            sys.stdout.write(
                f"\rVehículos:{veh_count}  Peatones:{ped_count}  Fauna:{fauna_count}  --> Semáforo:{estado_semaforo}     "
            )
            sys.stdout.flush()

            fv = frame.copy()
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                cls = labels.get(d["class_id"], str(d["class_id"]))
                cv2.rectangle(fv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(fv, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(fv, f"Semaforo:{estado_semaforo}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Video Vehículos", fv)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nFinalizando...")
    finally:
        worker.stop()
        worker.join()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
