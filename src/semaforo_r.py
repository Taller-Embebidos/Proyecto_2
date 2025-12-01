import cv2
import numpy as np
import os
import time
import sys
import math

# ===========================
# CONFIGURACIÓN RASPBERRY PI 4
# ===========================
VIDEO_PATH = "video_test.mp4"  # Cambia el nombre si tu archivo es otro
WINDOW_NAME = "Semaforo Inteligente (RPi4)"

# Procesar 1 de cada N frames (para aligerar carga si hace falta)
PROCESS_EVERY_N_FRAMES = 2  # puedes poner 1 para procesar todos

# Número de hilos
cpu_cores = os.cpu_count() or 4
NUM_THREADS = min(4, cpu_cores)

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_THREADS)

try:
    cv2.setNumThreads(NUM_THREADS)
    cv2.setUseOptimized(True)
except Exception as e:
    print("Advertencia: no se pudo configurar hilos de OpenCV:", e)

print(f"CPU cores detectados: {cpu_cores}, usando hilos: {NUM_THREADS}")
print(f"Procesando 1 de cada {PROCESS_EVERY_N_FRAMES} frames")

# Tamaño de entrada del modelo
img_height, img_width = 640, 640

# ==========================================
#  CARGA DE MODELO TFLITE
# ==========================================
def load_tflite():
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter

TFLiteInterpreter = load_tflite()

try:
    interpreter = TFLiteInterpreter(
        model_path="yolo11n_float16.tflite",
        num_threads=NUM_THREADS
    )
except TypeError:
    # Por si la versión de tf.lite no acepta num_threads
    interpreter = TFLiteInterpreter(model_path="yolo11n_float16.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Thresholds
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
conf_threshold = CONF_THRESHOLD
nms_threshold = NMS_THRESHOLD

# ===========================
# Parámetros generales
# ===========================
CROSS_X1 = 0
CROSS_Y1 = 230
CROSS_X2 = 640
CROSS_Y2 = 360

ALLOWED_CLASSES = ["person", "car", "bus", "truck", "motorcycle", "bicycle", "cat", "dog"]
VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle", "bicycle"]
ANIMAL_CLASSES = ["cat", "dog"]

# ===========================
# Semáforos
# ===========================
# Vehicular inicia en VERDE, peatonal en ROJO
semaforo_vehicular = "GREEN"
semaforo_peatonal = "RED"
semaphore_state_since = time.time()

MIN_RED_TIME = 20.0
MIN_GREEN_TIME = 30.0

# tiempo acumulado de personas en cruce (para la regla de los 3s)
people_detected_since = None  # timestamp cuando empezamos a ver gente en el cruce

# ===========================
# FUNCIONES AUXILIARES
# ===========================
def is_in_crossing_zone(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (CROSS_X1 <= cx <= CROSS_X2) and (CROSS_Y1 <= cy <= CROSS_Y2)

def preprocess(frame):
    frame_resized = cv2.resize(frame, (640, 360))
    input_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    h, w = frame_resized.shape[:2]
    scale = min(img_height / h, img_width / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(frame_resized, (new_w, new_h))
    y_offset = (img_height - new_h) // 2
    x_offset = (img_width - new_w) // 2
    input_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_img, axis=0)

    return input_data, (x_offset, y_offset, scale, frame_resized.shape)

def get_color_by_class(class_name):
    if class_name == "person":
        return (0, 0, 255)
    elif class_name in VEHICLE_CLASSES:
        return (255, 0, 0)
    elif class_name in ANIMAL_CLASSES:
        return (0, 165, 255)
    else:
        return (0, 255, 0)

def overlap_correction(detections, labels):
    if len(detections) <= 1:
        return detections

    filtered = []
    for i, (bbox1, score1, class_id1) in enumerate(detections):
        if class_id1 >= len(labels):
            continue
        class_name1 = labels[class_id1]
        x1_1, y1_1, x2_1, y2_1 = bbox1
        keep = True

        for j, (bbox2, score2, class_id2) in enumerate(detections):
            if i == j or class_id2 >= len(labels):
                continue
            class_name2 = labels[class_id2]
            x1_2, y1_2, x2_2, y2_2 = bbox2

            overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
            overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
            overlap_area = overlap_x * overlap_y
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)

            if class_name1 == "motorcycle" and class_name2 == "person" and overlap_area > 0.5 * area1 and score2 > score1:
                keep = False
                break

        if keep:
            filtered.append((bbox1, score1, class_id1))
    return filtered

def postprocess(outputs, orig_dims, conf_threshold=0.3, nms_threshold=0.4):
    x_offset, y_offset, scale, (orig_h, orig_w, _) = orig_dims
    predictions = outputs[0][0]

    if predictions.shape[0] == 8400:
        predictions = predictions.T

    bbox_data = predictions[:4, :]
    scores = predictions[4:, :]

    class_ids = np.argmax(scores, axis=0)
    class_scores = np.max(scores, axis=0)

    valid_indices = class_scores > conf_threshold
    if not np.any(valid_indices):
        return []

    bboxes = bbox_data[:, valid_indices].T
    scores_valid = class_scores[valid_indices]
    class_ids_valid = class_ids[valid_indices]

    cx = bboxes[:, 0]
    cy = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]

    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height

    x1 = (x1 - x_offset) / scale
    y1 = (y1 - y_offset) / scale
    x2 = (x2 - x_offset) / scale
    y2 = (y2 - y_offset) / scale

    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    bboxes_orig = np.stack([x1, y1, x2, y2], axis=1)
    bboxes_xywh = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in bboxes_orig]

    indices = cv2.dnn.NMSBoxes(bboxes_xywh, scores_valid.tolist(), conf_threshold, nms_threshold)

    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            result.append((bboxes_orig[i], scores_valid[i], class_ids_valid[i]))
    return result

# ===========================
# LOOP PRINCIPAL (RPi4 + TIEMPO REAL)
# ===========================
def main():
    global semaforo_vehicular, semaforo_peatonal, semaphore_state_since, people_detected_since

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: No se puede abrir el video: {VIDEO_PATH}")
        return 1

    # FPS del video para sincronizar tiempo real
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    frame_period = 1.0 / fps

    print(f"FPS del video: {fps:.2f}  ->  periodo por frame: {frame_period:.4f} s")

    frame_count = 0
    processed_frames = 0
    start_time = time.time()

    video_start_time = time.time()
    frame_index = 0

    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin del video")
            break

        frame_count += 1
        frame_index += 1
        now = time.time()

        # Valores por defecto para info visual
        fps_inference = 0.0
        total_detections = 0
        crossing_people_count = 0
        people_detected_duration = 0.0

        # ¿Saltamos este frame?
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            frame_display = cv2.resize(frame, (640, 360))
            cv2.putText(frame_display, "[RPi4] Frame saltado", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame_display, f"Config: 1/{PROCESS_EVERY_N_FRAMES} frames", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        else:
            # Procesamiento completo
            frame_display = cv2.resize(frame, (640, 360))
            orig_h, orig_w = frame_display.shape[:2]

            input_data, transform_dims = preprocess(frame)
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            processed_frames += 1
            elapsed = time.time() - start_time
            fps_inference = processed_frames / elapsed if elapsed > 0 else 0.0

            outputs = [interpreter.get_tensor(output_details[i]["index"]) for i in range(len(output_details))]
            detections = overlap_correction(
                postprocess(outputs, transform_dims, conf_threshold, nms_threshold),
                labels
            )

            # Zona de cruce (línea más delgada)
            cv2.rectangle(frame_display, (CROSS_X1, CROSS_Y1), (CROSS_X2, CROSS_Y2), (0, 255, 255), 1)

            crossing_people_count = 0
            has_vehicle = False
            total_detections = 0

            # Dibujar detecciones
            for bbox, score, class_id in detections:
                if class_id >= len(labels):
                    continue

                class_name = labels[class_id]
                total_detections += 1

                if class_name not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                color = get_color_by_class(class_name)

                # Borde más delgado
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 1)

                label = f"{class_name}: {score:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame_display, (x1, y1 - label_size[1] - 4),
                              (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame_display, label, (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if class_name in VEHICLE_CLASSES:
                    has_vehicle = True

                if class_name == "person" and is_in_crossing_zone(bbox):
                    crossing_people_count += 1
                    # Borde de alerta un poco menos grueso
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_display, "EN CRUCE!", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # Tiempo continuo con personas en el cruce
            if crossing_people_count > 0:
                if people_detected_since is None:
                    people_detected_since = now
                people_detected_duration = now - people_detected_since
            else:
                people_detected_since = None
                people_detected_duration = 0.0

            # Lógica semáforo con regla de 3 segundos
            has_ped_3s = people_detected_duration >= 3.0
            time_in_state = now - semaphore_state_since

            if semaforo_vehicular == "GREEN":
                # Solo cambiamos a rojo si:
                # - hay personas en cruce por >= 3s
                # - y se cumplió MIN_GREEN_TIME o ya no hay vehículos
                if has_ped_3s and (time_in_state >= MIN_GREEN_TIME or not has_vehicle):
                    semaforo_vehicular = "RED"
                    semaphore_state_since = now
            elif semaforo_vehicular == "RED":
                if time_in_state >= MIN_RED_TIME:
                    semaforo_vehicular = "GREEN"
                    semaphore_state_since = now

            semaforo_peatonal = "RED" if semaforo_vehicular == "GREEN" else "GREEN"

            # Visualización semáforos (círculos llenos, sin cambio)
            color_v = (0, 255, 0) if semaforo_vehicular == "GREEN" else (0, 0, 255)
            cv2.circle(frame_display, (600, 60), 15, color_v, -1)
            cv2.putText(frame_display, f"Vehicular: {semaforo_vehicular}", (350, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_v, 1)

            color_p = (0, 255, 0) if semaforo_peatonal == "GREEN" else (0, 0, 255)
            cv2.circle(frame_display, (40, 60), 15, color_p, -1)
            cv2.putText(frame_display, f"Peatonal: {semaforo_peatonal}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_p, 1)

        # Info extra (simplificada)
        cv2.putText(frame_display, "Entorno: RPi4", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

        # Solo dejamos el tiempo de personas en cruce (como pediste)
        cv2.putText(frame_display, f"T. personas cruce: {people_detected_duration:4.1f}s", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Mostrar
        cv2.imshow(WINDOW_NAME, frame_display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Salida por teclado (q)")
            break

        # ==========================
        # CONTROL TIEMPO REAL + SKIP DINÁMICO
        # ==========================
        ideal_time = video_start_time + frame_index * frame_period
        now2 = time.time()
        delay = ideal_time - now2

        if delay > 0:
            # Vamos adelantados: esperamos
            time.sleep(delay)
        else:
            # Vamos atrasados: si el atraso es mayor que 1 frame, saltamos frames extra
            atraso = -delay
            if atraso > frame_period:
                skip_frames = min(3, int(atraso / frame_period))
                for _ in range(skip_frames):
                    ret_skip, frame_skip = cap.read()
                    if not ret_skip:
                        break
                    frame_count += 1
                    frame_index += 1
                # No procesamos esos frames, solo adelantamos el video

    total_real = time.time() - t0
    duracion_video = frame_index * frame_period

    print("========================================")
    print(f"Frames leídos: {frame_count}")
    print(f"Frames procesados: {processed_frames}")
    print(f"Duración ideal del video: {duracion_video:.3f} s")
    print(f"Duración real del procesamiento: {total_real:.3f} s")
    print(f"Diferencia: {total_real - duracion_video:.3f} s")
    print("========================================")

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())
