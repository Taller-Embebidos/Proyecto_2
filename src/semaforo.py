import cv2
import numpy as np
import os
import time
import sys

# ===========================
# Configuración automática para PC vs RPi vs QEMU - CORREGIDA
# ===========================
def detect_environment():
    """Detecta automáticamente el entorno de ejecución - VERSIÓN CORREGIDA"""
    
    # PRIMERO: Usar la variable del wrapper (más confiable)
    wrapper_env = os.environ.get('SEMAFORO_ENVIRONMENT', '').lower()
    if wrapper_env in ['qemu', 'rpi4']:
        print(f"✓ Usando entorno del wrapper: {wrapper_env}")
        return wrapper_env
    
    # SEGUNDO: Detección automática de respaldo
    machine = os.uname().machine
    print(f"Info máquina: {machine}")
    
    # Métodos para detectar QEMU
    qemu_indicators = [
        "/etc/qemu-banner",
        "/proc/device-tree/model"
    ]
    
    is_qemu = False
    for indicator in qemu_indicators:
        if os.path.exists(indicator):
            try:
                with open(indicator, 'r') as f:
                    content = f.read()
                    if 'QEMU' in content.upper():
                        is_qemu = True
                        print(f"✓ QEMU detectado por: {indicator}")
                        break
            except:
                continue
    
    # También verificar en uname
    if not is_qemu and 'qemu' in machine.lower():
        is_qemu = True
        print("✓ QEMU detectado por uname")
    
    if is_qemu:
        return "qemu"
    elif machine.startswith(("arm", "aarch")):
        return "rpi4"
    else:
        return "pc"

# Detectar entorno
environment = detect_environment()
print(f"Entorno final: {environment}")

# ===========================
# CONFIGURACIÓN POR ENTORNO - AJUSTADA PARA QEMU
# ===========================
if environment == "rpi4":
    # Optimización para RPi4
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
    cv2.setNumThreads(2)
    PROCESS_EVERY_N_FRAMES = 2
    WINDOW_NAME = "Semaforo Inteligente (RPi4)"
    CONF_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.4
    
elif environment == "qemu":
    # QEMU es EXTREMADAMENTE lento - ajustes MUY agresivos
    print(" QEMU detectado - aplicando OPTIMIZACIONES EXTREMAS para emulación")
    PROCESS_EVERY_N_FRAMES = 10  # Procesar solo 1 de cada 10 frames
    WINDOW_NAME = "Semaforo Inteligente (QEMU - MUY LENTO)"
    CONF_THRESHOLD = 0.7  # Umbral MUY alto para MUY pocas detecciones
    NMS_THRESHOLD = 0.6   # Menos NMS para más velocidad
    
    # Optimizaciones extremas
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1" 
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    cv2.setNumThreads(0)  # Deshabilitar threads de OpenCV
    
    # Reducir resolución de procesamiento
    global img_height, img_width
    img_height, img_width = 320, 320  # Mitad de resolución para YOLO
    
else:  # PC
    PROCESS_EVERY_N_FRAMES = 1
    WINDOW_NAME = "Semaforo Inteligente"
    CONF_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.4

print(f"Configuración: Procesando 1 de cada {PROCESS_EVERY_N_FRAMES} frames")

# ==========================================
#  EL RESTO DE TU CÓDIGO ORIGINAL (CON AJUSTES)
# ==========================================
def load_tflite():
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter

TFLiteInterpreter = load_tflite()
interpreter = TFLiteInterpreter(model_path="yolo11n_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

img_height, img_width = 640, 640
# Usar los thresholds según el entorno
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
semaforo_vehicular = "RED"
semaforo_peatonal = "GREEN"
semaphore_state_since = time.time()

MIN_RED_TIME = 20.0
MIN_GREEN_TIME = 30.0

# ===========================
# TUS FUNCIONES ORIGINALES (preprocess, is_in_crossing_zone, etc.)
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
# Loop principal
# ===========================
def main():
    global semaforo_vehicular, semaforo_peatonal, semaphore_state_since
    
    cap = cv2.VideoCapture("video_test.mp4")
    
    if not cap.isOpened():
        print("ERROR: No se puede abrir el video")
        return 1

    frame_count = 0
    processed_frames = 0
    start_time = time.time()
    
    print(f"Iniciando procesamiento en modo: {environment}")
    print(f"Procesando 1 de cada {PROCESS_EVERY_N_FRAMES} frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin del video")
            break

        frame_count += 1
        now = time.time()

        # Saltar frames según el entorno
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            frame_display = cv2.resize(frame, (640, 360))
            cv2.putText(frame_display, f"[{environment.upper()}] Frame saltado", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame_display, f"Config: 1/{PROCESS_EVERY_N_FRAMES} frames", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            try:
                cv2.imshow(WINDOW_NAME, frame_display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except Exception as e:
                print(f"Error mostrando ventana: {e}")
            continue

        # Procesamiento completo del frame
        frame_display = cv2.resize(frame, (640, 360))
        orig_h, orig_w = frame_display.shape[:2]

        input_data, transform_dims = preprocess(frame)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        processed_frames += 1
        elapsed = time.time() - start_time
        fps_inference = processed_frames / elapsed if elapsed > 0 else 0.0

        outputs = [interpreter.get_tensor(output_details[i]["index"]) for i in range(len(output_details))]
        detections = overlap_correction(postprocess(outputs, transform_dims, conf_threshold, nms_threshold), labels)

        # Dibujar zona de cruce
        cv2.rectangle(frame_display, (CROSS_X1, CROSS_Y1), (CROSS_X2, CROSS_Y2), (0, 255, 255), 2)

        crossing_people_count = 0
        has_vehicle = False
        total_detections = 0

        # Dibujar detecciones YOLO
        for bbox, score, class_id in detections:
            if class_id >= len(labels):
                continue
                
            class_name = labels[class_id]
            total_detections += 1
            
            if class_name not in ALLOWED_CLASSES:
                continue

            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            color = get_color_by_class(class_name)
            
            # Dibujar bounding box
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta con clase y confianza
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_display, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame_display, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Lógica para semáforos
            if class_name in VEHICLE_CLASSES:
                has_vehicle = True

            if class_name == "person" and is_in_crossing_zone(bbox):
                crossing_people_count += 1
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame_display, "EN CRUCE!", (x1, y1 - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Lógica del semáforo
        has_ped_or_animal = crossing_people_count > 0
        time_in_state = now - semaphore_state_since

        if semaforo_vehicular == "GREEN":
            if has_ped_or_animal and (time_in_state >= MIN_GREEN_TIME or not has_vehicle):
                semaforo_vehicular = "RED"
                semaphore_state_since = now
        elif semaforo_vehicular == "RED":
            if time_in_state >= MIN_RED_TIME:
                semaforo_vehicular = "GREEN"
                semaphore_state_since = now

        semaforo_peatonal = "RED" if semaforo_vehicular == "GREEN" else "GREEN"

        # Visualización
        color_v = (0, 255, 0) if semaforo_vehicular == "GREEN" else (0, 0, 255)
        cv2.circle(frame_display, (600, 60), 15, color_v, -1)
        cv2.putText(frame_display, f"Vehicular: {semaforo_vehicular}", (350, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_v, 2)

        color_p = (0, 255, 0) if semaforo_peatonal == "GREEN" else (0, 0, 255)
        cv2.circle(frame_display, (40, 60), 15, color_p, -1)
        cv2.putText(frame_display, f"Peatonal: {semaforo_peatonal}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_p, 2)

        # Información adicional
        cv2.putText(frame_display, f"Entorno: {environment}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_display, f"Detecciones: {total_detections}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_display, f"Personas en cruce: {crossing_people_count}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_display, f"FPS inferencia: {fps_inference:.1f}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostrar ventana
        try:
            cv2.imshow(WINDOW_NAME, frame_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            print(f"Error mostrando ventana: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Procesamiento completado. Frames: {frame_count}, Procesados: {processed_frames}")
    return 0

if __name__ == "__main__":
    sys.exit(main())