import cv2
import numpy as np
import os

# Configuración automática para PC vs RPi
is_raspberry_pi = os.path.exists('/proc/device-tree/model')

if is_raspberry_pi:
    # Solo optimización de threads para RPi4
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
    cv2.setNumThreads(2)

def load_tflite():
    if is_raspberry_pi:
        try:
            from tflite_runtime.interpreter import Interpreter
            return Interpreter
        except ImportError:
            import tensorflow as tf
            return tf.lite.Interpreter
    else:
        import tensorflow as tf
        return tf.lite.Interpreter

# Cargar el Interpreter apropiado
TFLiteInterpreter = load_tflite()

# Load TFLite model (mismo modelo para ambas plataformas)
interpreter = TFLiteInterpreter(model_path="yolo11n_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Load labels 
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

img_height, img_width = 640, 640
conf_threshold = 0.3
nms_threshold = 0.4

def preprocess(frame):
    # Reducir resolución primero para mantener proporción
    frame_resized = cv2.resize(frame, (640, 360))
    
    # Crear canvas cuadrado para el modelo
    input_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Calcular escala y padding para mantener relación de aspecto
    h, w = frame_resized.shape[:2]
    scale = min(img_height / h, img_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Redimensionar manteniendo relación de aspecto
    resized = cv2.resize(frame_resized, (new_w, new_h))
    
    # Centrar la imagen en el canvas
    y_offset = (img_height - new_h) // 2
    x_offset = (img_width - new_w) // 2
    input_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Preprocesamiento para el modelo
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.astype(np.float32) / 255.0  
    input_data = np.expand_dims(input_img, axis=0)
    
    return input_data, (x_offset, y_offset, scale, frame_resized.shape)

def get_color_by_class(class_name):
    """Asigna colores según la clase del objeto"""
    vehicle_classes = ["car", "bus", "truck", "motorcycle", "bicycle"]
    large_transport = ["train", "airplane", "boat"]
    
    if class_name == "person":
        return (0, 0, 255)  # ROJO
    elif class_name in vehicle_classes:
        return (255, 0, 0)  # AZUL
    elif class_name in large_transport:
        return (0, 255, 255)  # AMARILLO
    elif class_name in ["traffic light", "stop sign", "parking meter"]:
        return (255, 255, 0)  # CYAN
    else:
        return (0, 255, 0)  # VERDE (default)

def overlap_correction(detections, labels):
    """Si hay superposición entre moto y persona, mantener la de mayor score"""
    if len(detections) <= 1:
        return detections
    
    filtered = []
    
    for i, (bbox1, score1, class_id1) in enumerate(detections):
        if class_id1 >= len(labels):
            continue
            
        class_name1 = labels[class_id1]
        x1_1, y1_1, x2_1, y2_1 = bbox1
        
        # Calcular superposición con otras detecciones
        keep = True
        for j, (bbox2, score2, class_id2) in enumerate(detections):
            if i == j or class_id2 >= len(labels):
                continue
                
            class_name2 = labels[class_id2]
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calcular área de superposición
            overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
            overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
            overlap_area = overlap_x * overlap_y
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            
            # Si hay mucha superposición entre moto y persona
            if (class_name1 == "motorcycle" and class_name2 == "person" and 
                overlap_area > 0.5 * area1 and score2 > score1):
                keep = False
                break
                
        if keep:
            filtered.append((bbox1, score1, class_id1))
    
    return filtered

def postprocess(outputs, orig_dims, conf_threshold=0.3, nms_threshold=0.4):
    """
    Procesa la salida del modelo YOLO TFLite
    """
    x_offset, y_offset, scale, (orig_h, orig_w, _) = orig_dims
    
    
    if len(outputs) == 1:
        predictions = outputs[0][0]  # Shape: (84, 8400) o (8400, 84)
        if predictions.shape[0] == 8400:  # Si es (8400, 84), transponer
            predictions = predictions.T
        
        # Extraer componentes
        bbox_data = predictions[:4, :]  # cx, cy, w, h
        scores = predictions[4:, :]     # scores de clases
        
        # Obtener confidence máxima y class_id para cada detección
        class_ids = np.argmax(scores, axis=0)
        class_scores = np.max(scores, axis=0)
        objectness = np.ones_like(class_scores)  
        
        # Combinar scores
        confidence_scores = class_scores * objectness
        
        # Filtrar por confidence
        valid_indices = confidence_scores > conf_threshold
        
        if not np.any(valid_indices):
            return []
        
        # Extraer bounding boxes válidas
        bboxes = bbox_data[:, valid_indices].T
        scores_valid = confidence_scores[valid_indices]
        class_ids_valid = class_ids[valid_indices]
        
        # Convertir de formato YOLO (cx, cy, w, h) a (x1, y1, x2, y2)
        cx = bboxes[:, 0]
        cy = bboxes[:, 1]
        w = bboxes[:, 2]
        h = bboxes[:, 3]
        
        x1 = (cx - w/2)
        y1 = (cy - h/2)
        x2 = (cx + w/2)
        y2 = (cy + h/2)
        
        # Escalar a dimensiones del canvas de entrada (640x640)
        x1 = x1 * img_width
        y1 = y1 * img_height
        x2 = x2 * img_width
        y2 = y2 * img_height
        
        # Ajustar coordenadas al frame original
        x1 = (x1 - x_offset) / scale
        y1 = (y1 - y_offset) / scale
        x2 = (x2 - x_offset) / scale
        y2 = (y2 - y_offset) / scale
        
        # Limitar coordenadas al frame
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        bboxes_orig = np.stack([x1, y1, x2, y2], axis=1)
        
        # Aplicar NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes_orig.tolist(), 
            scores_valid.tolist(), 
            conf_threshold, 
            nms_threshold
        )
        
        result = []
        if len(indices) > 0:
            for i in indices.flatten():
                result.append((
                    bboxes_orig[i], 
                    scores_valid[i], 
                    class_ids_valid[i]
                ))
        
        return result
    
    else:
        print(f"Número inesperado de salidas: {len(outputs)}")
        return []

cap = cv2.VideoCapture("video_test.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Reducir resolución como en el código original
    frame_display = cv2.resize(frame, (640, 360))
    orig_h, orig_w = frame_display.shape[:2]
    
    # Preprocesar
    input_data, transform_dims = preprocess(frame)
    
    # Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Obtener todas las salidas
    outputs = []
    for i in range(len(output_details)):
        output = interpreter.get_tensor(output_details[i]['index'])
        outputs.append(output)
    
    # Postprocesar
    detections = postprocess(outputs, transform_dims, conf_threshold, nms_threshold)
    detections = overlap_correction(detections, labels)
    allowed_classes = ["person", "car", "bus", "truck", "motorcycle", "bicycle", "cat", "dog"]
    # Dibujar detecciones
    for bbox, score, class_id in detections:
        if class_id < len(labels) and labels[class_id] in allowed_classes:
            x1, y1, x2, y2 = bbox
            class_name = labels[class_id]
            label = f"{class_name}:{score:.2f}"
            color = get_color_by_class(class_name)
            
            cv2.rectangle(frame_display, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        color, 1)
            
            cv2.putText(frame_display, label, 
                    (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.imshow("YOLO TFLite", frame_display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()