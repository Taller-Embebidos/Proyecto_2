# AI Coding Agent Instructions - Sistema Cruce Inteligente

## Project Overview

**Purpose:** Intelligent pedestrian crossing system using edge AI on Raspberry Pi 4 with YOLO-based object detection (embedded systems university project).

**Key Architecture:** 
- Edge computing (Raspberry Pi 4 + Yocto Linux)
- Real-time video processing (USB cameras, 640x360)
- TensorFlow Lite + OpenCV for inference
- Adaptive traffic light control based on pedestrian/vehicle detection

---

## Critical Code Patterns & Conventions

### 1. Platform Detection & Optimization
**Pattern:** Detect Raspberry Pi vs. PC and optimize thread count accordingly.

**Location:** `src/semaforo.py` (lines 1-18)

```python
is_raspberry_pi = os.path.exists('/proc/device-tree/model')
if is_raspberry_pi:
    # RPi4: Set aggressive thread limits
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
    cv2.setNumThreads(2)
```

**When adding new code:** Always check `is_raspberry_pi` before resource allocation.

### 2. TensorFlow Lite Interpreter Selection
**Pattern:** Use `tflite_runtime` on RPi (lightweight), fallback to `tensorflow` for dev.

```python
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
```

**Convention:** Load with `model_path="yolo11n_float16.tflite"` (float16 preferred for RPi).

### 3. Image Preprocessing: Aspect Ratio Preservation
**Critical Logic:** Maintain aspect ratio by padding to 640x640 (YOLO input), track offset for coordinate remapping.

```python
def preprocess(frame):
    # 1. Reduce resolution first (640×360)
    # 2. Create 640×640 canvas
    # 3. Calculate scale factor
    # 4. Center image with padding
    # 5. Return (input_data, transform_dims) with metadata
```

**Why:** YOLO needs square input; coordinate post-processing requires inverse transform metadata.

### 4. Coordinate System Transformation
**Critical:** Detections in model output (640×640) must be reverse-transformed to original frame dimensions.

```python
# postprocess() tracks: x_offset, y_offset, scale from preprocess()
# Then: x_orig = (x_model - offset) / scale
# Finally: clip to frame bounds
```

**Pattern:** Always preserve transform metadata (x_offset, y_offset, scale, original_shape).

### 5. Model Output Handling
**Pattern:** Output format varies; handle both (84, 8400) and (8400, 84) shapes.

```python
predictions = outputs[0][0]
if predictions.shape[0] == 8400:  # Transpose if needed
    predictions = predictions.T

# Extract: bbox (4 values) + class_scores (80 values)
bbox_data = predictions[:4, :]
scores = predictions[4:, :]
```

**Convention:** YOLO11n outputs 8400 anchor points × 84 dims (4 bbox + 80 classes).

### 6. NMS and Overlap Correction
**Pattern:** Apply OpenCV NMS, then custom overlap correction for specific scenarios (motorcycle overlapping with person).

```python
indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_threshold, nms_threshold)
detections = overlap_correction(detections, labels)
```

**Domain Logic:** If motorcycle confidence > person confidence AND overlap >50%, keep motorcycle only.

### 7. Class Filtering & Color Coding
**Pattern:** Only display detected classes in `allowed_classes` list; assign colors by class category.

```python
allowed_classes = ["person", "car", "bus", "truck", "motorcycle", "bicycle", "cat", "dog"]

def get_color_by_class(class_name):
    vehicle_classes = ["car", "bus", "truck", ...]
    # Person: RED (0,0,255), Vehicles: BLUE (255,0,0), etc.
```

**Convention:** Use BGR color format (OpenCV standard, not RGB).

---

## Development Workflow

### Testing & Running Code

**Development (PC/Linux):**
```bash
# Ensure video file exists in src/
# Run detection with minimal overhead
python src/test.py          # Lightweight YOLO11n
python src/semaforo.py      # Full TFLite pipeline
```

**Target Platform (RPi):**
- Build Yocto image per `docs/Install_guide.md`
- Deploy via `bitbake core-image-minimal`
- Transfer Python scripts & model files to `/root/` on RPi

**Critical Distinction:** `test.py` uses PyTorch (fast iteration); `semaforo.py` uses TFLite (production RPi).

### Model Files
- **yolo11n.pt** (~10MB): PyTorch format for `test.py` (development)
- **yolo11n_float16.tflite** (~6MB): Optimized for RPi inference speed
- **yolo11n_float32.tflite**: Full precision (slower on RPi, use only if accuracy critical)
- **yolo11n_saved_model/**: TensorFlow SavedModel format (reference)

**Convention:** Always prefer float16 on RPi (2x faster, minimal accuracy loss).

### Input Resolution Strategy
- **Capture:** USB cameras → OpenCV reads at native resolution
- **Processing:** Downscale to 640×360 (preserves 16:9 aspect, reduces compute)
- **Model:** Pad to 640×640 square for YOLO inference
- **Display:** Show at 640×360 in OpenCV window

**Why:** Full HD processing (1920×1080) would exceed RPi4 5W power budget.

---

## Integration Points & Dependencies

### Hardware Dependencies
- **USB Cameras:** Two USB 2.0 webcams for stereo coverage
- **Video Input:** Uses OpenCV's `cv2.VideoCapture("video_test.mp4")` locally or `/dev/video0` on RPi
- **Output:** GPIO control of traffic light (future module)

### Software Stack
- **Python 3.9.16** (via Yocto meta-python)
- **OpenCV 4.7.0** (via meta-oe layer)
- **TensorFlow Lite 2.13.0** (custom recipe or tflite_runtime package)
- **NumPy** (for numerical operations)

### Critical External Files
- `labels.txt`: 80 COCO class names (order must match YOLO model output)
- Model `.tflite` files: Quantized models optimized for ARM inference

---

## Performance & Optimization Constraints

### Real-Time Requirements
- **Target:** ≤500ms inference + preprocessing per frame (per design doc RF001)
- **Frame rate:** 15 FPS (67ms per frame) achievable on RPi4
- **Bottleneck:** TFLite inference (~300-400ms on RPi4 for full frame)

### Memory Constraints
- **RPi4 RAM:** 4GB (shared with OS)
- **Model loaded:** ~15MB in memory
- **Optimization:** Pre-allocate tensors, avoid frame copies in loops

### Power Budget
- **Design target:** ≤15W per RPi (RNF003 from design doc)
- **Implication:** Avoid parallel processing; sequential pipeline only

### Optimization Checklist for New Features
- [ ] Profile with `time.time()` around inference calls
- [ ] Use cv2.dnn.NMSBoxes (optimized) over custom NMS
- [ ] Avoid unnecessary color space conversions (do once in preprocess)
- [ ] Pre-allocate numpy arrays instead of creating in loops

---

## Known Issues & Edge Cases

### Resolution Mismatches
**Issue:** Frame dimensions don't match expected 640×360 after downscaling.
**Solution:** Ensure preprocess() always produces 640×640 canvas regardless of input size.

### Coordinate Remapping Errors
**Issue:** Detections appear in wrong location on display.
**Solution:** Verify offset calculation: `(x_model - x_offset) / scale` with boundary clipping.

### Overlapping Motorcycle/Person
**Issue:** Both detected with high overlap; should prioritize one.
**Solution:** `overlap_correction()` already handles this; check label matching in `labels.txt`.

### Missing tflite_runtime on RPi
**Issue:** RPi falls back to full TensorFlow (slow, high memory).
**Solution:** Pre-install `tflite-runtime` in Yocto image via custom recipe.

---

## Project Structure & Key Files

```
.
├── src/
│   ├── semaforo.py           # Production TFLite pipeline (RPi target)
│   ├── test.py               # Development PyTorch pipeline (PC target)
│   ├── labels.txt            # 80 COCO class names
│   └── yolo11n*.tflite       # TFLite models (float16 preferred)
├── docs/
│   ├── Propuesta de diseño.md # Full system architecture & requirements
│   └── Install_guide.md       # Yocto build procedure
└── README.md                  # Quick start (needs update)
```

---

## When Modifying Core Logic

### Before changing preprocess()
- Verify output shape is always (1, 640, 640, 3) as float32
- Test coordinate remapping with known detection at image corners

### Before changing postprocess()
- Ensure all outputs format variations handled (transposed shapes)
- Run NMS before custom overlap_correction()

### Before optimizing inference
- Profile which stage is slowest (preprocess, invoke, postprocess)
- Never sacrifice accuracy for speed without testing on RPi

### Before adding new classes
- Update `labels.txt` if model trained on custom dataset
- Update `allowed_classes` filter list if new detection categories needed
- Verify new colors don't conflict in `get_color_by_class()`

---

## Yocto & Build System Context

This project requires a **Yocto-built Linux image** (not standard Raspberry Pi OS).

- **Build host:** Fedora 40+ or RHEL 10 (see `Install_guide.md` for Toolbx/Podman setup)
- **Target machine:** `MACHINE ??= "raspberrypi4"` in local.conf
- **Build command:** `bitbake core-image-minimal` (~2-4 hours)
- **Layers:** meta-raspberrypi, meta-python, meta-oe required

**For code changes:** Modify Python scripts directly on deployed RPi (no rebuild needed unless dependencies change).

---

## Questions to Ask Before Starting Work

1. **Target Platform:** Running on RPi4 or PC for testing?
   - RPi → Use float16 model, watch thread counts
   - PC → Use float32, parallel processing OK

2. **Model Changes:** Using YOLO11n or different architecture?
   - Check output tensor shapes and adjust postprocess()
   - Verify labels.txt matches new model

3. **New Detections:** Adding classes beyond COCO-80?
   - Retrain model and regenerate labels.txt
   - Update allowed_classes filtering

4. **Performance Issue:** Is bottleneck preprocess, inference, or postprocess?
   - Profile each stage separately before optimizing
