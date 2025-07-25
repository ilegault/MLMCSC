# Object Detection Module

This module provides comprehensive specimen detection and tracking capabilities for microscope automation using YOLOv8.

## Features

### Core Detection Capabilities
- **Real-time detection** (>30 FPS on modern hardware)
- **Multi-specimen tracking** with unique IDs
- **Confidence thresholding** (default 0.8)
- **Non-maximum suppression** to eliminate duplicate detections
- **Motion detection** and stability analysis
- **Rotation detection** and correction
- **Auto-centering** calculations
- **Region of Interest (ROI)** extraction

### Advanced Features
- **Specimen stability tracking** - detects when specimens are stationary
- **Multi-object tracking** - maintains consistent IDs across frames
- **Rotation angle detection** - calculates specimen orientation
- **Center offset calculation** - for auto-centering functionality
- **Performance monitoring** - real-time FPS tracking

## Quick Start

### Basic Usage

```python
from src.models import SpecimenDetector
import cv2

# Initialize detector
detector = SpecimenDetector(
    confidence_threshold=0.8,
    device='auto'  # Uses GPU if available
)

# Load image
frame = cv2.imread('microscope_image.jpg')

# Detect specimens
results = detector.detect_specimen(frame)

# Process results
for result in results:
    print(f"Specimen {result.specimen_id}: "
          f"confidence={result.confidence:.2f}, "
          f"stable={result.is_stable}, "
          f"rotation={result.rotation_angle:.1f}°")
```

### Real-time Detection

```python
import cv2
from src.models import SpecimenDetector

detector = SpecimenDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect specimens
    results = detector.detect_specimen(frame)
    
    # Draw results
    for result in results:
        x, y, w, h = [int(v) for v in result.bbox]
        color = (0, 255, 0) if result.is_stable else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"ID:{result.specimen_id}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## API Reference

### SpecimenDetector Class

#### Constructor
```python
SpecimenDetector(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.8,
    nms_threshold: float = 0.4,
    device: str = 'auto',
    max_detections: int = 10
)
```

**Parameters:**
- `model_path`: Path to custom YOLO model (uses pre-trained if None)
- `confidence_threshold`: Minimum confidence for detections (0.0-1.0)
- `nms_threshold`: Non-maximum suppression threshold (0.0-1.0)
- `device`: Device to run inference on ('cpu', 'cuda', 'auto')
- `max_detections`: Maximum number of detections per frame

#### Key Methods

##### detect_specimen(frame)
Detect specimens in the given frame.

**Parameters:**
- `frame`: Input image as numpy array

**Returns:**
- `List[DetectionResult]`: List of detection results

##### track_specimen(frame)
Track specimens across frames (alias for detect_specimen).

##### is_specimen_stable(specimen_id)
Check if a specific specimen is stable.

**Parameters:**
- `specimen_id`: ID of the specimen to check

**Returns:**
- `bool`: True if specimen is stable

##### extract_roi(frame, detection_result, padding=20)
Extract region of interest around a detected specimen.

**Parameters:**
- `frame`: Input image
- `detection_result`: DetectionResult object
- `padding`: Additional padding around bounding box

**Returns:**
- `Optional[np.ndarray]`: Extracted ROI image

##### auto_center(detection_results)
Calculate centering adjustments for auto-centering.

**Parameters:**
- `detection_results`: List of detection results

**Returns:**
- `Optional[Tuple[float, float]]`: (dx, dy) adjustments needed

### DetectionResult Class

Data class containing detection information:

```python
@dataclass
class DetectionResult:
    specimen_id: int              # Unique specimen ID
    bbox: List[float]            # [x, y, width, height]
    confidence: float            # Detection confidence (0.0-1.0)
    is_stable: bool             # Whether specimen is stationary
    rotation_angle: float       # Rotation angle in degrees
    center_offset: List[float]  # [dx, dy] offset from frame center
    timestamp: float            # Detection timestamp
```

## Training Custom Models

### Data Preparation

1. **Organize your dataset:**
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. **Create annotations in YOLO format:**
```
# Each line: class_id x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.1 0.15
```

### Training Script

```python
from src.models import SpecimenTrainer

# Initialize trainer
trainer = SpecimenTrainer(
    dataset_path="data/specimen_dataset",
    output_dir="training_output"
)

# Prepare dataset (if you have raw images and labels)
config_path = trainer.prepare_dataset(
    source_images_dir="data/raw/images",
    source_labels_dir="data/raw/labels"
)

# Train model
results = trainer.train_model(
    config_path=config_path,
    model_size='n',  # nano for speed
    epochs=100,
    batch_size=16
)

print(f"Best model saved at: {results['best_model_path']}")
```

### Using Custom Models

```python
# Load custom trained model
detector = SpecimenDetector(
    model_path="path/to/your/best.pt",
    confidence_threshold=0.8
)
```

## Annotation Tools

### Simple Annotation Tool

A GUI tool for creating training annotations:

```python
from src.models import SimpleAnnotationTool

# Define your classes
class_names = ['specimen', 'cell', 'bacteria', 'particle', 'debris']

# Launch annotation tool
tool = SimpleAnnotationTool(class_names)
tool.run()
```

### Format Conversion

Convert between different annotation formats:

```python
from src.models import AnnotationConverter

converter = AnnotationConverter(class_names)

# Convert COCO to YOLO
converter.coco_to_yolo('annotations.json', 'yolo_labels/')

# Convert Pascal VOC to YOLO
converter.pascal_voc_to_yolo('voc_annotations/', 'yolo_labels/')
```

## Performance Optimization

### Hardware Requirements
- **Minimum**: CPU with 4+ cores, 8GB RAM
- **Recommended**: NVIDIA GPU with 4GB+ VRAM, 16GB RAM
- **Optimal**: RTX 3060 or better, 32GB RAM

### Speed vs Accuracy Trade-offs

| Model Size | Speed (FPS) | Accuracy | Use Case |
|------------|-------------|----------|----------|
| YOLOv8n    | 60-100      | Good     | Real-time applications |
| YOLOv8s    | 40-60       | Better   | Balanced performance |
| YOLOv8m    | 25-40       | High     | High accuracy needed |
| YOLOv8l    | 15-25       | Higher   | Research applications |
| YOLOv8x    | 10-15       | Highest  | Maximum accuracy |

### Optimization Tips

1. **Use appropriate model size** for your hardware
2. **Adjust confidence threshold** based on your needs
3. **Reduce input resolution** for faster inference
4. **Use GPU acceleration** when available
5. **Batch processing** for multiple images

## Troubleshooting

### Common Issues

**Low FPS Performance:**
- Check GPU availability: `torch.cuda.is_available()`
- Reduce input image size
- Use smaller model (YOLOv8n)
- Close other GPU-intensive applications

**Poor Detection Accuracy:**
- Lower confidence threshold
- Train custom model with your data
- Improve image quality (lighting, focus)
- Add data augmentation during training

**Memory Issues:**
- Reduce batch size during training
- Use smaller model
- Reduce input image resolution
- Close other memory-intensive applications

**Installation Issues:**
```bash
# Install required dependencies
pip install ultralytics torch torchvision opencv-python numpy scipy

# For GPU support (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Examples

See `examples/object_detection_example.py` for comprehensive usage examples including:
- Real-time detection demo
- Auto-capture functionality
- Performance benchmarking
- Sample image testing

## Contributing

When contributing to the object detection module:

1. Follow the existing code style
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update this README for new functionality
5. Test with various microscope images

## License

This module is part of the MLMCSC project. See the main project LICENSE file for details.