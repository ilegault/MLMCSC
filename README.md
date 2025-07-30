# MLMCSC - Machine Learning Microscope Control System

A comprehensive system for real-time microscope control and Charpy fracture surface analysis using machine learning.

## 🚀 Features

- **Live Microscope Control**: Real-time camera interface with calibration
- **YOLO Object Detection**: Automatic fracture surface detection
- **Shear Classification**: ML-based shear percentage estimation
- **Real-time Analysis**: Live processing and visualization
- **Modular Architecture**: Clean, extensible codebase
- **Research-Oriented**: Organized for experiments and development

## 📁 Project Structure

```
MLMCSC/
├── 📁 src/                    # Source code
│   ├── mlmcsc/               # Main package
│   │   ├── core/             # Core functionality
│   │   ├── detection/        # Object detection
│   │   ├── classification/   # Classification models
│   │   ├── camera/           # Camera interface
│   │   ├── preprocessing/    # Data preprocessing
│   │   ├── utils/            # Utilities
│   │   └── config/           # Configuration
│   └── apps/                 # Applications
│       ├── live_viewer/      # Live microscope viewer
│       ├── trainer/          # Model training
│       └── analyzer/         # Analysis tools
├── 📁 experiments/           # Research experiments
│   ├── detection/           # Detection experiments
│   ├── classification/      # Classification experiments
│   ├── notebooks/          # Jupyter notebooks
│   └── configs/            # Experiment configs
├── 📁 models/               # Trained models
│   ├── detection/          # YOLO models
│   ├── classification/     # ML models
│   └── archived/           # Old versions
├── 📁 data/                # Datasets
│   ├── raw/               # Original data
│   ├── processed/         # Processed data
│   ├── annotations/       # Annotations
│   └── samples/           # Sample images
├── 📁 tools/               # Utility scripts
├── 📁 tests/               # Test suite
├── 📁 docs/                # Documentation
└── 📁 results/             # Experiment results
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)
- Webcam or microscope camera

### Quick Install
```bash
# Clone the repository
git clone https://github.com/yourusername/MLMCSC.git
cd MLMCSC

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Development Install
```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## 🚀 Quick Start

### 1. Live Microscope Viewer
```bash
# Using main entry point
python main.py live

# Or directly
python src/apps/live_viewer/simple_live_model.py
```

### 2. Train Models
```bash
# Train detection model
python main.py train detection --data data/raw/charpy_dataset

# Train classification model
python main.py train classification --data data/processed/shiny_training_data
```

### 3. Analyze Data
```bash
# Analyze dataset
python main.py analyze dataset --path data/raw/charpy_dataset
```

## 📋 Configuration

Create a configuration file:
```bash
python main.py config --create-default
```

Edit `config/default.yaml`:
```yaml
camera:
  device_id: 1
  resolution: [1280, 720]
  target_fps: 30

model:
  yolo_model_path: "models/detection/best.pt"
  classification_model_path: "models/classification/charpy_shear_regressor.pkl"
  confidence_threshold: 0.3

display:
  window_scale: 0.8
  show_detections: true
  show_predictions: true
```

## 🔄 Migration from Old Structure

If you have an existing MLMCSC installation, use the migration guide:
```bash
python main.py migrate
```

Or import the compatibility layer:
```python
import compatibility  # Provides backward compatibility
```

## 📊 Usage Examples

### Live Viewer with Custom Models
```python
from apps.live_viewer import WorkingLiveMicroscopeViewer

viewer = WorkingLiveMicroscopeViewer(
    yolo_model_path="models/detection/my_model.pt",
    regression_model_path="models/classification/my_classifier.pkl"
)

viewer.load_calibration()
viewer.load_models()
viewer.connect_camera()
viewer.run()
```

### Classification Only
```python
from mlmcsc.classification import ShinyRegionBasedClassifier

classifier = ShinyRegionBasedClassifier()
classifier.load_model("models/classification/charpy_shear_regressor.pkl")

# Predict shear percentage
result = classifier.predict(image)
print(f"Shear percentage: {result['shear_percentage']:.1f}%")
```

### Detection Only
```python
from mlmcsc.detection import ObjectDetector

detector = ObjectDetector("models/detection/best.pt")
detections = detector.detect(image)

for detection in detections:
    print(f"Found {detection['class']} with confidence {detection['confidence']:.2f}")
```

## 🧪 Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Type Checking
```bash
mypy src/
```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Training Guide](docs/training.md)
- [Configuration Reference](docs/configuration.md)
- [Migration Guide](docs/migration.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLO team for object detection framework
- OpenCV community for computer vision tools
- scikit-learn for machine learning utilities

## 📞 Support

- Create an issue for bug reports
- Start a discussion for questions
- Check the documentation for guides

---

**MLMCSC v2.0** - Restructured for better organization and maintainability