# License Plate Detection & Recognition System

A comprehensive end-to-end automatic license plate recognition (ALPR) system using YOLOv8 for detection and TrOCR for optical character recognition. This project achieves **98.5% mAP@0.5** and **74.1% mAP@0.5:0.95** on license plate detection.

## Overview

This project implements a complete pipeline for detecting vehicle license plates in images and extracting the alphanumeric characters using state-of-the-art deep learning models. The system is designed for real-time processing and can be integrated into parking management, toll collection, and traffic monitoring systems.

## Key Features

- **High-Performance Detection**: YOLOv8s-based license plate detection with 98.3% recall
- **Accurate OCR**: TrOCR (Transformer-based OCR) for robust text extraction
- **Complete Training Pipeline**: From XML to YOLO format conversion
- **Real-time Processing**: Optimized for fast inference with CUDA support
- **Automated Workflow**: Batch processing capabilities for multiple images
- **Payment Integration**: Example implementation for parking/toll systems

## Technical Architecture

### Detection Model
- **Model**: YOLOv8s (small variant)
- **Input Size**: 640x640
- **Training Epochs**: 50
- **Batch Size**: 16
- **Optimizer**: AdamW with automatic hyperparameter tuning

### OCR Model
- **Model**: Microsoft TrOCR (Base-Printed)
- **Processor**: ViT-based image encoder + GPT-2 text decoder
- **Beam Search**: 5 beams for improved accuracy
- **Post-processing**: Regex-based text cleaning

## Performance Metrics

| Metric | Value |
|--------|-------|
| Precision | 95.7% |
| Recall | 98.3% |
| mAP@0.5 | 98.5% |
| mAP@0.5:0.95 | 74.1% |
| Inference Speed | ~8ms per image (GPU) |

## Dataset

- **Source**: Kaggle - Car Plate Detection Dataset
- **Images**: 433 training/validation images
- **Format**: XML annotations converted to YOLO format
- **Classes**: 1 (license plate)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Dependencies

```bash
pip install ultralytics kagglehub opencv-python pillow torch torchvision transformers
```

## Usage

### 1. Data Preparation

```python
import kagglehub
from pathlib import Path
import shutil

# Download dataset
src_dir = Path(kagglehub.dataset_download("andrewmvd/car-plate-detection"))
work_dir = Path("/path/to/working/directory")
shutil.copytree(src_dir, work_dir)
```

### 2. Convert Annotations (XML to YOLO)

The notebook includes a complete conversion script that:
- Parses XML annotations
- Normalizes bounding box coordinates
- Generates YOLO-format `.txt` files

### 3. Train YOLOv8 Model

```bash
yolo detect train \
  data=car_plate.yaml \
  model=yolov8s.pt \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=plate_detector
```

### 4. Run Inference

```python
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2

# Load models
yolo = YOLO('path/to/best.pt')
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
trocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

# Detect and read plate
img = cv2.imread('test_image.jpg')
results = yolo(img)
# ... OCR processing (see notebook for complete code)
```

## Project Structure

```
Number-plate-detection/
├── README.md
├── plate_detection-2_github.ipynb    # Main notebook
└── car_plate_dataset/                # Dataset directory
    ├── images/
    ├── annotations/
    ├── labels/
    └── car_plate.yaml
```

## Applications

- **Parking Management**: Automated entry/exit logging
- **Toll Collection**: Contactless payment systems
- **Traffic Monitoring**: Speed enforcement and vehicle tracking
- **Security**: Access control for restricted areas
- **Fleet Management**: Vehicle identification and tracking

## Model Training Details

### Data Augmentation
- Mosaic augmentation
- Random scaling (0.5x)
- HSV color space adjustments
- Random horizontal flips
- Albumentations: Blur, MedianBlur, ToGray, CLAHE

### Training Configuration
- Learning rate: 0.002 (AdamW optimizer)
- Warmup epochs: 3
- Weight decay: 0.0005
- Mosaic: First 40 epochs
- Image preprocessing: Auto-augmentation

## Results Visualization

The training process shows consistent improvement across all metrics:
- Rapid convergence in first 15 epochs
- Stable performance after epoch 40
- Best model saved at highest mAP@0.5:0.95

## Future Enhancements

- [ ] Multi-country license plate support
- [ ] Real-time video stream processing
- [ ] Mobile deployment (ONNX/TFLite)
- [ ] Integration with vehicle databases
- [ ] Enhanced OCR for damaged/dirty plates
- [ ] Web API for inference

## Technologies Used

- **YOLOv8**: Ultralytics implementation
- **TrOCR**: Hugging Face Transformers
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **Kagglehub**: Dataset management
- **Google Colab**: Development environment

## License

This project is available for educational and research purposes.

## Acknowledgments

- Dataset: [Kaggle - Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- TrOCR: [Microsoft Research](https://huggingface.co/microsoft/trocr-base-printed)

## Contact

For questions or collaborations, please open an issue in this repository.

---

**Note**: This project was developed as part of a computer vision research initiative focusing on automated vehicle recognition systems.
