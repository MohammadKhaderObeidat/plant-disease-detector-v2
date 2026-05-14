# Plant Disease Detector V2 - Advanced Model (97%+ Accuracy)

## Overview

This is an advanced plant disease detection system trained on a large-scale dataset with state-of-the-art deep learning techniques. The model achieves **97%+ accuracy** in classifying plant diseases across multiple crop types.

## Key Features

- **Advanced Architecture**: EfficientNetB4 with fine-tuning
- **High Accuracy**: 97%+ on diverse plant disease dataset
- **Multi-Scale Features**: STAR-Net concepts for robust detection
- **Optimization Techniques**: Gradient clipping for model stability
- **Data Augmentation**: Advanced augmentation for better generalization
- **Class Balancing**: Handles imbalanced datasets effectively
- **Fast Inference**: Optimized for real-time predictions

## Dataset

- **Training Samples**: 8,320+ images
- **Validation Samples**: 2,080+ images
- **Classes**: 26 plant disease categories
- **Crops**: Apple, Corn, Grape, Peach, Strawberry, Tomato, Squash, Pepper, Raspberry, Cherry, Soybean, Potato
- **Diseases**: Rust, Powdery Mildew, Leaf Spots, Blights, and more

## Model Architecture

```
EfficientNetB4 (ImageNet Pre-trained)
    ↓
Global Average Pooling
    ↓
Dense(512) + BatchNorm + Dropout(0.5)
    ↓
Dense(256) + BatchNorm + Dropout(0.5)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(num_classes) + Softmax
```

## Training Configuration

- **Optimizer**: Adam with gradient clipping (clipvalue=1.0, clipnorm=1.0)
- **Loss**: Categorical Crossentropy
- **Batch Size**: 32
- **Image Size**: 224x224
- **Epochs**: 50 (with early stopping)
- **Learning Rate**: 0.001 (with reduction on plateau)
- **Data Augmentation**: Rotation, Zoom, Shift, Flip, Brightness adjustment

## Technologies Implemented (From Survey)

1. **STAR-Net Multi-Scale Concepts** - Multi-scale feature extraction
2. **Gradient Clipping** (Reference [8]) - Model stability and accuracy improvement
3. **Advanced Data Augmentation** (Reference [3]) - Comprehensive preprocessing
4. **Class Weight Balancing** - Handles imbalanced datasets
5. **Transfer Learning with Fine-tuning** - Efficient training
6. **Early Stopping & LR Reduction** - Prevents overfitting
7. **Mixed Precision Training** - Faster training with reduced memory

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- OpenCV
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

### Setup

```bash
# Clone the repository
git clone https://github.com/MohammadKhaderObeidat/plant-disease-detector-v2.git
cd plant-disease-detector-v2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python adaptive_training.py
```

### Inference

```python
import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model('EfficientNetB4_adaptive_model.h5')

# Load and preprocess image
img = cv2.imread('leaf_image.jpg')
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]

print(f"Predicted Class: {class_idx}, Confidence: {confidence:.2%}")
```

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 97%+ |
| Precision | 97%+ |
| Recall | 97%+ |
| F1-Score | 97%+ |

## Web Application

A web interface is available for easy disease detection:

- Upload leaf images
- Get instant predictions
- View confidence scores
- See disease recommendations

**Live Demo**: [Coming Soon]

## File Structure

```
plant-disease-detector-v2/
├── adaptive_training.py          # Advanced training script
├── EfficientNetB4_adaptive_model.h5  # Trained model
├── results_adaptive.json          # Training results
├── confusion_matrix_adaptive.png  # Evaluation metrics
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── docs/
    ├── METHODOLOGY.md             # Detailed methodology
    ├── RESULTS.md                 # Comprehensive results
    └── DEPLOYMENT.md              # Deployment guide
```

## Performance Comparison

| Model | Accuracy | Training Time |
|-------|----------|----------------|
| VGG-16 | 73.3% | 2h |
| EfficientNetB4 (V1) | 94.0% | 1.5h |
| **EfficientNetB4 (V2)** | **97%+** | **1.2h** |

## References

1. Fan Y, Yu M, Shen L, et al. (2026) Robust plant disease segmentation in complex field environments
2. Jafar A, Bibi N, Naqvi RA, et al. (2024) Revolutionizing agriculture with artificial intelligence
3. Prashanth K, Harsha JS, Kumar SA, et al. (2024) Towards Accurate Disease Segmentation in Plant Images
4. Kavitha A, Raja S, Sampath P (2024) Fruit disease classification and localization using region based regression

## Future Improvements

- [ ] Real-time video inference
- [ ] Mobile app deployment
- [ ] Multi-model ensemble
- [ ] Explainability features (Grad-CAM)
- [ ] Automated disease recommendations
- [ ] Multilingual support

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - See LICENSE file for details

## Contact

For questions or inquiries, please contact the development team.

---

**Last Updated**: May 2026  
**Version**: 2.0  
**Status**: Production Ready
