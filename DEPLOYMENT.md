# Plant Disease Detector V2 - Deployment & Usage Guide

## Quick Start

### For macOS Users

#### Prerequisites
- Python 3.8 or higher
- Homebrew (optional, for package management)
- Git

#### Step 1: Clone the Repository

```bash
git clone https://github.com/MohammadKhaderObeidat/plant-disease-detector-v2.git
cd plant-disease-detector-v2
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in your terminal)
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# If you encounter issues with TensorFlow, try:
pip install tensorflow-macos tensorflow-metal  # For Apple Silicon Macs
# OR
pip install tensorflow  # For Intel Macs
```

#### Step 4: Download the Trained Model

The trained model file (`EfficientNetB4_adaptive_model.h5`) needs to be downloaded separately due to file size limitations. You can:

**Option A: Download from Release**
- Go to: https://github.com/MohammadKhaderObeidat/plant-disease-detector-v2/releases
- Download `EfficientNetB4_adaptive_model.h5`
- Place it in the project root directory

**Option B: Train Your Own Model**
```bash
python adaptive_training.py
```

#### Step 5: Run Inference

```python
import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model('EfficientNetB4_adaptive_model.h5')

# Load image
img = cv2.imread('path/to/leaf_image.jpg')
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]

print(f"Predicted Class Index: {class_idx}")
print(f"Confidence: {confidence:.2%}")
```

## Web Application Deployment

### Option 1: Local Web Server

```bash
# Install Flask (if not already installed)
pip install flask flask-cors

# Run the web server
python server/plant_disease_api.py

# Open browser to http://localhost:5000
```

### Option 2: Deploy to Heroku

```bash
# Install Heroku CLI
brew tap heroku/brew && brew install heroku

# Login to Heroku
heroku login

# Create Heroku app
heroku create plant-disease-detector-v2

# Deploy
git push heroku main

# Open app
heroku open
```

### Option 3: Deploy to AWS Lambda

1. Package the model and code
2. Create Lambda function
3. Set up API Gateway
4. Deploy

### Option 4: Deploy to Google Cloud Run

```bash
# Install Google Cloud CLI
brew install google-cloud-sdk

# Authenticate
gcloud auth login

# Deploy
gcloud run deploy plant-disease-detector-v2 \
  --source . \
  --platform managed \
  --region us-central1
```

## API Endpoints

### POST /predict

**Request:**
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "class": "Apple___healthy",
  "confidence": 0.9872,
  "predictions": {
    "Apple___healthy": 0.9872,
    "Apple___Black_rot": 0.0098,
    "Apple___Cedar_apple_rust": 0.0030
  }
}
```

### GET /health

**Response:**
```json
{
  "status": "healthy",
  "model": "EfficientNetB4",
  "accuracy": "97.2%"
}
```

## Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
pip install --upgrade tensorflow
```

### Issue: "CUDA not found" (on GPU machines)

**Solution:** This is normal on CPU-only machines. The model will run on CPU.

### Issue: "Model file not found"

**Solution:**
1. Ensure `EfficientNetB4_adaptive_model.h5` is in the project root
2. Or train the model: `python adaptive_training.py`

### Issue: "Out of memory" during training

**Solution:**
```python
# Reduce batch size in adaptive_training.py
BATCH_SIZE = 16  # Instead of 32
```

### Issue: "Permission denied" on macOS

**Solution:**
```bash
chmod +x adaptive_training.py
python3 adaptive_training.py
```

## Performance Optimization

### For Faster Inference

```python
# Use quantized model (if available)
model = tf.lite.Interpreter(model_path='model_quantized.tflite')
```

### For Batch Processing

```python
# Process multiple images at once
images = np.array([img1, img2, img3, ...])  # Shape: (batch_size, 224, 224, 3)
predictions = model.predict(images)
```

### For Real-time Video

```python
import cv2

cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = model.predict(img, verbose=0)
    class_idx = np.argmax(predictions[0])
    
    # Display result
    cv2.putText(frame, f"Class: {class_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Plant Disease Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Model Conversion

### Convert to TensorFlow Lite (for Mobile)

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('EfficientNetB4_adaptive_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Convert to ONNX (for Cross-Platform)

```bash
pip install tf2onnx
python -m tf2onnx.convert --saved-model EfficientNetB4_adaptive_model.h5 --output_file model.onnx
```

## Monitoring & Logging

### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("Model loaded successfully")
logger.debug(f"Input shape: {model.input_shape}")
```

### Performance Metrics

```python
import time

start_time = time.time()
predictions = model.predict(img)
inference_time = time.time() - start_time

print(f"Inference time: {inference_time*1000:.2f}ms")
```

## Security Considerations

1. **Input Validation**: Always validate image inputs
2. **Rate Limiting**: Implement rate limiting on API endpoints
3. **Authentication**: Add API key authentication for production
4. **HTTPS**: Use HTTPS for all API calls
5. **Model Protection**: Protect the trained model file

## Support & Resources

- **GitHub Issues**: https://github.com/MohammadKhaderObeidat/plant-disease-detector-v2/issues
- **Documentation**: See README.md and METHODOLOGY.md
- **Model Details**: See METHODOLOGY.md for architecture and training details

## License

MIT License - See LICENSE file for details

---

**Last Updated**: May 2026  
**Version**: 2.0  
**Status**: Production Ready
