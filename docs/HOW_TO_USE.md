# How to Use the Plant Disease Detector Website

## 📱 Quick Start - Using the Website

### Step 1: Access the Website

Open your browser and go to:
```
https://MohammadKhaderObeidat.github.io/plant-disease-detector-v2/app.html
```

### Step 2: Upload an Image

You have **two options**:

#### Option A: Click to Upload
1. Click the **"Choose Image"** button
2. Select an image from your computer
3. The image will appear in the preview area

#### Option B: Drag and Drop
1. Take an image file from your computer
2. Drag it directly onto the upload area
3. Drop it in the designated zone
4. The image will appear in the preview area

### Step 3: Analyze the Image

1. After uploading, click the **"Analyze"** button
2. Wait 2-3 seconds for the AI to process
3. You'll see a loading spinner while it analyzes

### Step 4: View Results

The results will show:
- **Top 3 Predictions** - Most likely diseases
- **Confidence Scores** - How confident the AI is (0-100%)
- **Disease Information** - Description of the detected disease
- **Treatment Recommendations** - What to do about it

---

## 🖼️ Image Requirements

### Supported Formats
- ✅ PNG (.png)
- ✅ JPG / JPEG (.jpg, .jpeg)
- ✅ GIF (.gif)
- ✅ WebP (.webp)

### Image Size
- **Maximum**: 10 MB
- **Recommended**: 500 KB - 2 MB
- **Resolution**: 224 x 224 pixels (or larger, will be resized)

### Best Practices for Accurate Results

1. **Clear Focus** - Make sure the leaf is in focus
2. **Good Lighting** - Use natural daylight or good lighting
3. **Clean Background** - Solid background works best
4. **Close-up Shot** - Fill most of the frame with the leaf
5. **Healthy Leaf** - Include the entire leaf if possible
6. **No Obstructions** - Don't cover parts of the leaf

### Example Good Images
```
✅ Close-up of a single leaf
✅ Well-lit with natural sunlight
✅ Clear disease symptoms visible
✅ Leaf fills most of the frame
✅ Clean, simple background
```

### Example Poor Images
```
❌ Blurry or out of focus
❌ Too dark or shadowy
❌ Multiple leaves mixed together
❌ Leaf is too small in frame
❌ Cluttered background
```

---

## 🔍 Understanding the Results

### Confidence Score

The confidence score tells you how sure the AI is about its prediction:

- **90-100%** - Very confident, likely accurate
- **70-89%** - Confident, probably accurate
- **50-69%** - Moderate confidence, may need verification
- **Below 50%** - Low confidence, get a second opinion

### Disease Categories

The model can detect:

1. **Healthy** - No disease detected
2. **Rust** - Reddish-brown pustules on leaves
3. **Powdery Mildew** - White powder-like coating
4. **Leaf Spot** - Dark spots or lesions
5. **Blight** - Rapid leaf damage
6. **And 15+ other plant diseases**

---

## 💡 Tips for Best Results

### 1. Multiple Angles
If unsure about results, try uploading:
- Different angles of the same leaf
- Different leaves from the same plant
- The average prediction is more reliable

### 2. Lighting Matters
- Use **natural daylight** when possible
- Avoid harsh shadows
- Don't use flash photography
- Cloudy daylight works well

### 3. Disease Progression
- Early stage diseases may be harder to detect
- Advanced symptoms are easier to identify
- Upload at different stages for better understanding

### 4. Plant Type
The model works best with:
- Apple leaves
- Corn/Maize leaves
- Grape leaves
- Potato leaves
- Tomato leaves
- Pepper leaves
- Cherry leaves
- Blueberry leaves
- Peach leaves
- And more...

---

## 🛠️ Advanced Usage - Using the Model Locally

### Option 1: Python Script (Recommended)

If you have Python installed on your computer:

#### Step 1: Clone the Repository
```bash
git clone https://github.com/MohammadKhaderObeidat/plant-disease-detector-v2.git
cd plant-disease-detector-v2
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Download the Model
```bash
# Download EfficientNetB4_adaptive_model.h5 from:
# https://github.com/MohammadKhaderObeidat/plant-disease-detector-v2/releases
# Place it in the project root directory
```

#### Step 4: Run Inference
```python
import tensorflow as tf
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model('EfficientNetB4_adaptive_model.h5')

# Load and preprocess image
img = cv2.imread('path/to/leaf_image.jpg')
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Make prediction
predictions = model.predict(img)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]

print(f"Predicted Class: {class_idx}")
print(f"Confidence: {confidence:.2%}")
print(f"All Predictions: {predictions[0]}")
```

### Option 2: Flask Web Server (For Local Network)

#### Step 1: Install Flask
```bash
pip install flask flask-cors
```

#### Step 2: Run the Server
```bash
python server/plant_disease_api.py
```

#### Step 3: Access Locally
```
http://localhost:5000
```

#### Step 4: Upload Images
- Open the local web interface
- Upload images
- Get predictions

---

## 🐛 Troubleshooting

### Issue: "Image won't upload"

**Solution:**
- Check file size (must be under 10 MB)
- Try a different image format (PNG, JPG)
- Refresh the page and try again
- Clear browser cache

### Issue: "Analyze button is disabled"

**Solution:**
- Make sure you've uploaded an image first
- The image preview should be visible
- Try uploading again

### Issue: "Results don't look right"

**Solution:**
- Try uploading a clearer image
- Ensure good lighting
- Try multiple angles
- Check if the leaf type is supported

### Issue: "Website is slow"

**Solution:**
- Compress your image before uploading
- Use a smaller image file
- Try a different browser
- Check your internet connection

---

## 📊 Batch Processing (Multiple Images)

### Using Python Script

```python
import os
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

# Load model
model = tf.keras.models.load_model('EfficientNetB4_adaptive_model.h5')

# Process all images in a folder
image_folder = 'path/to/images'
results = []

for image_file in os.listdir(image_folder):
    if image_file.endswith(('.jpg', '.png', '.jpeg')):
        # Load image
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = model.predict(img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        results.append({
            'image': image_file,
            'class': class_idx,
            'confidence': confidence
        })
        
        print(f"{image_file}: Class {class_idx}, Confidence {confidence:.2%}")

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 🎯 Best Practices

### 1. Data Privacy
- Images are processed locally in your browser
- No images are sent to external servers (unless using API)
- Your data stays private

### 2. Model Accuracy
- The model is 97%+ accurate on training data
- Real-world accuracy may vary
- Always verify results with expert opinion

### 3. Regular Updates
- Check for model updates periodically
- New versions may have better accuracy
- Follow the GitHub repository for updates

### 4. Feedback
- If results seem wrong, try:
  - Different image angle
  - Better lighting
  - Clearer focus
  - Different leaf from same plant

---

## 📚 Additional Resources

### Documentation
- **README.md** - Project overview
- **METHODOLOGY.md** - Technical details
- **DEPLOYMENT.md** - Setup instructions

### GitHub Repository
```
https://github.com/MohammadKhaderObeidat/plant-disease-detector-v2
```

### Model Files
- **EfficientNetB4_adaptive_model.h5** - Trained model (71.56 MB)
- **class_indices.json** - Class mapping

---

## 🚀 Getting Started Right Now

1. **Open the app**: https://MohammadKhaderObeidat.github.io/plant-disease-detector-v2/app.html
2. **Take a photo** of a leaf with your phone or camera
3. **Upload the image** by clicking or dragging
4. **Click Analyze** and wait for results
5. **Read the recommendations** and take action

---

## ❓ FAQ

**Q: Can I use this on my phone?**
A: Yes! The website works on mobile browsers. Just open the link on your phone.

**Q: How accurate is the model?**
A: 97%+ accuracy on the training dataset. Real-world accuracy depends on image quality.

**Q: What if the result is wrong?**
A: Try uploading a clearer image or different angle. Always verify with an expert.

**Q: Can I use this for commercial purposes?**
A: Check the LICENSE file in the repository for usage terms.

**Q: How do I get the trained model?**
A: Download from the GitHub releases page.

**Q: Can I train my own model?**
A: Yes! Follow the DEPLOYMENT.md guide for training instructions.

---

**Last Updated**: May 2026  
**Version**: 2.0  
**Status**: Production Ready
