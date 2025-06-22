from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from models.discriminator import Discriminator
import io
import os
import numpy as np
import cv2
import base64

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model with proper architecture
model = Discriminator()
model_path = 'models/saved_models_fixed/discriminator.pth'

if os.path.exists(model_path):
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Load with strict=False to handle minor mismatches
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully with strict=False")
else:
    print(f"Error: Model not found at {model_path}")
    model = None

# Enhanced image preprocessing
def preprocess_image(img):
    # Convert to OpenCV format
    img_cv = np.array(img)
    img_cv = img_cv[:, :, ::-1].copy()  # Convert RGB to BGR
    
    # Detect faces using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        # If no face detected, use center crop
        height, width = img_cv.shape[:2]
        min_dim = min(height, width)
        startx = width//2 - min_dim//2
        starty = height//2 - min_dim//2
        face_img = img_cv[starty:starty+min_dim, startx:startx+min_dim]
        face_detected = False
    else:
        # Use largest face found
        (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
        # Expand face area by 20%
        expand = 0.2
        x = max(0, int(x - w * expand))
        y = max(0, int(y - h * expand))
        w = min(img_cv.shape[1] - x, int(w * (1 + 2*expand)))
        h = min(img_cv.shape[0] - y, int(h * (1 + 2*expand)))
        face_img = img_cv[y:y+h, x:x+w]
        face_detected = True
    
    # Convert back to PIL
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_img)
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transform(face_pil), face_detected

THRESHOLD = 0.5  # Higher threshold for better security

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    # Handle canvas image data
    if 'canvas_data' in request.form:
        try:
            # Extract base64 image data
            image_data = request.form['canvas_data'].split(',')[1]
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
    # Handle file upload
    elif 'image' in request.files:
        file = request.files['image']
        img_bytes = file.read()
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except:
            return jsonify({'error': 'Invalid image format'}), 400
    else:
        return jsonify({'error': 'No image provided'}), 400
    
    # Preprocess image with face detection
    try:
        img_t, face_detected = preprocess_image(img)
        img_t = img_t.unsqueeze(0).to(device)
    except Exception as e:
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 400
    
    with torch.no_grad():
        output = model(img_t).item()
    
    # Confidence score (sigmoid output)
    confidence = min(100, max(0, output * 100))
    result = "Verified Access" if output > THRESHOLD else "Not Verified"
    
    # Determine confidence level
    if output > THRESHOLD + 0.1:
        confidence_level = "High"
    elif output > THRESHOLD:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    return jsonify({
        'result': result,
        'confidence': round(confidence, 2),
        'confidence_level': confidence_level,
        'raw_score': output,
        'face_detected': face_detected
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
