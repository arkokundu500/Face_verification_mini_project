# from flask import Flask, render_template, request, jsonify  # Import Flask and related modules
# import torch  # Import PyTorch
# from torchvision import transforms  # Import torchvision transforms
# from PIL import Image  # Import PIL for image processing
# import base64  # Import base64 for encoding/decoding
# import cv2  # Import OpenCV for image processing
# import numpy as np  # Import numpy
# from models.discriminator import Discriminator  # Import custom Discriminator model

# app = Flask(__name__)  # Initialize Flask app

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available

# # Load discriminator model
# model = Discriminator()  # Instantiate Discriminator model
# model.load_state_dict(torch.load('saved_models/discriminator.pth', map_location=device))  # Load model weights
# model.to(device)  # Move model to device
# model.eval()  # Set model to evaluation mode

# # Image preprocessing transform
# transform = transforms.Compose([  # Compose image transforms
#     transforms.Resize(64),  # Resize to 64x64
#     transforms.CenterCrop(64),  # Center crop to 64x64
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize tensor
# ])

# # Load OpenCV face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Load Haar cascade for face detection

# def process_image(img):
#     """Convert image to OpenCV format and detect faces"""
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # Detect faces
#     
#     if len(faces) == 0:  # If no faces detected
#         return None  # Return None
#     
#     (x, y, w, h) = faces[0]  # Get first detected face
#     return img[y:y+h, x:x+w]  # Return cropped face region

# def verify_face(face_img):
#     """Run face verification model"""
#     face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL
#     face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Apply transforms and add batch dimension
#     
#     with torch.no_grad():  # Disable gradient calculation
#         output = model(face_tensor).squeeze()  # Run model and remove batch dimension
#         return output.item()  # Return output as Python float

# @app.route('/')
# def home():
#     return render_template('index-new.html')  # Render home page

# @app.route('/verify', methods=['POST'])
# def verify():
#     try:
#         img = None  # Initialize image variable
#         
#         # Handle JSON base64 payload
#         if request.content_type == 'application/json':  # If request is JSON
#             data = request.get_json()  # Parse JSON
#             img_data = data['image']  # Get image data
#             
#             # Remove header if present
#             if ',' in img_data:  # If base64 header present
#                 _, encoded = img_data.split(",", 1)  # Split header and data
#             else:
#                 encoded = img_data  # Use data as is
#                 
#             img_bytes = base64.b64decode(encoded)  # Decode base64
#             nparr = np.frombuffer(img_bytes, np.uint8)  # Convert to numpy array
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode image
#         
#         # Handle multipart file upload
#         elif 'image' in request.files:  # If image file uploaded
#             file = request.files['image']  # Get file
#             img_bytes = file.read()  # Read file bytes
#             nparr = np.frombuffer(img_bytes, np.uint8)  # Convert to numpy array
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode image
#         
#         else:
#             return jsonify({'error': 'Unsupported request format'}), 400  # Return error if format unsupported
#         
#         # Process image and detect face
#         face_img = process_image(img)  # Detect face in image
#         if face_img is None:  # If no face detected
#             return jsonify({'error': 'No face detected. Please try again.'}), 400  # Return error
#         
#         # Verify face
#         confidence = verify_face(face_img)  # Get model confidence
#         threshold = 0.8  # Set verification threshold
#         result = "Verified Access" if confidence > threshold else "Not Verified"  # Determine result
#         
#         return jsonify({'result': result, 'confidence': confidence})  # Return result and confidence
#     
#     except Exception as e:  # Handle exceptions
#         return jsonify({'error': str(e)}), 500  # Return error message

# if __name__ == '__main__':  # If script is run directly
#     app.run(debug=True, host='0.0.0.0')  # Run Flask app in debug mode

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
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5  # Verification threshold
MODEL_PATH = 'models/saved_models_fixed/discriminator.pth'  # Updated to match new training output

# Global model variable
model = None

def load_model():
    """Load the trained discriminator model with proper error handling"""
    global model
    try:
        # Initialize model with same architecture as training
        model = Discriminator()
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return False
            
        # Load state dictionary with proper device mapping
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Handle potential key mismatches between training and inference
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        logger.info(f"Using device: {device}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None
        return False

def enhanced_preprocess_image(img):
    """
    Enhanced image preprocessing pipeline aligned with training data augmentation
    Includes face detection, cropping, and normalization matching training transforms
    """
    try:
        # Convert PIL to OpenCV format (RGB to BGR)
        img_cv = np.array(img)
        img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR conversion
        
        # Initialize Haar Cascade face detector with same parameters as training
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters for better accuracy
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # Same as training
            minNeighbors=5,       # Reduced false positives
            minSize=(30, 30),     # Minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Face detection and cropping logic
        if len(faces) == 0:
            # Fallback: center crop if no face detected
            height, width = img_cv.shape[:2]
            min_dim = min(height, width)
            startx = width//2 - min_dim//2
            starty = height//2 - min_dim//2
            face_img = img_cv[starty:starty+min_dim, startx:startx+min_dim]
            face_detected = False
            logger.warning("No face detected, using center crop")
        else:
            # Use the largest detected face
            (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
            
            # Expand face region by 20% for better coverage (matching training)
            expand = 0.2
            x = max(0, int(x - w * expand))
            y = max(0, int(y - h * expand))
            w = min(img_cv.shape[1] - x, int(w * (1 + 2*expand)))
            h = min(img_cv.shape[0] - y, int(h * (1 + 2*expand)))
            
            face_img = img_cv[y:y+h, x:x+w]
            face_detected = True
            logger.info(f"Face detected at ({x}, {y}, {w}, {h})")
        
        # Convert back to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_img)
        
        # Apply same transformations as training (without data augmentation)
        # This matches the enhanced training transforms but without random augmentations
        transform = transforms.Compose([
            transforms.Resize(64),                    # Same as training
            transforms.CenterCrop(64),               # Consistent with training
            transforms.ToTensor(),                   # Convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Same normalization
        ])
        
        # Apply transformations and add batch dimension
        img_tensor = transform(face_pil).unsqueeze(0)
        
        return img_tensor, face_detected
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise

def calculate_confidence_metrics(raw_score, threshold=THRESHOLD):
    """
    Calculate comprehensive confidence metrics matching training evaluation
    """
    # Ensure score is within valid range
    confidence_percentage = min(100, max(0, raw_score * 100))
    
    # Determine verification result
    verified = raw_score > threshold
    result = "Verified Access" if verified else "Not Verified"
    
    # Enhanced confidence level categorization (matching training logic)
    if raw_score > threshold + 0.15:  # High confidence threshold
        confidence_level = "High"
    elif raw_score > threshold + 0.05:  # Medium confidence threshold
        confidence_level = "Medium"
    elif raw_score > threshold - 0.05:  # Low but above threshold
        confidence_level = "Low" if verified else "Very Low"
    else:
        confidence_level = "Very Low"
    
    return {
        'result': result,
        'confidence': round(confidence_percentage, 2),
        'confidence_level': confidence_level,
        'raw_score': round(raw_score, 4),
        'verified': verified
    }

@app.route('/')
def home():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/verify', methods=['POST'])
def verify():
    """
    Enhanced verification endpoint with comprehensive error handling
    Supports both canvas data (webcam capture) and file uploads
    """
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded")
        return jsonify({
            'error': 'Model not loaded. Please check server configuration.',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    try:
        img = None
        input_source = None
        
        # Handle canvas data from webcam capture
        if 'canvas_data' in request.form:
            try:
                # Extract and decode base64 image data
                canvas_data = request.form['canvas_data']
                if ',' in canvas_data:
                    image_data = canvas_data.split(',')[1]
                else:
                    image_data = canvas_data
                    
                img_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                input_source = "camera"
                logger.info("Processing camera capture")
                
            except Exception as e:
                logger.error(f"Canvas data processing failed: {str(e)}")
                return jsonify({
                    'error': f'Invalid canvas data: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }), 400
        
        # Handle file upload
        elif 'image' in request.files:
            try:
                file = request.files['image']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                    
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                input_source = "file_upload"
                logger.info(f"Processing file upload: {file.filename}")
                
            except Exception as e:
                logger.error(f"File upload processing failed: {str(e)}")
                return jsonify({
                    'error': f'Invalid image file: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }), 400
        else:
            return jsonify({
                'error': 'No image data provided. Please capture or upload an image.',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Preprocess image with enhanced pipeline
        try:
            img_tensor, face_detected = enhanced_preprocess_image(img)
            img_tensor = img_tensor.to(device)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return jsonify({
                'error': f'Image processing failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Perform inference with the enhanced model
        try:
            with torch.no_grad():  # Disable gradient computation for inference
                model.eval()  # Ensure model is in evaluation mode
                raw_output = model(img_tensor)
                
                # Handle different output formats
                if isinstance(raw_output, torch.Tensor):
                    if raw_output.dim() > 0:
                        raw_score = raw_output.item()
                    else:
                        raw_score = float(raw_output)
                else:
                    raw_score = float(raw_output)
                    
                logger.info(f"Model inference completed. Raw score: {raw_score}")
                
        except Exception as e:
            logger.error(f"Model inference failed: {str(e)}")
            return jsonify({
                'error': f'Model inference failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        # Calculate comprehensive confidence metrics
        confidence_metrics = calculate_confidence_metrics(raw_score)
        
        # Prepare response with enhanced information
        response = {
            'result': confidence_metrics['result'],
            'confidence': confidence_metrics['confidence'],
            'confidence_level': confidence_metrics['confidence_level'],
            'raw_score': confidence_metrics['raw_score'],
            'face_detected': face_detected,
            'verified': confidence_metrics['verified'],
            'input_source': input_source,
            'threshold': THRESHOLD,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log verification attempt
        logger.info(f"Verification completed: {response['result']} "
                   f"(confidence: {response['confidence']}%, "
                   f"face_detected: {face_detected})")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Unexpected error in verification: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# Initialize the application
def initialize_app():
    """Initialize the Flask application with proper setup"""
    # Load the model on startup
    if not load_model():
        logger.warning("Failed to load model during startup. Application will run but verification will not work.")
    
    logger.info("Smart Doorbell Facial Verification System initialized")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Device: {device}")
    logger.info(f"Verification threshold: {THRESHOLD}")

if __name__ == '__main__':
    # Initialize the application
    initialize_app()
    
    # Run the Flask application
    app.run(
        debug=True, 
        host='0.0.0.0',  # Allow external connections
        port=5000,
        threaded=True    # Enable threading for better performance
    )
