import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename # For secure file uploads

# --- Flask App Setup ---
app = Flask(__name__)

# Configure upload folder (create it if it doesn't exist)
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max upload size: 16MB

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Model Loading ---
MODEL_PATH = 'healthy_vs_rotten.h5'
IMG_SIZE = (224, 224) # Ensure this matches your model's input size
CLASS_NAMES = ['fresh', 'rotten'] # Ensure this matches your model's class order

# Load the model once when the app starts
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully for the Flask app.")
except Exception as e:
    print(f"Error loading model for Flask app: {e}")
    print("Please ensure 'healthy_vs_rotten.h5' is in the same directory as app.py.")
    model = None # Set model to None so routes can handle missing model

# --- Routes ---

@app.route('/')
def home():
    """Renders the home page (index.html)."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Renders the about page (about.html)."""
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles image uploads and predictions.
    GET: Renders the prediction page (you'll create this HTML next).
    POST: Processes the uploaded image and returns prediction.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500

            try:
                # Preprocess the image
                img = load_img(filepath, target_size=IMG_SIZE)
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
                img_array = img_array / 255.0 # Rescale pixels to [0,1]

                # Make prediction
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                confidence = predictions[0][predicted_class_index]

                predicted_class_name = CLASS_NAMES[predicted_class_index]

                # Clean up the uploaded file (optional, but good practice)
                os.remove(filepath)

                return jsonify({
                    'prediction': predicted_class_name,
                    'confidence': float(confidence), # Convert to float for JSON
                    'image_url': f'/static/uploads/{filename}' # This would be if you wanted to display the uploaded image
                })

            except Exception as e:
                print(f"Prediction error: {e}")
                return jsonify({'error': f'Error processing image: {e}'}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    
    # For GET request to /predict, we'll render a prediction HTML page
    return render_template('predict.html') # You will create predict.html next

@app.route('/contact')
def contact():
    """Renders a placeholder contact page."""
    # You'll create contact.html later if needed
    return render_template('contact.html')


# --- Run the Flask App ---
if __name__ == '__main__':
    # Use debug=True during development for auto-reloading and error messages
    # Set debug=False in production
    app.run(debug=True)
