from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import gc

app = Flask(__name__, 
            static_folder='static',
            template_folder='thesis-main')

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["https://upcyclerender.onrender.com", "http://localhost:8000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Create the model architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(480, 480, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    return model

# Initialize model as None
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = create_model()
        model_path = os.path.join(os.path.dirname(__file__), 'wasteModel.h5')
        model.load_weights(model_path)
        print("Model loaded successfully")

def preprocess_image(image_data, img_size=(480, 480)):
    try:
        # Convert image to RGB if necessary
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        # Resize image
        image = image.resize(img_size, Image.Resampling.LANCZOS)
        # Convert to numpy array and normalize
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model')
def model_page():
    return render_template('pages/model.html')

@app.route('/about')
def about():
    return render_template('pages/about.html')

@app.route('/contact')
def contact():
    return render_template('pages/contact.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Load model if not already loaded
        load_model()
        
        print("Received a frame")
        image_data = file.read()
        image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(image, verbose=0)
        prediction_index = np.argmax(prediction)
        prediction_confidence = prediction[0][prediction_index]
        prediction_class = 'Biodegradable' if prediction_index == 0 else 'Non-Biodegradable'
        
        print(f"Prediction: {prediction_class}, Confidence: {prediction_confidence * 100:.2f}%")
        
        # Clear memory
        del image
        del prediction
        gc.collect()
        tf.keras.backend.clear_session()
        
        return jsonify({
            "prediction": prediction_class,
            "confidence": float(prediction_confidence) * 100
        })
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/ws', methods=['POST'])
def websocket_endpoint():
    try:
        data = request.get_data()
        load_model()
        image = preprocess_image(data)
        
        prediction = model.predict(image, verbose=0)
        prediction_index = np.argmax(prediction)
        prediction_confidence = prediction[0][prediction_index]
        prediction_class = 'Biodegradable' if prediction_index == 0 else 'Non-Biodegradable'
        
        # Clear memory
        del image
        del prediction
        gc.collect()
        tf.keras.backend.clear_session()
        
        return jsonify({"prediction": prediction_class, "confidence": float(prediction_confidence) * 100})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host="0.0.0.0", port=port)