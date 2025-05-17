import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import sys

app = Flask(__name__)

# Get the absolute path to the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# For Vercel deployment
STATIC_DIR = os.path.join(BASE_DIR, '.vercel', 'output', 'static')
if os.path.exists(STATIC_DIR):
    BASE_DIR = STATIC_DIR

# Global variables for model and class mapping
model = None
class_mapping = None

def load_model_and_mapping():
    global model, class_mapping
    try:
        print("Current working directory:", os.getcwd(), file=sys.stderr)
        print("BASE_DIR:", BASE_DIR, file=sys.stderr)
        
        # Load the model
        model_path = os.path.join(BASE_DIR, 'my_model_50epochs.keras')
        print("Looking for model at:", model_path, file=sys.stderr)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully", file=sys.stderr)

        # Load class indices and create a mapping dictionary
        class_indices_path = os.path.join(BASE_DIR, 'class_indices.xlsx')
        print("Looking for class indices at:", class_indices_path, file=sys.stderr)
        if not os.path.exists(class_indices_path):
            raise FileNotFoundError(f"Class indices file not found at {class_indices_path}")
        class_df = pd.read_excel(class_indices_path)
        class_mapping = dict(zip(class_df['Class Index'], class_df['Class Name']))
        print("Class mapping loaded successfully", file=sys.stderr)
        
        return True
    except Exception as e:
        print(f"Error loading model or mapping: {str(e)}", file=sys.stderr)
        return False

def is_cat_breed(breed_name):
    cat_indicators = ['persian', 'siamese', 'maine', 'bengal', 'ragdoll', 'birman', 
                     'abyssinian', 'sphynx', 'manx', 'russian', 'bombay', 'himalayan']
    return any(indicator in breed_name.lower() for indicator in cat_indicators)

def preprocess_image(image):
    # Resize the image to match the input size your model expects (assuming 224x224)
    img = image.resize((224, 224))
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, class_mapping
    
    # Load model and mapping if not already loaded
    if model is None or class_mapping is None:
        success = load_model_and_mapping()
        if not success:
            return jsonify({'error': 'Failed to load model or mapping'}), 500
    
    try:
        # Get the image from the POST request
        file = request.files['file']
        # Open and preprocess the image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        # Get the breed name from our mapping dictionary
        predicted_breed = class_mapping[predicted_class_index]
        # Determine if it's a cat or dog breed
        is_cat = is_cat_breed(predicted_breed)
        animal_type = "I am a Cat" if is_cat else "I am a Dog"
        # Replace underscores with spaces and capitalize words for better display
        predicted_breed = predicted_breed.replace('_', ' ').title()
        confidence = float(predictions[0][predicted_class_index])
        
        return jsonify({
            'breed': predicted_breed,
            'confidence': f'{confidence:.2%}',
            'animal_type': animal_type
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Load model and mapping at startup
load_model_and_mapping()

if __name__ == '__main__':
    app.run(debug=True) 