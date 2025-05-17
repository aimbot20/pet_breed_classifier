import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import sys
import gdown
import tempfile

app = Flask(__name__)

# Google Drive file IDs (you'll need to replace these with your actual file IDs)
MODEL_FILE_ID = "YOUR_MODEL_FILE_ID"  # You'll need to upload the model to Google Drive and get its ID
EXCEL_FILE_ID = "YOUR_EXCEL_FILE_ID"  # You'll need to upload the excel file to Google Drive and get its ID

# Global variables for model and class mapping
model = None
class_mapping = None
temp_dir = tempfile.mkdtemp()

def download_from_drive(file_id, output_path):
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading file: {str(e)}", file=sys.stderr)
        return False

def load_model_and_mapping():
    global model, class_mapping, temp_dir
    try:
        print("Starting model and mapping loading process...", file=sys.stderr)
        
        # Download and load the model
        model_path = os.path.join(temp_dir, 'my_model_50epochs.keras')
        if not os.path.exists(model_path):
            print("Downloading model file...", file=sys.stderr)
            if not download_from_drive(MODEL_FILE_ID, model_path):
                raise Exception("Failed to download model file")
        
        print("Loading model...", file=sys.stderr)
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully", file=sys.stderr)

        # Download and load class indices
        class_indices_path = os.path.join(temp_dir, 'class_indices.xlsx')
        if not os.path.exists(class_indices_path):
            print("Downloading class indices file...", file=sys.stderr)
            if not download_from_drive(EXCEL_FILE_ID, class_indices_path):
                raise Exception("Failed to download class indices file")
        
        print("Loading class mapping...", file=sys.stderr)
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
        print(f"Error during prediction: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 