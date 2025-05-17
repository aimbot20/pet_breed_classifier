import os
from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import pandas as pd
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Get the absolute path to the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print(f"Base directory: {BASE_DIR}")

# Global variables for model and class mapping
interpreter = None
class_mapping = None

def load_model_and_mapping():
    global interpreter, class_mapping
    try:
        # Load the TFLite model
        model_paths = [
            os.path.join(BASE_DIR, 'static', 'optimized_model.tflite'),
            os.path.join(BASE_DIR, 'optimized_model.tflite'),
            '/var/task/static/optimized_model.tflite',
            '/var/task/optimized_model.tflite'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            print(f"Trying model path: {model_path}")
            if os.path.exists(model_path):
                interpreter = tflite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                print(f"Model loaded successfully from {model_path}")
                model_loaded = True
                break
        
        if not model_loaded:
            raise FileNotFoundError("Model file not found in any of the expected locations")

        # Load class indices and create a mapping dictionary
        indices_paths = [
            os.path.join(BASE_DIR, 'static', 'class_indices.xlsx'),
            os.path.join(BASE_DIR, 'class_indices.xlsx'),
            '/var/task/static/class_indices.xlsx',
            '/var/task/class_indices.xlsx'
        ]
        
        indices_loaded = False
        for indices_path in indices_paths:
            print(f"Trying class indices path: {indices_path}")
            if os.path.exists(indices_path):
                class_df = pd.read_excel(indices_path)
                class_mapping = dict(zip(class_df['Class Index'], class_df['Class Name']))
                print(f"Class mapping loaded successfully from {indices_path}")
                indices_loaded = True
                break
        
        if not indices_loaded:
            raise FileNotFoundError("Class indices file not found in any of the expected locations")
        
        return True
    except Exception as e:
        print(f"Error loading model or mapping: {str(e)}")
        print(f"Current directory contents: {os.listdir(BASE_DIR)}")
        if os.path.exists(os.path.join(BASE_DIR, 'static')):
            print(f"Static directory contents: {os.listdir(os.path.join(BASE_DIR, 'static'))}")
        print(f"Var task contents (if exists): {os.listdir('/var/task') if os.path.exists('/var/task') else 'Not available'}")
        if os.path.exists('/var/task/static'):
            print(f"Var task static contents: {os.listdir('/var/task/static')}")
        return False

def is_cat_breed(breed_name):
    cat_indicators = ['persian', 'siamese', 'maine', 'bengal', 'ragdoll', 'birman', 
                     'abyssinian', 'sphynx', 'manx', 'russian', 'bombay', 'himalayan']
    return any(indicator in breed_name.lower() for indicator in cat_indicators)

def preprocess_image(image):
    try:
        # Resize the image to match the input size your model expects (224x224)
        img = image.resize((224, 224))
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, 0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global interpreter, class_mapping
    
    try:
        # Load model and mapping if not already loaded
        if interpreter is None or class_mapping is None:
            print("Loading model and mapping...")
            success = load_model_and_mapping()
            if not success:
                return jsonify({'error': 'Failed to load model or mapping. Check server logs for details.'}), 500
        
        # Get the image from the POST request
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400
            
        # Open and preprocess the image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
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

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Try to load model at startup in development
    if not os.getenv('VERCEL_ENV'):
        load_model_and_mapping()
    app.run(debug=True) 