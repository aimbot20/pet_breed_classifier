import os
from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import pandas as pd
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Get the absolute path to the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Global variables for model and class mapping
interpreter = None
class_mapping = None

def load_model_and_mapping():
    global interpreter, class_mapping
    try:
        # Load the TFLite model
        model_path = os.path.join(BASE_DIR, 'optimized_model.tflite')
        if not os.path.exists(model_path):
            return False
            
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Load class indices
        indices_path = os.path.join(BASE_DIR, 'class_indices.xlsx')
        if not os.path.exists(indices_path):
            return False
            
        class_df = pd.read_excel(indices_path)
        class_mapping = dict(zip(class_df['Class Index'], class_df['Class Name']))
        return True
    except Exception as e:
        print(f"Error loading model or mapping: {str(e)}")
        return False

def is_cat_breed(breed_name):
    cat_indicators = ['persian', 'siamese', 'maine', 'bengal', 'ragdoll', 'birman', 
                     'abyssinian', 'sphynx', 'manx', 'russian', 'bombay', 'himalayan']
    return any(indicator in breed_name.lower() for indicator in cat_indicators)

def preprocess_image(image):
    try:
        img = image.resize((224, 224))
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
        if interpreter is None or class_mapping is None:
            success = load_model_and_mapping()
            if not success:
                return jsonify({'error': 'Failed to load model or mapping'}), 500
        
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400
            
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(predictions[0])
        
        predicted_breed = class_mapping[predicted_class_index]
        is_cat = is_cat_breed(predicted_breed)
        animal_type = "I am a Cat" if is_cat else "I am a Dog"
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
    return jsonify({
        "status": "healthy",
        "cwd": os.getcwd(),
        "files": os.listdir(os.getcwd())
    })

if __name__ == '__main__':
    app.run() 