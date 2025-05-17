import os
from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Get the absolute path to the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print(f"Current BASE_DIR: {BASE_DIR}")

# Global variables for model and class mapping
interpreter = None
class_mapping = None

def load_model_and_mapping():
    global interpreter, class_mapping
    try:
        # Load the TFLite model
        model_path = os.path.join(BASE_DIR, 'optimized_model.tflite')
        print(f"Looking for model at: {model_path}")
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            print(f"Current directory contents: {os.listdir(BASE_DIR)}")
            return False
            
        print("Loading model...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully!")
        
        # Load class indices
        indices_path = os.path.join(BASE_DIR, 'class_indices.xlsx')
        print(f"Looking for class indices at: {indices_path}")
        if not os.path.exists(indices_path):
            print(f"Class indices file not found at {indices_path}")
            return False
            
        print("Loading class indices...")
        class_df = pd.read_excel(indices_path)
        class_mapping = dict(zip(class_df['Class Index'], class_df['Class Name']))
        print("Class indices loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model or mapping: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir(os.getcwd())}")
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
            print("Model or mapping not loaded, attempting to load...")
            success = load_model_and_mapping()
            if not success:
                return jsonify({'error': 'Failed to load model or mapping. Check server logs for details.'}), 500
        
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400
            
        print("Processing uploaded image...")
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        print("Running inference...")
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
        
        print(f"Prediction successful: {predicted_breed} ({confidence:.2%})")
        return jsonify({
            'breed': predicted_breed,
            'confidence': f'{confidence:.2%}',
            'animal_type': animal_type
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    try:
        files = os.listdir(os.getcwd())
        return jsonify({
            "status": "healthy",
            "cwd": os.getcwd(),
            "files": files,
            "base_dir": BASE_DIR,
            "base_dir_contents": os.listdir(BASE_DIR)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Try to load model at startup
    print("Starting application...")
    success = load_model_and_mapping()
    if success:
        print("Initial model and mapping load successful!")
    else:
        print("Failed to load model and mapping at startup")
    app.run(debug=True) 