from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from io import BytesIO

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='optimized_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class indices
class_df = pd.read_excel('class_indices.xlsx')
class_names = class_df['class_name'].tolist()

def handler(event, context):
    """
    Netlify Function handler
    """
    # Parse the incoming request
    if event['httpMethod'] == 'GET':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': render_template('index.html')
        }
    
    elif event['httpMethod'] == 'POST':
        try:
            # Get the image file from the request
            image_file = request.files['image']
            image = Image.open(BytesIO(image_file.read()))
            
            # Preprocess the image
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Make prediction
            interpreter.set_tensor(input_details[0]['index'], image_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
            # Get the predicted class
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': jsonify({
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })
            }
            
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': jsonify({'error': str(e)})
            }
    
    return {
        'statusCode': 405,
        'headers': {'Content-Type': 'application/json'},
        'body': jsonify({'error': 'Method not allowed'})
    } 