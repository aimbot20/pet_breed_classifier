from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import base64
from io import BytesIO
import os

# Get the directory where the function is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the TFLite model
model_path = os.path.join(current_dir, 'optimized_model.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class indices
class_indices_path = os.path.join(current_dir, 'class_indices.xlsx')
class_df = pd.read_excel(class_indices_path)
class_names = class_df['class_name'].tolist()

def handler(event, context):
    """
    Netlify Function handler
    """
    # Parse the incoming request
    if event['httpMethod'] == 'GET':
        # Read the HTML template
        with open(os.path.join(current_dir, 'templates', 'index.html'), 'r') as f:
            html_content = f.read()
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': html_content
        }
    
    elif event['httpMethod'] == 'POST':
        try:
            # Parse the body
            body = json.loads(event['body'])
            
            # Get the base64 encoded image
            image_data = body.get('image', '').split('base64,')[-1]
            image_bytes = base64.b64decode(image_data)
            
            # Open and preprocess the image
            image = Image.open(BytesIO(image_bytes))
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
            
            # Make prediction
            interpreter.set_tensor(input_details[0]['index'], image_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
            # Get the predicted class
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })
            }
            
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': str(e)})
            }
    
    return {
        'statusCode': 405,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({'error': 'Method not allowed'})
    } 