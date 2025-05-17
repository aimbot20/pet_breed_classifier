# Pet Breed Classifier Web App

This web application uses a deep learning model to classify cat and dog breeds from uploaded images. The model can identify 132 different breeds with high accuracy.

## Features

- Modern, responsive web interface
- Drag and drop image upload
- Real-time image preview
- Breed prediction with confidence score
- Support for various image formats (PNG, JPG, JPEG)

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the following files are present in the root directory:
- `my_model_50epochs.keras` (the trained model)
- `class_indices.xlsx` (breed mapping file)

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Click the upload area or drag and drop an image of a cat or dog
2. Preview the image
3. Click "Predict Breed" to get the classification result
4. View the predicted breed and confidence score
5. Use the "Reset" button to try another image

## Technical Details

- Backend: Flask (Python)
- Frontend: HTML, JavaScript, Tailwind CSS
- Model: TensorFlow/Keras
- Image processing: PIL (Python Imaging Library)

## Deployment

The application is ready to be deployed on Vercel. Follow these steps:

1. Create a GitHub repository and push the code
2. Connect your GitHub repository to Vercel
3. Configure the build settings in Vercel
4. Deploy the application

## License

MIT License 