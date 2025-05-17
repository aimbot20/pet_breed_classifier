#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p .vercel/output/static

# Copy static files
cp -r templates .vercel/output/static/
cp optimized_model.tflite .vercel/output/static/
cp class_indices.xlsx .vercel/output/static/

# Copy the application files
cp app.py .vercel/output/
cp requirements.txt .vercel/output/

echo "Build completed!" 