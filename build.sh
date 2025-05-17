#!/bin/bash

# Install dependencies
pip install -r vercel_requirements.txt

# Create necessary directories
mkdir -p .vercel/output/static

# Copy static files
cp -r templates .vercel/output/static/
cp optimized_model.tflite .vercel/output/static/
cp class_indices.xlsx .vercel/output/static/

# Copy the application files
cp app.py .vercel/output/
cp vercel_requirements.txt .vercel/output/requirements.txt

echo "Build completed!" 