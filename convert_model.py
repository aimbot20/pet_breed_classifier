import tensorflow as tf
import os

# Load the model
model = tf.keras.models.load_model('my_model_50epochs.keras')

# Configure the converter for optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Convert the model
tflite_model = converter.convert()

# Save the optimized TFLite model
with open('optimized_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted and optimized successfully!")

# Print size comparison
original_size = os.path.getsize('my_model_50epochs.keras') / (1024 * 1024)
new_size = os.path.getsize('optimized_model.tflite') / (1024 * 1024)
print(f"\nSize comparison:")
print(f"Original model: {original_size:.2f} MB")
print(f"Optimized model: {new_size:.2f} MB")
print(f"Size reduction: {((original_size - new_size) / original_size * 100):.1f}%") 