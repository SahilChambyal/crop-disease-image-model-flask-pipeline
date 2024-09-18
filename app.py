from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model (Pickle file)
MODEL_PATH = 'model.pkl'  # Provide the correct path to your pickle model file
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Define the class labels with the disease names based on your model's training data
CLASS_LABELS = [
    'Healthy', 'Powdery Mildew', 'Leaf Spot', 'Blight', 
    'Rust', 'Downy Mildew', 'Disease A', 'Disease B',
    'Disease C', 'Disease D', 'Disease E', 'Disease F', 
    'Disease G', 'Disease H', 'Disease I'
]  # Update with actual class labels

def prepare_image(image, target_size):
    # Resize and preprocess the image for prediction
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize if required by your model
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 64, 64, 3)
    return image

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image from the uploaded file
    image = Image.open(io.BytesIO(file.read()))
    
    # Preprocess the image and prepare it for prediction
    processed_image = prepare_image(image, target_size=(64, 64))  # Adjust target size to match model's expected input size

    # Run the image through the model to get the prediction
    preds = model.predict(processed_image)
    predicted_class_idx = np.argmax(preds)  # Get the index of the class with highest probability
    
    # Debugging print statements to verify output
    print(f"Predicted class index: {predicted_class_idx}")
    print(f"Number of class labels: {len(CLASS_LABELS)}")
    print(f"Prediction output shape: {preds.shape}")
    
    # Check if the predicted class index is within the range of CLASS_LABELS
    if predicted_class_idx >= len(CLASS_LABELS):
        return jsonify({'error': 'Predicted class index is out of range'}), 500

    predicted_class = CLASS_LABELS[predicted_class_idx]  # Map the index to the disease name

    # Return the prediction and the disease name as a JSON response
    return jsonify({
        'prediction': predicted_class,
        'confidence': float(np.max(preds))  # Optionally include the confidence score
    })

if __name__ == '__main__':
    app.run(debug=True)
