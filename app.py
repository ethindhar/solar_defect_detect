import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="solar_panel_model.tflite")
interpreter.allocate_tensors()

# Get input and output details for the interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ['dust', 'snow', 'electrical damage', 'physical damage', 'bird', 'clean']

# Route to serve the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file:
        try:
            # Convert the FileStorage object into a BytesIO object
            img_bytes = io.BytesIO(file.read())

            # Use load_img to process the image
            img = load_img(img_bytes, target_size=(256, 256))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Ensure the output is of float type before processing
            output_data = np.array(output_data, dtype=np.float32)

            # Get the predicted class and confidence
            predicted_class_index = np.argmax(output_data[0])
            predicted_class = class_labels[predicted_class_index]
            confidence = float(output_data[0][predicted_class_index])

            # Return the result
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': f"Error processing image: {str(e)}"}), 500
    else:
        return jsonify({'error': 'No file uploaded'}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
