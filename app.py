import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Model file ka naam
MODEL_PATH = "voice_model.tflite"

# TFLite Model load karein
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

@app.route('/')
def home():
    return "Mesh AI API is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['audio_data']
        input_array = np.array(data, dtype=np.float32).reshape(1, 16000, 1)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        
        prediction = interpreter.get_tensor(output_details[0]['index'])
        result = int(np.argmax(prediction))

        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)