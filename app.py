import os
import numpy as np
# TensorFlow ki jagah tflite_runtime use karein jo light aur fast hai
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Local machine par testing ke liye agar full tensorflow ho
    from tensorflow import lite as tflite

from flask import Flask, request, jsonify

app = Flask(__name__)

# Model file ka path
MODEL_PATH = "voice_model.tflite"

# TFLite Model load karne ka logic
if os.path.exists(MODEL_PATH):
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
else:
    interpreter = None
    print(f"Error: {MODEL_PATH} nahi mili! Please check your repository.")

@app.route('/')
def home():
    return "Mesh AI API is Live on TFLite Runtime (Python 3.10)!"

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({"status": "error", "message": "Model not loaded on server"})
        
    try:
        # Request se data nikalna
        json_data = request.json
        if not json_data or 'audio_data' not in json_data:
            return jsonify({"status": "error", "message": "audio_data key missing in JSON"})
            
        data = json_data['audio_data']
        
        # Audio input ko process karein (1, 16000, 1) shape mein
        input_array = np.array(data, dtype=np.float32).reshape(1, 16000, 1)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        
        # Prediction result
        prediction = interpreter.get_tensor(output_details[0]['index'])
        result = int(np.argmax(prediction))

        return jsonify({
            "status": "success", 
            "prediction": result,
            "message": "AI voice processing complete"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    # Render dynamic port use karta hai
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)