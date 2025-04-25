from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('rice_deficiency_model.h5')
classes = ['Nitrogen(N)', 'Phosphorus(P)', 'Potassium(K)']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

@app.route('/keepalive', methods=['GET'])
def api_health():
    return jsonify(Message="Success"), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Ensure upload directory exists
    os.makedirs('uploads', exist_ok=True)

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(img)
    os.remove(filepath)  # cleanup

    class_idx = np.argmax(prediction[0])
    class_label = classes[class_idx]
    confidence = float(prediction[0][class_idx])

    return jsonify({'class': class_label, 'confidence': confidence})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)