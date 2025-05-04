from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)
model = YOLO('yolo_banana_nutrient_best.pt')
classes = ['Boron', 'Calcium', 'Healthy', 'Iron', 'Magnesium', 'Manganese', 'Potassium', 'Sulfur', 'Zinc']

@app.route('/keepalive', methods=['GET'])
def api_health():
    return jsonify(Message="Success"), 200

@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file.stream).convert("RGB")

    # Run classification
    results = model(image)

    pred = results[0].probs.top1
    confidence = results[0].probs.data[pred].item()

    return jsonify({
        "class": classes[pred],
        "confidence": round(confidence, 4)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)