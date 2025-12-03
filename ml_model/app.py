import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from io import BytesIO
import json
import joblib

app = Flask(__name__)
CORS(app)


# Load Random Forest Model
model = joblib.load("Disease_prediction_model.pkl")
print("Random Forest model loaded successfully.")


# Load Class Labels
with open("./labels.json") as f:
    class_indices = json.load(f)

# Convert index -> class name
labels = {v: k for k, v in class_indices.items()}
print("Labels loaded:", labels)


# Load MobileNet Feature Extractor
feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)
print("MobileNetV2 feature extractor loaded.")


# Image Preprocessing Function
def preprocess_image(file):
    img = load_img(BytesIO(file.read()), target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)   
    img = np.expand_dims(img, axis=0)

    features = feature_extractor.predict(img)
    return features


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    print("Received keys:", request.files.keys())

    if "file" in request.files:
        file = request.files["file"]
    elif "image" in request.files:
        file = request.files["image"]
    else:
        return jsonify({"error": "No image provided. Use 'file' or 'image' field."}), 400

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        features = preprocess_image(file)

        pred_index = model.predict(features)[0]
        predicted_label = labels[pred_index]

        confidence = model.predict_proba(features)[0][pred_index]
        print(f"Predicted: {predicted_label} ({confidence:.2%})")
        return jsonify({
            "prediction": predicted_label,
            "confidence": float(confidence),
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


# Start Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
