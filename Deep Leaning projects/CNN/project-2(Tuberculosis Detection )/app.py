import os
import numpy as np
import tensorflow as tf
import pydicom
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Load the trained CNN model
MODEL_PATH = "C:/Users/suppa/Desktop/coding/data_scince/Deep_Learning/CNN/TB classification CNN project/TB_Classification_WebApp/model/tb_classification_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dicom", "dcm"}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image function
def preprocess_image(image_path):
    file_ext = image_path.split(".")[-1].lower()

    if file_ext in ["dcm", "dicom"]:  # DICOM processing
        dicom_data = pydicom.dcmread(image_path)
        img = dicom_data.pixel_array  # Extract image data
        img = cv2.resize(img, (224, 224))  # Resize to match model input
        img = np.stack((img,) * 3, axis=-1)  # Convert grayscale to 3-channel RGB
    else:  # PNG/JPG processing
        img = Image.open(image_path).convert("RGB")  # Ensure RGB format
        img = img.resize((224, 224))

    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Preprocess image
        img = preprocess_image(file_path)

        # Get prediction
        prediction = model.predict(img)[0][0]
        result = "TB Detected" if prediction > 0.5 else "No TB Detected"

        return jsonify({"prediction": result, "confidence": float(prediction)})
    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    app.run(debug=True)
