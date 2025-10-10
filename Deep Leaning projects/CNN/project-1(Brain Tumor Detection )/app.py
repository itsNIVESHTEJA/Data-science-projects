from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
MODEL_PATH = r"C:\Users\suppa\Desktop\MCA Projects\veeresh\brain tumor web app\model\brain_tumor_resnet50.h5"
model = load_model(MODEL_PATH)

# Classes
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.route('/')
def index():
    # Render without prediction initially
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", prediction="No file selected")

    # Save uploaded file
    basepath = os.path.dirname(__file__)
    upload_folder = os.path.join(basepath, 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = round(100 * np.max(preds), 2)

    # Render index.html with prediction
    return render_template("index.html",
                           prediction=predicted_class,
                           confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
