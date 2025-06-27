**Plant Disease Detection from Leaf Images**

This project uses deep learning to classify apple leaf diseases from images. It helps automate the identification of common diseases affecting apple plants, using a lightweight Convolutional Neural Network (CNN) deployed through a simple web interface using Streamlit.

*Objectives*

Classify four types of apple leaf conditions using image recognition.

Build a lightweight, fast CNN model with high accuracy.

Provide an intuitive web-based GUI to upload and diagnose leaf images.

 Tools & Technologies

Programming Language: Python

Libraries: TensorFlow, Keras, NumPy, OpenCV, PIL, Streamlit

Framework: Convolutional Neural Networks (CNN)

Dataset: PlantVillage Dataset

Deployment: Streamlit Web App

 Classes Used (from PlantVillage)

Apple Black Rot
Apple Cedar Rust
Apple Scab
Apple Healthy

 *How to Run*

1. Clone the repository

git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection

2. Install Dependencies

pip install -r requirements.txt

3. Train the Model (Optional)

python main.py

4. Run the Web App

streamlit run app.py

Project Structure

plant-disease-detection/
├── app.py                # Streamlit Web App
├── main.py               # Model training script
├── plant_model_small.h5  # Trained model (347 KB)
├── requirements.txt      # Required Python packages
├── README.md             # Project description
└── /PlantVillage         # Filtered dataset (4 apple classes)

 Sample Output

Upload an apple leaf image using the web interface

Model returns predicted disease and confidence score

Fast, responsive and mobile-friendly

Demo Video

 Watch the Demo

Deliverables

 Trained Model: plant_model_small.h5

 Source Code: main.py, app.py

 GUI App (Streamlit)

Filtered Dataset

 Project Report (PDF)

 Demo Video

 Future Work

Add more plant species and diseases

Convert to mobile app using TensorFlow Lite

Improve model accuracy using transfer learning

**Author**

S. Nivesh TejaLinkedIn | GitHub


