
# 🧠 Brain Tumor Classification using Deep Learning (CNN + Flask Web App)

## 📘 Overview
This project focuses on **classifying brain MRI images** to detect the presence and type of **brain tumors** using a **Convolutional Neural Network (CNN)**.  
The goal is to assist in **early detection and diagnosis of brain tumors** by automating MRI image analysis.  
A **Flask web application** enables users to **upload MRI images** and receive real-time classification results.

---

## 🧠 Key Features
- Deep Learning-based **CNN model** for brain tumor detection and classification  
- **Data preprocessing & augmentation** for robust training  
- **Model evaluation** using accuracy, confusion matrix, ROC curve, precision, recall, and F1-score  
- **Flask Web App** for real-time MRI image prediction  
- Responsive **HTML/CSS frontend** for a smooth user experience  
- Supports **multiple brain tumor types** (e.g., glioma, meningioma, pituitary tumor)

---

## 🧩 Project Workflow
1. **Data Collection & Preprocessing**
   - Collected MRI images and organized them into tumor classes.
   - Applied resizing, normalization, and augmentation for model robustness.

2. **Model Building**
   - Implemented a **CNN architecture** using TensorFlow/Keras.
   - Used convolutional layers, MaxPooling, Dropout, and Dense layers.
   - Trained the model with Adam optimizer and categorical cross-entropy loss.

3. **Model Evaluation**
   - Evaluated using metrics: **accuracy, precision, recall, F1-score**, confusion matrix, and ROC curves.
   - Visualized performance metrics with Matplotlib/Seaborn.

4. **Deployment**
   - Integrated the trained model into a **Flask backend**.
   - Developed a **responsive HTML/CSS frontend** for uploading MRI images.
   - Displayed predictions and confidence scores in real-time.

---

## ⚙️ Tech Stack
| Category | Tools & Libraries |
|----------|----------------- |
| Programming | Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV |
| Web Framework | Flask |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML, CSS |

---

## 📊 Results
- Achieved **high classification accuracy** on MRI test images.
- Real-time predictions with a user-friendly interface.

| Metric | Value (Example) |
|--------|----------------|
| Training Accuracy | 98.2% |
| Validation Accuracy | 95.6% |
| F1-Score | 0.94 |

*(Replace with your actual results.)*

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App
```bash
python app.py
```

### 4️⃣ Access the Web Interface
Open your browser at:  
👉 **http://127.0.0.1:5000/**  

Upload a brain MRI image and get real-time tumor classification.

---

## 📁 Project Structure
```
brain-tumor-classification/
│
├── dataset/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
├── models/
│   └── brain_tumor_model.keras
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── brain_tumor_cnn.ipynb
├── requirements.txt
└── README.md
```

---

## 📈 Future Enhancements
- Use **transfer learning (VGG16/ResNet)** for improved accuracy  
- Add **Grad-CAM** for explainable AI to visualize tumor regions  
- Deploy on **cloud platforms (AWS / Render / Hugging Face Spaces)**  
- Extend to **predict tumor size and severity**

---

## 👨‍💻 Author
**S. Nivesh Teja**  
📧 [tsaikumar158@gmail.com](mailto:tsaikumar158@gmail.com)  
🔗 [GitHub](https://github.com/your-username)  
🔗 [LinkedIn](https://www.linkedin.com/in/s-niveshteja)

---

⭐ **If you find this project useful, consider giving it a star!**
