#  Loan Approval Prediction App

A web-based machine learning application built using **Flask** and **HTML** to predict whether a loan application will be approved based on user input. This project demonstrates the deployment of a machine learning model via a simple web interface.

---

##  Features

- Collects user input through a responsive HTML form
- Predicts loan approval using a trained machine learning model
- Flask handles backend logic and model inference
- Displays prediction result instantly on the same page
- Easily extendable to support model retraining or database logging

---

## 🛠 Tech Stack

- **Frontend**: HTML, CSS (optional styling)
- **Backend**: Python, Flask
- **ML Libraries**: Scikit-learn, Pandas, NumPy, Joblib/Pickle
- **Model**: Logistic Regression / Random Forest (or any classification model)

---

##  Project Structure
loan-approval-app/
├── app.py # Flask application
├── model.pkl # Trained ML model (serialized)
├── templates/
│ └── index.html # HTML form and results display
├── static/ # (Optional CSS/JS files)
├── README.md # Project documentation
└── requirements.txt # Dependencies

