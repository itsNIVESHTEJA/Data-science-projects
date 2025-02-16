from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the pre-trained model and scaler
model = load(r'C:\Users\suppa\Desktop\coding\data_scince\ML\classification\Banking\classification_project-2(banking _loan)\model.joblib')
scaler = load(r'C:\Users\suppa\Desktop\coding\data_scince\ML\classification\Banking\classification_project-2(banking _loan)\scaler.joblib')

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # Retrieve form data
            no_of_dep = int(request.form['no_of_dep'])
            grad = request.form['grad']
            self_emp = request.form['self_emp']
            Annual_Income = float(request.form['Annual_Income'])
            Loan_Amount = float(request.form['Loan_Amount'])
            Loan_Dur = int(request.form['Loan_Dur'])
            Cibil = int(request.form['Cibil'])
            Assets = float(request.form['Assets'])

            # Encode categorical variables
            grad_s = 0 if grad == 'Graduated' else 1
            emp_s = 0 if self_emp == 'No' else 1

            # Prepare the input data for prediction
            pred_data = pd.DataFrame(
                [[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
                columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets']
            )

            # Scale the data
            pred_data_scaled = scaler.transform(pred_data)

            # Make the prediction
            prediction = model.predict(pred_data_scaled)

            # Determine the result
            if prediction[0] == 1:
                result = "Loan is Approved"
            else:
                result = "Loan is Rejected"

        except Exception as e:
            result = f"Error: {e}"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
