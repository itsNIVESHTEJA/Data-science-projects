from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import PowerTransformer 
app = Flask(__name__)

# Load the pre-trained model (ensure the path is correct)
model = load(r'C:\Users\suppa\Desktop\coding\data_scince\ML\regression\regression_project-(cars)\cars_model1.joblib')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from the form
            present_price = float(request.form['present_price'])
            kms_driven = float(request.form['kms_driven'])
            fuel_type = request.form['fuel_type']
            seller_type = request.form['seller_type']
            transmission = request.form['transmission']
            owner = int(request.form['owner'])
            year = int(request.form['year'])

            # Feature engineering
            current_year = pd.Timestamp.now().year
            car_age = current_year - year

            # Encode categorical features
            fuel_type_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
            seller_type_map = {'Dealer': 0, 'Individual': 1}
            transmission_map = {'Manual': 0, 'Automatic': 1}

            fuel_type = fuel_type_map.get(fuel_type, 0)  # Default to Petrol if unknown
            seller_type = seller_type_map.get(seller_type, 0)  # Default to Dealer if unknown
            transmission = transmission_map.get(transmission, 0)  # Default to Manual if unknown

            # Prepare input data for prediction
            input_data = pd.DataFrame([[present_price, kms_driven, fuel_type, seller_type, transmission, owner, car_age]],
                                      columns=['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Car_Age'])

            # Apply the same preprocessing steps (log transformation for continuous features)
            input_data['Present_Price'] = np.log1p(input_data['Present_Price'])
            input_data['Kms_Driven'] = np.log1p(input_data['Kms_Driven'])
            
            
            #pt = PowerTransformer(method='yeo-johnson') 
            #input_data['Kms_Driven'] = pt.fit_transform(input_data['Kms_Driven']) 
            
            # If using Polynomial features, transform the data before prediction
            #from sklearn.preprocessing import PolynomialFeatures
            #poly = PolynomialFeatures(degree=2, include_bias=False)
            #input_data_poly = poly.fit_transform(input_data)

            # Make prediction using the model
            prediction = model.predict(input_data)
            #original_value = np.expm1(log_value)
            # Return the result
            return render_template('index.html', prediction_text=f'The predicted selling price of the car is: ₹{prediction[0]:,.2f}')

        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', prediction_text="Error occurred. Please check the input fields.")

if __name__ == "__main__":
    app.run(debug=True)
