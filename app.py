from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

app = Flask(__name__)

# Load the brain and squisher
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Create a folder for images if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the long string of 30 numbers from the new 'full_data' box
        raw_data = request.form['full_data']
        input_data = [float(x.strip()) for x in raw_data.split(',')]
        
        if len(input_data) != 30:
            return f"<h3>Error: You entered {len(input_data)} values, but the robot needs exactly 30!</h3>"

        # Squish and Predict
        scaled_data = scaler.transform([np.array(input_data)])
        prediction = model.predict(scaled_data)
        
        result = "HEALTHY (Benign)" if prediction[0] == 1 else "SICK (Malignant)"
        return render_template('index.html', prediction_text=f'Diagnosis: {result}')
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3>"

if __name__ == "__main__":
    app.run(debug=True)