import os, json
import joblib
from flask import *
import numpy as np
import pandas as pd

model = 'gradientBoosting.pkl'
classifier = joblib.load(model)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    if request.method == 'POST':
        pregnancy = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        bloodPressure = int(request.form['BloodPressure'])
        skinThickness = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])

        data = np.array([[pregnancy, glucose, bloodPressure, skinThickness, insulin, bmi, dpf, age]])
        prediction = classifier.predict(data)
        return render_template_string('model.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
