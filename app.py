import joblib
from flask import request, render_template, Flask
import numpy as np

model = 'gradientBoosting.pkl'
classifier = joblib.load(model)

app = Flask(__name__)
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE='Lax',
)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/entry')
def entry():
    return render_template('model.html')


@app.route('/diagnosis', methods=["POST", "GET"])
def diagnosis():
    if request.method == 'POST':
        pregnancy = int(0)
        glucose = int(request.form['Glucose'])
        bloodPressure = int(request.form['BloodPressure'])
        skinThickness = int(20)
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])
        data = np.array([[pregnancy, glucose, bloodPressure, skinThickness,
                          insulin, bmi, dpf, age]])
        pred = classifier.predict(data)
        return render_template('model.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=False)
