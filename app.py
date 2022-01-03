import joblib
from flask import request, render_template, Flask
import numpy as np

model = 'gradientBoosting.pkl'
classifier = joblib.load(model)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE='Lax',
)


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/predict', methods=["POST", "GET"])
def predict():
    return render_template('predict.html')


@app.route('/diagnosis', methods=["POST", "GET"])
def diagnosis():
    if request.method == 'POST':
        Pregnancies = int(0)
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(20)
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(0.6)
        Age = int(request.form['Age'])
        data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age]])
        pred = classifier.predict(data)
        return render_template('predict.html', prediction=pred)


if __name__ == '__main__':
    app.run(host='192.168.0.105', port=9180, debug=True)
