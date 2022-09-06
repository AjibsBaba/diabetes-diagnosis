import joblib
import numpy as np
from flask import request, render_template, Flask

model = 'modelExport.pkl'
classifier = joblib.load(model)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE='Lax',
)


@app.route('/')
def landing():
    return render_template('home.html')


@app.route('/predict', methods=["POST", "GET"])
def predict():
    return render_template('test.html')


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
        return render_template('test.html', prediction=pred)
