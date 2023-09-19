from flask import Flask, request, render_template
import numpy as np
import pickle
import os

print("Current working directory:", os.getcwd())
# Rest of your code...
# Loading Models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
# Creating Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = float(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
        transform_features = preprocessor.transform(features)
        predicted_value = dtr.predict(transform_features)
        
        return render_template('index.html', predicted_value=predicted_value[0])

# Python Main
if __name__ == '__main__':
    app.run(debug=True)
