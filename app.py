import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/') #default page
def home():
    return render_template('home.html') #return a welcome message

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
        output = regmodel.predict(new_data)
        print(output)
        return jsonify({'prediction': float(output[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_input = scaler.transform(np.array(data).reshape(1, -1))
        output = regmodel.predict(final_input)[0]
        return render_template('home.html', prediction_text='Predicted House Price: ${:,.2f}k'.format(output))
    except Exception as e:
        return render_template('home.html', error_text='Error: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
