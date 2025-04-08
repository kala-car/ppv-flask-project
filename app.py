from flask import Flask, request, jsonify, send_file
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Load trained model and scaler
scaler = joblib.load('scaler.pkl')
model = keras.models.load_model('ppv_ann_model.h5', 
                                custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError})

# Serve the HTML file
@app.route('/')
def index():
    return send_file('E:/Project/index.html')

# Prediction function
def predict_ppv(user_input):
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)
    prediction = model.predict(user_input)
    return float(prediction[0][0])

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    test_input = data['features']
    
    predicted_ppv = predict_ppv(test_input)
    
    return jsonify({
        'predicted_ppv': predicted_ppv
    })

if __name__ == '__main__':
    app.run(debug=True)
