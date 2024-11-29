from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import time
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('xgboost.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    # Check if a file is sent
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    # Get the uploaded file
    file = request.files['file']
    
    # Load the file into a Pandas DataFrame
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error reading the CSV file: {str(e)}'}), 400
    
    # Log the columns and shape of the incoming data
    print("Incoming data columns:", data.columns)
    print("Shape of incoming data:", data.shape)
    
    # Drop any extra columns if the data has more than 13 columns
    expected_columns = 13
    if data.shape[1] > expected_columns:
        print("Extra columns detected. Dropping extra columns...")
        data = data.iloc[:, :expected_columns]  # Keep only the first 13 columns
    
    # Ensure the data has the expected number of columns
    if data.shape[1] != expected_columns:
        return jsonify({'error': f'Feature shape mismatch, expected: {expected_columns}, got: {data.shape[1]}'}), 400
    
    # Convert data to a format suitable for the model
    features = data.values  # Ensure shape matches the model
    
    # Initialize and fit the StandardScaler on the incoming data
    try:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)  # Fit the scaler and transform features
    except Exception as e:
        return jsonify({'error': f'Error scaling the data: {str(e)}'}), 500
    
    # Make predictions
    try:
        predictions = model.predict(scaled_features)
    except Exception as e:
        return jsonify({'error': f'Error making predictions: {str(e)}'}), 500
    
    # Calculate time taken
    processing_time = time.time() - start_time
    
    # Return predictions and time
    return jsonify({
        'predictions': predictions.tolist(),
        'time_taken': processing_time
    })

if __name__ == '__main__':
    app.run(debug=True)
