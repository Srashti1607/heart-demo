from flask import Flask, request, jsonify
import pandas as pd
import time
import pickle
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the trained model (make sure the model file is in the correct directory)
try:
    with open('xgboost.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'xgboost.pkl' not found.")
    model = None

@app.before_request
def log_request_info():
    """Log the request method and URL for debugging."""
    print(f"Request method: {request.method}")
    print(f"Request URL: {request.url}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        total_start_time = time.time()  # Start time for the entire process

        # Check if a file is sent
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        # Get the uploaded file
        file = request.files['file']

        # Load the file into a Pandas DataFrame
        try:
            data = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading the CSV file: {str(e)}")  # Log the error
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
            print(f"Feature shape mismatch, expected: {expected_columns}, got: {data.shape[1]}")  # Log this error
            return jsonify({'error': f'Feature shape mismatch, expected: {expected_columns}, got: {data.shape[1]}'}), 400

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Scale the data (fit on all rows first for consistency)
        try:
            scaled_data = scaler.fit_transform(data)
        except Exception as e:
            print(f"Error scaling the data: {str(e)}")  # Log the error
            return jsonify({'error': f'Error scaling the data: {str(e)}'}), 500

        predictions = []
        row_index = 0

        # While loop to process data row-by-row
        while row_index < scaled_data.shape[0]:
            try:
                # Extract a single row and reshape it for prediction
                row = scaled_data[row_index].reshape(1, -1)

                # Predict for the single row
                prediction = model.predict(row)

                # Append the prediction to the results
                predictions.append(int(prediction[0]))  # Ensure it's a Python native type
                time.sleep(0.1)

            except Exception as e:
                print(f"Error predicting row {row_index}: {str(e)}")  # Log the error
                return jsonify({'error': f'Error predicting row {row_index}: {str(e)}'}), 500

            # Move to the next row
            row_index += 1

        total_processing_time = time.time() - total_start_time  # Total time for all predictions

        # Return predictions and time information
        return jsonify({
            'predictions': predictions,
            'total_time': total_processing_time,
        }), 200

    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Log the general error
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500
