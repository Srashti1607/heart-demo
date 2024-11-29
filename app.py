from flask import Flask, request, jsonify
import pandas as pd
import time
import pickle
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the trained model
try:
    with open('xgboost.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    logging.error("Model file 'xgboost.pkl' not found.")
    model = None
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    model = None

@app.before_request
def log_request_info():
    """Log the request method and URL for debugging."""
    logging.info(f"Request method: {request.method}")
    logging.info(f"Request URL: {request.url}")


@app.route('/')
def home():
    """Home route to verify the app is running."""
    return "Application running", 200


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        total_start_time = time.time()  # Start time for the entire process

        # Check if a file is sent
        if 'file' not in request.files:
            raise ValueError("No file provided")

        file = request.files['file']
        
        # Load the file into a Pandas DataFrame
        try:
            data = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Error reading the CSV file: {str(e)}'}), 400
        
        # Expected number of columns (features)
        expected_columns = 13  # Adjust this to match your model's requirements
        
        # Automatically handle extra columns by selecting the first expected_columns
        if data.shape[1] > expected_columns:
            logging.warning(f"Uploaded data has {data.shape[1]} columns, trimming to {expected_columns}.")
            data = data.iloc[:, :expected_columns]

        # Validate if the data now has the correct number of columns
        if data.shape[1] != expected_columns:
            raise ValueError(f"Feature shape mismatch, expected: {expected_columns}, got: {data.shape[1]}")

        # Split the data into X (features)
        X = data.values

        # Initialize and fit the StandardScaler on the incoming data
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(X)  # Fit the scaler and transform features
        except Exception as e:
            return jsonify({'error': f'Error scaling the data: {str(e)}'}), 500

        # Model prediction
        try:
            predictions = model.predict(scaled_features)
        except Exception as e:
            return jsonify({'error': f'Error making predictions: {str(e)}'}), 500

        total_processing_time = time.time() - total_start_time

        return jsonify({
            'predictions': predictions.tolist(),
            'total_time': total_processing_time
        }), 200

    except Exception as e:
        # Log the exception and provide detailed feedback
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500



if __name__ == '__main__':
    if model:
        app.run(debug=True)  # Ensure to listen on all interfaces
    else:
        logging.error("Exiting: Model not loaded.")
