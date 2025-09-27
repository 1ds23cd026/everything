from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
@app.route('/')
def home():
    return "Welcome! Use the /predict endpoint with a POST request to get predictions."

# Load the saved model once when the app starts
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Assuming input is a JSON with a list of feature values:
        # e.g. {"features": [8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]}
        
        features = data.get('features')
        
        if features is None:
            return jsonify({'error': 'No features provided'}), 400
        
        # Convert features list to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(features_array)
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
