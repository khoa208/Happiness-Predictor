from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle, os

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Serve the index.html file at the root URL
@app.route('/')
def index():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input JSON data
        input_data = request.json

        # Prepare input features
        features = [
            input_data['Happiness Rank'],
            input_data['Standard Error'],
            input_data['Economy (GDP per Capita)'],
            input_data['Family'],
            input_data['Health (Life Expectancy)'],
            input_data['Freedom'],
            input_data['Trust (Government Corruption)'],
            input_data['Generosity'],
            input_data['Dystopia Residual']
        ]

        # Scale the input features
        scaled_features = scaler.transform([features])

        # Make prediction
        prediction = model.predict(scaled_features)[0]
        rounded_prediction = round(prediction, 2)

        # Return the prediction
        return jsonify({'Happiness Score': rounded_prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

