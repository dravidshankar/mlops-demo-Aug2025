import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Load the trained model
with open('lr_model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

def prepare_features(ride):
    return pd.DataFrame([{
        'trip_distance': ride['trip_distance'],
        'trip_duration': ride['trip_duration']
    }])

def predict(features_df):
    preds = model.predict(features_df)
    return float(preds[0])

app = Flask('fare-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    if ride is None:
        return jsonify({'error': 'Invalid or missing JSON in request'}), 400

    try:
        features_df = prepare_features(ride)
        pred = predict(features_df)

        result = {
            'predicted_fare': pred
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask on port 8000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
