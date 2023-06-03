from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# load the model from disk
try:
    model = pickle.load(open('finalized_model.sav', 'rb'))
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        print("Received data:", data)
        reshaped_data = np.array(data).reshape(1, -1)
        print("Reshaped data:", reshaped_data)
        prediction = model.predict(reshaped_data)
        print("Prediction:", prediction)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        print("Error in predict function:", e)
        return jsonify({'error': 'Invalid data. Please check your input and try again.'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Server is running!'}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)