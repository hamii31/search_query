import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify

app = Flask(__name__)

model = load_model('rnn_model_v1.2.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

max_seq_len = 50 

@app.route('/ping')
def ping():
    return "pong",200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    query = data['query']

    # Tokenize and pad the input text
    seq = tokenizer.texts_to_sequences([query])
    padded = pad_sequences(seq, maxlen=max_seq_len, padding='post')

    preds = model.predict(padded)
    pred_class_idx = preds.argmax(axis=1)[0]
    pred_class_label = label_encoder.inverse_transform([pred_class_idx])[0]
    confidence = float(preds[0][pred_class_idx])

    return jsonify({
        'redirect_url': pred_class_label,
        'confidence': confidence
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
