from flask import Flask, request, jsonify
import joblib
from src.preprocessing import clean_text

# Load model & vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fake News Detection API is running 🚀"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' field"}), 400

    text = data["text"]
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    result = "FAKE" if prediction == 1 else "REAL"

    return jsonify({
        "input_text": text,
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)
