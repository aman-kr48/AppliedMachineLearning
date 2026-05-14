import os
import joblib
from flask import Flask, request, jsonify
from score import score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")

model = joblib.load(MODEL_PATH)

app = Flask(__name__)


@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json()
    text = data["text"]

    threshold = 0.5
    prediction, propensity = score(text, model, threshold)

    return jsonify({
        "prediction": bool(prediction),
        "propensity": float(propensity)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)