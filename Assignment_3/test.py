import os
import time
import signal
import subprocess
import joblib
import requests
from score import score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")

model = joblib.load(MODEL_PATH)


def test_score():
    score("Let us meet sometime this week", model, 0.5)

    non_spam = "Hello, can we postpone the meeting."
    spam = "Congratulations! you won a lottery of $50000. Call 3425 now to claim your money."

    prediction1, propensity1 = score(non_spam, model, 0.5)
    prediction2, propensity2 = score(spam, model, 0.5)

    prediction3, propensity3 = score(non_spam, model, 0)
    prediction4, propensity4 = score(spam, model, 0)
    prediction5, propensity5 = score(non_spam, model, 1)
    prediction6, propensity6 = score(spam, model, 1)

    assert isinstance(prediction1, bool)
    assert isinstance(propensity1, float)

    assert prediction1 in [True, False]
    assert 0 <= propensity1 <= 1

    assert not prediction1
    assert prediction2

    assert prediction3
    assert prediction4
    assert not prediction5
    assert not prediction6


def test_flask():
    
    process = subprocess.Popen(
        ["uv", "run", "python", "app.py"],
        cwd=BASE_DIR,
        start_new_session=True,
    )

    try:
        url = "http://localhost:5000/score"

        for _ in range(20):
            try:
                requests.post(url, json={"text": "ping"})
                break
            except requests.exceptions.ConnectionError:
                time.sleep(1)

        response = requests.post(
            url,
            json={"text": "Congratulations! you won a lottery of $50000. Call 3425 now to claim your money."},
        )
        data = response.json()

        assert response.status_code == 200
        assert "prediction" in data
        assert "propensity" in data
        assert isinstance(data["prediction"], bool)
        assert 0 <= data["propensity"] <= 1
    finally:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()