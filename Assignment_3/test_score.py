from sklearn.base import BaseEstimator
import joblib

def score(text:str, model:BaseEstimator, threshold:float) -> tuple[bool, float]:
    propensity = model.predict_proba([text])[0][1]
    if propensity >= threshold:
        return True,  propensity
    else:
        return False, propensity

model = joblib.load("models/best_model.pkl")

def test_score():
    # smoke test
    score("Let us meet sometime this week", model, 0.5)

    non_spam = "Hello, can we postpone the meeting."
    spam = "Congratulations! you won a lottery of $50000. Call 3425 now to claim your money."

    prediction1, propensity1 = score(non_spam, model, 0.5)
    prediction2, propensity2 = score(spam, model, 0.5)

    prediction3, propensity3 = score(non_spam, model, 0)
    prediction4, propensity4 = score(spam, model, 0)
    prediction5, propensity5 = score(non_spam, model, 1)
    prediction6, propensity6 = score(spam, model, 1)

    # format test
    assert isinstance(prediction1, bool)
    assert isinstance(propensity1, float)

    # sanity check
    assert prediction1 in [True, False]
    assert 0 <= propensity1 <= 1

    # typical input
    assert not prediction1
    assert prediction2

    # edge case input
    assert prediction3
    assert prediction4
    assert not prediction5
    assert not prediction6 