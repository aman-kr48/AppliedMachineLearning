from sklearn.base import BaseEstimator

def score(text:str, model:BaseEstimator, threshold:float) -> tuple[bool, float]:
    propensity = model.predict_proba([text])[0][1]
    if propensity >= threshold:
        return True, propensity
    else:
        return False, propensity