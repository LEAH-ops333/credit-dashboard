import numpy as np

class ManualVoting:
    """不重新训练"""
    def __init__(self, models_list):
        self.models_list = models_list

    def predict_proba(self, X):
        probs = np.array([model.predict_proba(X) for model in self.models_list])
        return np.mean(probs, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)