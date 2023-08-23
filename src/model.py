import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class IrisClassifier:
    def __init__(self):
        self.classifier = LogisticRegression(max_iter=1000, solver='lbfgs')
        self.target_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict_species(self, samples):
        predictions = self.classifier.predict(samples)
        species_predictions = [self.target_mapping[prediction] for prediction in predictions]
        return species_predictions
