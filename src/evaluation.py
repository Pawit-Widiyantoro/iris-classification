import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from model import IrisClassifier
import joblib
import os

# Get the path to the current script
script_dir = os.path.dirname(__file__)

# Load the saved model
model_path = os.path.join(script_dir, '../model/iris_model.pkl')
loaded_classifier = joblib.load(model_path)

iris = load_iris()

# Convert the numpy array data to pandas dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Create new column for the target
df['species'] = iris.target

X = df.drop('species', axis=1)
y = df['species']

# Initialize the loaded classifier
classifier = loaded_classifier

# Cross-validation as k-times
k = 5
accuracy_score = cross_val_score(classifier.classifier, X, y, cv=k, scoring="accuracy")
precision_score = cross_val_score(classifier.classifier, X, y, cv=k, scoring="precision_macro")
recall_score = cross_val_score(classifier.classifier, X, y, cv=k, scoring="recall_macro")
f1_score = cross_val_score(classifier.classifier, X, y, scoring="f1_macro")

# Get the mean of all evaluation metrics
accuracy_mean = np.mean(accuracy_score)
precision_mean = np.mean(precision_score)
recall_mean = np.mean(recall_score)
f1_mean = np.mean(f1_score)

# Print all evaluation metrics
print(f"Accuracy score : {accuracy_mean:.2f}")
print(f"Precision score : {precision_mean:.2f}")
print(f"Recall score : {recall_mean:.2f}")
print(f"F1 score : {f1_mean:.2f}")

# Fit the loaded model (not necessary in this case)
# classifier.fit(X, y)

# Sample input for prediction
sample = [[5., 2.9, 0.9, 0.1]]

# Predict species
species_predictions = classifier.predict_species(sample)
print(f"Species prediction : {species_predictions}")
