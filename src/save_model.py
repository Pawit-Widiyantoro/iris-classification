# train_model.py
from model import IrisClassifier
import pandas as pd
from sklearn.datasets import load_iris
import joblib

# Load the data
# Preprocess the data if needed
iris = load_iris()

# Convert the numpy array data to pandas dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Create new column for the target
df['species'] = iris.target

# Create an instance of the IrisClassifier
classifier = IrisClassifier()

# Load your data (X, y)
X = df.drop('species', axis=1)
y = df['species']

# Fit the model
classifier.fit(X, y)

# Save the model using joblib or pickle
joblib.dump(classifier, 'iris_model.pkl')
