# iris-classification
This repository contains a simple machine learning project about Iris flower classification using the scikit-learn library. The project demonstrates the Machine Learning Life Cycle process of data preprocessing, model training, cross-validation, and evaluation.
## Project Structure
- `notebook/`: Jupyter Notebook containing the step-by-step process of data exploration, model training, and evaluation.
- `src/`: Source code directory.
  - `model.py`: Python file defining the `IrisClassifier` class.
  - `save_model.py` : Python file to save the model to `model.pkl`.
  - `evaluation.py`: Python file to evaluate the model and print metrics.
- `model/`: Directory to store the final model.
  - `model.pkl`: Serialized model using joblib.
- `requirements.txt`: List of required Python packages.
## Usage
1. Clone the Repository.
2. Create Virtual Environment(optional).
3. Install the required dependencies in `requirements.txt`.
4. Explore the notebook:
  - Open the iris_classification.ipynb notebook in Jupyter Notebook.
  - Follow the instructions and explanations to understand the Iris Classification process.
5. Use the pre-trained model:
  - If you want to use the pre-trained model for predictions, load it using joblib or pickle in your own Python script.
6. Customize and contribute:
  - Feel free to customize the code, add features, and contribute to the project.
