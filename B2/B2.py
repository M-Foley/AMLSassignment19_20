from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create the random forest classifier 
def create_model():
	return RandomForestClassifier(n_estimators=100, min_samples_split=2)

# Train the model wiht X and y, get the accuracy and then return that and the model
def train_model(model, X, y):
	model.fit(X, y)
	return model.score(X,y), model

# Test the model and return the accuracy
def test_model(model, X, y):
	return model.score(X, y)