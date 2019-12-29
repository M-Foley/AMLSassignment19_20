from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

# Create a logistic regression model
def create_model():
    return LogisticRegression(max_iter=400, solver='liblinear')

# Train the logistic regression model with X and y, using girdsearch and cross validation
def train_model(model, X, y):
	penalty = ['l1', 'l2']
	C = np.logspace(0, 4, 10)
	hyperparams = dict(C=C, penalty=penalty)
	clf = GridSearchCV(model, hyperparams, cv=5, verbose=0)
	best_model = clf.fit(X, y)
	print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
	print('Best C:', best_model.best_estimator_.get_params()['C'])
	return best_model.score(X,y), best_model

# Test the model and retrieve the accuracy 
def test_model(model, X, y):
    predictions = model.predict(X)
    return model.score(X, y)