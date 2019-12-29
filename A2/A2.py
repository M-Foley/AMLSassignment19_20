from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Create an SVM model
def create_model():
	return svm.SVC()

# Train the SVM model with X and y, using girdsearch and cross validation
def train_model(model, X, y):
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.0001],
	                     'C': [1, 10, 100]},
	                    {'kernel': ['linear'], 'gamma': [1, 0.1, 0.0001], 'C': [1, 10, 100]}]

	clf = GridSearchCV(model, tuned_parameters, cv=5)
	best_model = clf.fit(X, y)
	accuracy = best_model.score(X,y)
	return accuracy, clf

# Test the model and retrieve the accuracy 
def test_model(model, X, y):
	predictions = model.predict(X)
	return accuracy_score(predictions, y)