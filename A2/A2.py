from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
def create_model():
	return svm.SVC(kernel='rbf')

def train_model(model, X, y):
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
	                     'C': [1, 10, 100, 1000]},
	                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

	scores = ['precision', 'recall']

	for score in scores:
	    print("# Tuning hyper-parameters for %s" % score)
	    print()

	    clf = GridSearchCV(
	        model, tuned_parameters, scoring='%s_macro' % score
	    )
	    clf.fit(X, y)

	    print("Best parameters set found on development set:")
	    print()
	    print(clf.best_params_)
	    print()
	    print("Grid scores on development set:")
	    print()
	    means = clf.cv_results_['mean_test_score']
	    stds = clf.cv_results_['std_test_score']
	    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	        print("%0.3f (+/-%0.03f) for %r"
	              % (mean, std * 2, params))
	    print()

	#C=[0.001, 0.01, 0.1, 1, 10]
	#gamma = [0.001, 0.01, 0.1, 1]
	#param_grid = {'C': C, 'gamma' : gamma}
	#best_model = GridSearchCV(model, param_grid, cv=5, return_train_score=True)
	#print('best model')
	#print(best_model)
	#best_model.fit(X, y)
	accuracy = clf.score(X,y)
	return accuracy, clf

def test_model(model, X, y):
	predictions = model.predict(X)
	return accuracy_score(predictions, y)