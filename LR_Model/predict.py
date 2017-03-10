from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
import numpy as np
import sklearn

TEST_SPLIT = .2

X, Y = load_svmlight_file("features.txt")

num_instances = len(Y)
num_test = int((1 - TEST_SPLIT) * num_instances)
indices = np.arange(num_instances)
np.random.shuffle(indices)

X = X[indices]
Y = Y[indices]

X_train = X[:num_test]
Y_train = Y[:num_test]
X_test = X[num_test:]
Y_test = Y[num_test:]

# print X_train.shape[0], X_test.shape[0]

clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=20, random_state=0)
preds = clf.fit(X_train, Y_train).predict(X_test)
print sklearn.metrics.accuracy_score(Y_test, preds)
