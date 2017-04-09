from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
import sklearn
import sys
from sklearn.externals import joblib

X_train, Y_train = load_svmlight_file(sys.argv[1], n_features=(10 ** 8))

clf = LogisticRegression(random_state=0)
clf = clf.fit(X_train, Y_train)
joblib.dump(clf, 'model.pkl') 
