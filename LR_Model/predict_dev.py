from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
import numpy as np
import sklearn
import os

X_train, Y_train = load_svmlight_file("train_features.txt")
X_test, Y_test = load_svmlight_file("dev_features.txt")

if not os.path.isfile("model.pkl"):
    clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=20, random_state=0)
    clf.fit(X_train, Y_train)
    joblib.dump(clf, 'model.pkl')
else:
    clf = joblib.load('model.pkl')

preds = clf.predict(X_test)

print sklearn.metrics.accuracy_score(Y_test, preds)

with open("predictions.txt", "w") as fp:
    for pred in preds:
        fp.write("%d\n" % pred)

