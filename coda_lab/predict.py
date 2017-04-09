from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
import numpy as np
import sklearn
import json
from itertools import groupby
import io
import sys

X_train, Y_train = load_svmlight_file(sys.argv[1], n_features=(10 ** 8))
X_test, Y_test = load_svmlight_file(sys.argv[2], n_features=(10 ** 8))

clf = LogisticRegression(random_state=0)
clf = clf.fit(X_train, Y_train)
probs = clf.predict_proba(X_test)
preds = clf.predict(X_test)
# with open("preds.txt", "w") as fp:
    # for pred in preds:
        # fp.write(unicode(pred) + '\n')

zipped = []
with open(sys.argv[3]) as fp:
    for i, line in enumerate(fp):
        try:
            dev_id, np = unicode(line, "utf-8").strip().split('\t')
            zipped.append((dev_id, probs[i][1], np))
        except Exception as e:
            pass

groups = {}
for k, g in groupby(zipped, lambda x: x[0]):
    groups[k] = max(list(g), key=lambda x: x[1])

groups = {k: item[2] for k, item in groups.items()}
with io.open(sys.argv[4], "w", encoding="utf8") as fp:
    data = json.dumps(groups, ensure_ascii=False)
    fp.write(unicode(data))
