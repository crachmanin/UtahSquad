from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
import sklearn
import json
from itertools import groupby
import io

X_train, Y_train = load_svmlight_file("np_train_features.txt")
X_test, Y_test = load_svmlight_file("np_dev_features.txt", n_features=X_train.shape[1])

clf = LogisticRegression(random_state=0)
clf = clf.fit(X_train, Y_train)
probs = clf.predict_proba(X_test)
preds = clf.predict(X_test)
with open("preds.txt", "w") as fp:
    for pred in preds:
        fp.write(unicode(pred) + '\n')

zipped = []
with open("np_dev_ids.txt") as fp:
    for i, line in enumerate(fp):
        try:
            dev_id, np = unicode(line, "utf-8").strip().split('\t')
            zipped.append((dev_id, probs[i][1], np))
        except Exception as e:
            print line.strip().split('\t'), str(e)

groups = {}
for k, g in groupby(zipped, lambda x: x[0]):
    groups[k] = max(list(g), key=lambda x: x[1])

groups = {k: item[2] for k, item in groups.items()}
with io.open("preds.json", "w", encoding="utf8") as fp:
    data = json.dumps(groups, ensure_ascii=False)
    fp.write(unicode(data))
