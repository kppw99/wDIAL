from preprocessing import *

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def _random_forest(tr_X, tr_y, te_X, te_y):
    print('[#] Start RandomForest Classifier')
    forest = RandomForestClassifier(n_estimators=100, verbose=1)
    forest.fit(tr_X, tr_y)
    pred_y = forest.predict(te_X)
    print(classification_report(te_y, pred_y))
    print('[*] Accuracy (RandomForest): ', metrics.accuracy_score(te_y, pred_y))


def do_RF(tr_file, te_file) -> None:
    tr_X, te_X, tr_y, te_y = load_data(tr_file, te_file)
    _random_forest(tr_X, tr_y, te_X, te_y)


if __name__=='__main__':
    do_RF('../data/ABFGI_train.csv', '../data/ABFGI_test.arff.csv')