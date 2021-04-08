from preprocessing import *

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def _svm(tr_X, tr_y, te_X, te_y):
    print('[#] Start SVM Classifier')
    svc = SVC(kernel='poly', C=1.0, verbose=1)
    svc.fit(tr_X, tr_y)
    pred_y = svc.predict(te_X)
    print(classification_report(te_y, pred_y))
    print('[*] Accuracy (SVM): ', metrics.accuracy_score(te_y, pred_y))


def do_SVM(tr_file, te_file) -> None:
    tr_X, te_X, tr_y, te_y = load_data(tr_file, te_file)
    _svm(tr_X, tr_y, te_X, te_y)


if __name__=='__main__':
    do_SVM('../data/ABFGI_train.csv', '../data/ABFGI_test.arff.csv')