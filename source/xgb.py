from preprocessing import *

from xgboost import XGBClassifier, DMatrix
from sklearn import metrics
from sklearn.metrics import classification_report


def _xgboost(tr_X, tr_y, te_X, te_y):
    print('[#] Start XGBoost ...')
    xgb = XGBClassifier(max_depth=5, objective='multi:softmax',
                        use_label_encoder=False, n_estimators=1000)

    pred_y = xgb.fit(tr_X.values, tr_y.values).predict(te_X.values)

    print(classification_report(te_y.values, pred_y))
    print('[*] Accuracy (XGBoost): ', metrics.accuracy_score(te_y.values, pred_y))


def do_XGBoost(tr_file, te_file) -> None:
    tr_X, te_X, tr_y, te_y = load_data(tr_file, te_file, label_encoder=True)
    _xgboost(tr_X, tr_y, te_X, te_y)


if __name__=='__main__':
    do_XGBoost('../data/all/ABFGI_train.csv', '../data/all/ABFGI_test.arff.csv')
